"""A3 — Avaliação por sujeito (n=10) + teste pareado de Wilcoxon.

Bugs do rascunho original (ver outputs.txt: crash no `import cv2`):

1. O crash real foi de AMBIENTE: o traceback roda em
   ``(base) doga@dogachine`` (conda base, Python 3.13), cujo
   ``opencv-python`` está linkado contra uma ``libtiff`` do sistema sem o
   símbolo ``jpeg12_write_raw_data``. O ambiente do próprio projeto
   (``.venv``, uv-managed, Python 3.11) tem a stack correta — foi só
   testado com ``.venv/bin/python`` neste script e ``cv2`` importa sem
   problema. **Rode sempre com `.venv/bin/python`, nunca com o Python do
   conda base.**
2. ``sys.path.insert(0, '../codes/')`` + imports soltos (``from
   model_registry import ...``) quebram os imports absolutos ``codes.``
   introduzidos em T3.6/UB-21 — mesmo problema do a02. Ver
   ``data/_common.py`` para o bootstrap correto (raiz do repo no
   ``sys.path``, import via ``codes.``).
3. ``SUJEITOS_POR_FOLD[fold]`` era uma variável indefinida com um
   comentário apontando para "Tabela 17 da dissertação" — mas a divisão
   por sujeito (GroupKFold, groups=dataset, sem shuffle/random_state) é
   uma função determinística de (ordem de metadata.csv, K). Reproduzi-la
   chamando o MESMO caminho de código do treino
   (``codes.unified_data.load_split_metadata`` +
   ``sklearn.model_selection.GroupKFold``, ver
   ``_common.fold_subject_splits``) é mais confiável do que copiar uma
   tabela à mão — exatamente a classe de bug (duas fontes da verdade que
   divergem) que UB-02 já mostrou ser real neste projeto.
4. O nome de checkpoint era montado com
   ``chave.replace('_unet_plus_plus', '')`` — confuso e desnecessário; a
   chave do registry (``swin_unet_plus_plus``) e o nome do arquivo de
   checkpoint são a MESMA string (ver ``codes/naming.py``); usar
   ``_common.load_fold_model`` remove a necessidade de montar o path à
   mão.

Também: os checkpoints `best_*` são selecionados por **val mIoU** (T3.3),
não por loss — a métrica primária do dataframe, não a secundária.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader

from _common import MODELS, OUT_DIR, fold_subject_splits, get_device, load_config, load_fold_model, resolve_run_dir

BATCH_SIZE = 8
NUM_WORKERS = 2


def evaluate_subject(model, sub_df: pd.DataFrame, config, device):
    """Score one subject's images with one fold-model.

    Returns the subject-level aggregate (confusion-matrix accumulated across
    all of the subject's images — the technically correct aggregate, matching
    how the trainer computes a fold's validation mIoU) AND a per-image row
    list (needed downstream by A4's failure-case selection and A5's
    distance-stratified analysis) computed from the SAME forward pass, so no
    image is scored twice.
    """
    from codes.unified_data import ThermalFaceDataset
    from codes.metrics import SegmentationMetrics, compute_segmentation_metrics

    ds = ThermalFaceDataset(sub_df, config, augment=False)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    subj_metric = SegmentationMetrics(num_classes=config.NUM_CLASSES, device=device)
    per_image_rows = []

    with torch.no_grad():
        for imgs, masks, sample_ids in dl:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            subj_metric.update(logits, masks)
            for i, sid in enumerate(sample_ids):
                r = compute_segmentation_metrics(logits[i:i + 1], masks[i:i + 1], config.NUM_CLASSES)
                per_image_rows.append({"sample_id": sid, "mIoU": r["mean_iou"], "Dice": r["mean_dice"]})

    return subj_metric.compute(), per_image_rows


def main() -> None:
    device = get_device()
    config = load_config()
    run_dir = resolve_run_dir(config)
    print(f"Avaliando checkpoints em: {run_dir}")

    folds = fold_subject_splits(config)
    print(f"Folds reproduzidos (leave-subjects-out GroupKFold, K={len(folds)}):")
    for f in folds:
        print(f"  fold {f['fold']}: val_subjects={f['val_subjects']}")
    n_val_subjects = sum(len(f["val_subjects"]) for f in folds)
    print(f"Total de sujeitos avaliados (cada um em exatamente 1 fold): {n_val_subjects}")

    subject_rows = []
    per_image_rows_all = []

    for nome, chave in MODELS:
        for f in folds:
            fold = f["fold"]
            model = load_fold_model(chave, fold, run_dir, config, device).eval()
            for subject in f["val_subjects"]:
                sub_df = f["val_df"][f["val_df"]["dataset"] == subject]
                subj_r, per_image = evaluate_subject(model, sub_df, config, device)
                subject_rows.append({
                    "arquitetura": nome,
                    "model_key": chave,
                    "fold": fold,
                    "sujeito": subject,
                    "n_imagens": len(sub_df),
                    "mIoU": subj_r["mean_iou"],
                    "Dice": subj_r["mean_dice"],
                    "background_iou": subj_r["background_iou"],
                    "per_class_iou": subj_r["per_class_iou"],
                })
                for row in per_image:
                    row.update({"arquitetura": nome, "model_key": chave, "fold": fold, "sujeito": subject})
                per_image_rows_all.extend(per_image)
                print(f"{nome:14} fold{fold} {subject}: mIoU={subj_r['mean_iou']:.4f} "
                      f"Dice={subj_r['mean_dice']:.4f} (n={len(sub_df)})")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    subj_df = pd.DataFrame(subject_rows)
    img_df = pd.DataFrame(per_image_rows_all)
    subj_path = OUT_DIR / "per_subject_metrics.csv"
    img_path = OUT_DIR / "per_image_metrics.csv"
    subj_df.to_csv(subj_path, index=False)
    img_df.to_csv(img_path, index=False)
    print(f"\nSalvo: {subj_path}")
    print(f"Salvo: {img_path}")
    print(
        "\n⚠️  Nota: estes números vêm dos checkpoints de MELHOR época "
        "(selecionados por val mIoU, T3.3/UB-18) — a coluna PRIMÁRIA do "
        "benchmark_comparison.csv, não a secundária (val da última época)."
    )

    print(f"\n=== Teste pareado de Wilcoxon (n={n_val_subjects} sujeitos, por arquitetura) ===")
    piv = subj_df.pivot_table(index="sujeito", columns="arquitetura", values="mIoU")
    print(piv)

    rng = np.random.default_rng(42)
    stats_rows = []
    for a, b in itertools.combinations(piv.columns, 2):
        d = (piv[a] - piv[b]).values
        if np.allclose(d, 0):
            print(f"{a} - {b}: diferenças todas zero, Wilcoxon indefinido")
            continue
        p = wilcoxon(piv[a], piv[b], method="exact").pvalue
        bs = [rng.choice(d, len(d), replace=True).mean() for _ in range(20000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        vitorias = int((d > 0).sum())
        print(f"{a} - {b}: Δ={d.mean():+.4f}  vitórias={vitorias}/{len(d)}  "
              f"p={p:.4f}  IC95=[{lo:+.4f},{hi:+.4f}]")
        stats_rows.append({
            "a": a, "b": b, "delta_mean": float(d.mean()), "wins_a": vitorias,
            "n": len(d), "wilcoxon_p": float(p), "ci95_lo": float(lo), "ci95_hi": float(hi),
        })

    stats_path = OUT_DIR / "wilcoxon_per_subject.csv"
    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)
    print(f"\nSalvo: {stats_path}")


if __name__ == "__main__":
    main()
