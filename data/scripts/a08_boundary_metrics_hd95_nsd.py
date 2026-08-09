"""A8 — HD95 e NSD (métricas de fronteira) — CORTÁVEL.

Não havia rascunho para este script (pendente). O documento-fonte já marca
esta ação como cortável se o tempo não fechar, e instrui: se não couber,
"não force — declare em §7.5 como limitação [...]. Limitação declarada é
aceitável; número inventado não é."

Este script segue essa instrução ao pé da letra: MONAI não está instalado
no ambiente do projeto (`.venv/bin/python -c "import monai"` falha aqui
com `ModuleNotFoundError`), e o disco tem só ~6 GB livres neste momento —
não instalei a dependência sem confirmação (poderia empurrar o disco para
100% em um pacote com várias dependências pesadas). Rode
``pip install monai`` manualmente quando quiser gerar estes números; até
lá, o script termina cedo com uma mensagem clara em vez de fabricar
valores ou travar com um traceback confuso.

Quando MONAI estiver disponível: mesma varredura de A3 (modelo do fold
correto por sujeito), mas em vez de IoU/Dice usa
``HausdorffDistanceMetric(percentile=95)`` e ``SurfaceDiceMetric``. NSD é
calculada com o limiar em PIXELS (não há calibração pixel→mm nos dados
brutos disponíveis nesta sessão) — reportado explicitamente como tal, para
não implicar uma precisão física que não existe.
"""

from __future__ import annotations

import sys

try:
    import monai  # noqa: F401
except ImportError:
    print(
        "MONAI não está instalado neste ambiente (.venv). Esta é uma "
        "limitação declarada (ação A8 é cortável — ver §7.5), não um "
        "número fabricado.\n\n"
        "Para gerar HD95/NSD: `.venv/bin/pip install monai` e rode de novo. "
        "Isso vai baixar dependências adicionais — confira o espaço em "
        "disco primeiro (`df -h`)."
    )
    sys.exit(0)

import numpy as np
import pandas as pd
import torch
from monai.metrics import HausdorffDistanceMetric, SurfaceDiceMetric
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader

from _common import MODELS, OUT_DIR, fold_subject_splits, get_device, load_config, load_fold_model, resolve_run_dir

BATCH_SIZE = 8
NUM_WORKERS = 2
# Limiar de NSD em PIXELS (não em mm — sem calibração física disponível).
NSD_THRESHOLD_PX = 1.0


def to_onehot(t: torch.Tensor, num_classes: int) -> torch.Tensor:
    return one_hot(t, num_classes).permute(0, 3, 1, 2).float()


def evaluate_subject_boundary(model, sub_df, config, device):
    from codes.unified_data import ThermalFaceDataset

    ds = ThermalFaceDataset(sub_df, config, augment=False)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    hd95 = HausdorffDistanceMetric(percentile=95, include_background=False, reduction="none")
    nsd = SurfaceDiceMetric(
        class_thresholds=[NSD_THRESHOLD_PX] * (config.NUM_CLASSES - 1),
        include_background=False, reduction="none",
    )

    with torch.no_grad():
        for imgs, masks, _sids in dl:
            imgs = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs).argmax(1)
            pred_oh = to_onehot(preds, config.NUM_CLASSES)
            gt_oh = to_onehot(masks, config.NUM_CLASSES)
            hd95(pred_oh, gt_oh)
            nsd(pred_oh, gt_oh)

    hd95_per_class = hd95.aggregate(reduction="mean_batch").cpu().numpy()
    nsd_per_class = nsd.aggregate(reduction="mean_batch").cpu().numpy()
    return {
        "hd95_mean_px": float(np.nanmean(hd95_per_class)),
        "nsd_mean": float(np.nanmean(nsd_per_class)),
    }


def main() -> None:
    device = get_device()
    config = load_config()
    run_dir = resolve_run_dir(config)
    folds = fold_subject_splits(config)

    rows = []
    for nome, chave in MODELS:
        for f in folds:
            model = load_fold_model(chave, f["fold"], run_dir, config, device).eval()
            for subject in f["val_subjects"]:
                sub_df = f["val_df"][f["val_df"]["dataset"] == subject]
                r = evaluate_subject_boundary(model, sub_df, config, device)
                r.update({"arquitetura": nome, "fold": f["fold"], "sujeito": subject})
                rows.append(r)
                print(f"{nome:14} fold{f['fold']} {subject}: "
                      f"HD95={r['hd95_mean_px']:.2f}px  NSD@{NSD_THRESHOLD_PX}px={r['nsd_mean']:.4f}")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "boundary_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSalvo: {out_path}")
    print(df.groupby("arquitetura")[["hd95_mean_px", "nsd_mean"]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
