"""A9 — Avaliação por sujeito na variante PRIMÁRIA (época final) → n = 10.

Por que este script existe. A ação A3 produziu métricas por sujeito a partir
dos checkpoints `best_*.pth`, que são selecionados por val mIoU — a variante
**secundária** do texto. A tabela principal da dissertação (Tab. 6.4) reporta a
variante **primária**, medida na última época do orçamento. Trocar o teste de
n=5 (por dobra, época final) por um de n=10 (por sujeito, melhor época)
misturaria as duas variantes: a comparação pareceria mais forte por ter mudado
de variante, não por ter mudado de unidade de pareamento.

Este script remove a ambiguidade avaliando os checkpoints de **época 99**
(0-indexado → a 100ª e última época), que existem para as 15 combinações
modelo×dobra em `outputs/<run>/checkpoints/`. Verificado: cada um carrega
`epoch == 99` e seu `val_ious[-1]` coincide com o `final_val_iou` do JSON de
métricas da dobra — é exatamente o estado que gerou a variante primária.

O resultado permite reportar n=10 **sem trocar de variante**: mesma métrica,
mesma época, apenas uma unidade de pareamento mais fina (sujeito em vez de
dobra), que é legítima porque cada um dos 10 sujeitos está em validação
exatamente uma vez ao longo das 5 dobras.

⚠️ Se os p-valores de época final NÃO atingirem 5%, o texto mantém n=5 e
reporta isso honestamente. A decisão é dos dados.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from _common import (
    MODELS,
    OUT_DIR,
    fold_subject_splits,
    get_device,
    load_config,
    load_fold_model_epoch,
    resolve_run_dir,
)
# Reutiliza a MESMA rotina de avaliação de A3 (autoridade única, R5): o que muda
# entre A3 e A9 é apenas de quais pesos o modelo é carregado.
from a03_per_subject_wilcoxon_best import evaluate_subject

FINAL_EPOCH = 99      # 0-indexado: a 100ª e última época do orçamento
BOOTSTRAP = 20000


def main() -> None:
    device = get_device()
    config = load_config()
    run_dir = resolve_run_dir(config)
    folds = fold_subject_splits(config)

    print(f"Checkpoints de época final ({FINAL_EPOCH}, 0-indexado) em: {run_dir}")
    for f in folds:
        print(f"  dobra {f['fold']}: val_subjects={f['val_subjects']}")

    subject_rows, per_image_rows = [], []
    for nome, chave in MODELS:
        for f in folds:
            fold = f["fold"]
            model = load_fold_model_epoch(chave, fold, FINAL_EPOCH, run_dir, config, device)
            for subject in f["val_subjects"]:
                sub_df = f["val_df"][f["val_df"]["dataset"] == subject]
                subj_r, per_image = evaluate_subject(model, sub_df, config, device)
                subject_rows.append({
                    "arquitetura": nome, "model_key": chave, "fold": fold,
                    "sujeito": subject, "n_imagens": len(sub_df),
                    "mIoU": subj_r["mean_iou"], "Dice": subj_r["mean_dice"],
                    "background_iou": subj_r["background_iou"],
                    "per_class_iou": subj_r["per_class_iou"],
                })
                for row in per_image:
                    row.update({"arquitetura": nome, "model_key": chave,
                                "fold": fold, "sujeito": subject})
                per_image_rows.extend(per_image)
                print(f"{nome:14} fold{fold} {subject}: mIoU={subj_r['mean_iou']:.4f} "
                      f"Dice={subj_r['mean_dice']:.4f} (n={len(sub_df)})")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    subj_df = pd.DataFrame(subject_rows)
    subj_path = OUT_DIR / "per_subject_metrics_final_epoch.csv"
    img_path = OUT_DIR / "per_image_metrics_final_epoch.csv"
    subj_df.to_csv(subj_path, index=False)
    pd.DataFrame(per_image_rows).to_csv(img_path, index=False)
    print(f"\nSalvo: {subj_path}\nSalvo: {img_path}")

    # Consistência com a Tab. 6.2: a média por sujeito ponderada pelo número de
    # imagens deve ficar próxima do final_val_iou por dobra. Reportada, não
    # assumida.
    print("\n=== Média por arquitetura (época final, agregada por sujeito) ===")
    print(subj_df.groupby("arquitetura")["mIoU"].agg(["mean", "std"]).to_string())

    print(f"\n=== Wilcoxon pareado por SUJEITO (n=10, variante primária) ===")
    piv = subj_df.pivot_table(index="sujeito", columns="arquitetura", values="mIoU")
    print(piv.to_string())

    rng = np.random.default_rng(42)
    rows = []
    for a, b in itertools.combinations(piv.columns, 2):
        d = (piv[a] - piv[b]).values
        p = wilcoxon(piv[a], piv[b], method="exact").pvalue
        bs = [rng.choice(d, len(d), replace=True).mean() for _ in range(BOOTSTRAP)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        wins = int((d > 0).sum())
        print(f"{a} - {b}: Δ={d.mean():+.4f}  vitórias={wins}/{len(d)}  "
              f"p={p:.4f}  IC95=[{lo:+.4f},{hi:+.4f}]  "
              f"{'SIGNIFICATIVO a 5%' if p < 0.05 else 'nao significativo a 5%'}")
        rows.append({"a": a, "b": b, "delta_mean": float(d.mean()), "wins_a": wins,
                     "n": len(d), "wilcoxon_p": float(p),
                     "ci95_lo": float(lo), "ci95_hi": float(hi),
                     "significant_5pct": bool(p < 0.05)})

    stats_path = OUT_DIR / "wilcoxon_per_subject_final_epoch.csv"
    pd.DataFrame(rows).to_csv(stats_path, index=False)
    print(f"\nSalvo: {stats_path}")
    print(f"\nPiso teórico do Wilcoxon exato bilateral: n=5 -> 2/2^5 = 0,0625 | "
          f"n=10 -> 2/2^10 = {2/2**10:.5f}")


if __name__ == "__main__":
    main()
