"""A5 — Estratificação por distância + pixels por classe (+ bônus Spearman).

Não havia rascunho para este script (pendente). Implementado a partir da
especificação do documento-fonte, usando ``per_image_metrics.csv`` (gerado
por `a03_per_subject_wilcoxon_best.py`, que já roda o mesmo modelo-do-fold-correto por amostra) em vez
de reavaliar os modelos — evita repetir ~24 mil forward passes.

(a) mIoU por faixa de distância — `Distance` vem dos CSVs brutos por sujeito
    (não existe em `data/processed/metadata.csv`), unido por `sample_id`.
(b) Pixels por classe — conta direto das máscaras processadas
    (`data/processed/masks/*.png`), que já são o rótulo rasterizado por
    classe (0..9), não os polígonos brutos.
(c) Bônus: correlação de Spearman entre IoU médio por classe e contagem de
    pixels da classe (excluindo o fundo, que domina a contagem e já é
    reportado à parte pela convenção de métrica do projeto — ver
    `codes/metrics.py`).
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _common import DATA_DIR, OUT_DIR, load_config, load_raw_subject_csv


def part_a_distance_stratification(config) -> None:
    img_metrics_path = OUT_DIR / "per_image_metrics_final_epoch.csv"
    if not img_metrics_path.exists():
        raise FileNotFoundError(f"{img_metrics_path} não existe — rode a9 primeiro.")
    df = pd.read_csv(img_metrics_path)

    subjects = sorted({sid.split("/")[0] for sid in df["sample_id"]})
    raw = pd.concat([load_raw_subject_csv(s) for s in subjects], ignore_index=True)
    raw = raw[["sample_id", "Distance"]]
    df = df.merge(raw, on="sample_id", how="left")

    df["faixa"] = pd.qcut(df["Distance"], 4, duplicates="drop")
    agg = df.groupby(["faixa", "arquitetura"], observed=True)["mIoU"].agg(["mean", "std", "count"])
    print("=== (a) mIoU por faixa de distância (n reportado — faixas com poucas\n"
          "    imagens não sustentam conclusão) ===")
    print(agg)
    out_path = OUT_DIR / "miou_by_distance.csv"
    agg.to_csv(out_path)
    print(f"Salvo: {out_path}")


def part_b_pixels_per_class(config) -> np.ndarray:
    mask_files = sorted((config.PROCESSED_DIR / "masks").glob("*.png"))
    if not mask_files:
        raise FileNotFoundError(f"Nenhuma máscara em {config.PROCESSED_DIR / 'masks'}")

    cont = np.zeros(config.NUM_CLASSES, dtype=np.int64)
    for f in mask_files:
        m = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        cont += np.bincount(m.ravel(), minlength=config.NUM_CLASSES)[:config.NUM_CLASSES]

    pixel_counts = {config.REGION_NAMES[i]: int(cont[i]) for i in range(config.NUM_CLASSES)}
    print("\n=== (b) Pixels por classe (soma sobre todas as máscaras processadas) ===")
    for name, n in pixel_counts.items():
        print(f"  {name:28} {n:>12,}")
    out_path = OUT_DIR / "pixels_per_class.csv"
    pd.Series(pixel_counts, name="pixel_count").to_csv(out_path, index_label="classe")
    print(f"Salvo: {out_path}")
    return cont


def part_c_spearman_bonus(config, pixel_counts: np.ndarray) -> None:
    subj_path = OUT_DIR / "per_subject_metrics_final_epoch.csv"
    if not subj_path.exists():
        print("\n(per_subject_metrics.csv ausente — rode a9 primeiro; pulando bônus Spearman)")
        return
    subj_df = pd.read_csv(subj_path)
    # per_class_iou round-trips through CSV as e.g. "[0.85, nan, 0.72, ...]" —
    # SegmentationMetrics.compute() legitimately emits NaN for a class absent
    # from a subject's data (see codes/metrics.py docstring), and bare `nan`
    # is not a valid ast.literal_eval literal, so parse as JSON with the
    # token case-fixed instead (json.loads accepts "NaN", not lowercase).
    per_class = np.stack(
        subj_df["per_class_iou"].apply(lambda s: json.loads(s.replace("nan", "NaN"))).values
    )
    mean_class_iou = np.nanmean(per_class, axis=0)  # shape (num_classes,)

    # Exclui o fundo (índice 0): domina a contagem de pixels e já é
    # reportado separadamente pela convenção de métrica do projeto.
    classes = np.arange(1, config.NUM_CLASSES)
    rho, p = spearmanr(mean_class_iou[classes], pixel_counts[classes])
    print(f"\n=== (c) Bônus — Spearman(IoU médio por classe, pixels por classe), "
          f"excl. fundo ===\nrho={rho:.4f}  p={p:.4f}")
    if p < 0.05 and abs(rho) > 0.5:
        print("  Correlação forte e significativa: a área da classe explica boa "
              "parte da variância de IoU entre classes — reportar isso em vez de "
              "invocar fisiologia térmica sem essa checagem (ver §6.5 no documento-fonte).")
    for i, name in zip(classes, [config.REGION_NAMES[i] for i in classes]):
        print(f"  {name:28} IoU_medio={mean_class_iou[i]:.4f}  pixels={pixel_counts[i]:,}")


def main() -> None:
    config = load_config()
    part_a_distance_stratification(config)
    pixel_counts = part_b_pixels_per_class(config)
    part_c_spearman_bonus(config, pixel_counts)


if __name__ == "__main__":
    main()
