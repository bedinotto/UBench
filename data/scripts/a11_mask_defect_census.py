"""A11 — Censo completo do defeito de derivação de máscaras (UB-27).

A análise que descobriu o defeito usou uma amostra de 400 imagens por grupo.
Números citáveis na dissertação precisam do conjunto **completo** — as 8074
imagens de `data/processed/metadata.csv`. Este script mede, sem GPU e sem tocar
nos dados:

(a) **Contagem de pontos por região no JSON de polígonos**, separada por grupo
    (perfil = 43 marcos preenchidos, frontal = 73). É a evidência direta do
    mecanismo: sob o mapeamento de 73 aplicado a uma linha de perfil, as regiões
    cujos índices são todos > 42 não recebem ponto algum.

(b) **Presença e área por classe nas máscaras rasterizadas**, por grupo. É a
    consequência medida: quantas imagens de fato têm cada classe na referência,
    e com quantos pixels.

(c) **Proporção perfil/frontal** restrita às imagens efetivamente processadas
    (A1 contou as linhas brutas dos CSVs, que incluem duplicatas de `ID`).

Saídas: `mask_defect_census.csv` (por classe × grupo) e
`mask_defect_summary.json` (os números que o texto cita).
"""

from __future__ import annotations

import json
from collections import Counter

import cv2
import numpy as np
import pandas as pd

from _common import DATA_DIR, OUT_DIR, landmark_files, landmark_xy_columns, load_config

PROFILE_POINTS = 43
FRONTAL_POINTS = 73


def build_bucket_index() -> pd.DataFrame:
    """sample_id → bucket (43/73), deduplicado, para as imagens processadas.

    Os CSVs brutos contêm `ID` repetidos (245 linhas, 210 delas em S2); o
    pré-processamento mantém um `sample_id` por imagem. A deduplicação aqui
    reproduz esse comportamento para que as proporções batam com o conjunto
    realmente treinado.
    """
    frames = []
    for path in landmark_files():
        df = pd.read_csv(path)
        xy = landmark_xy_columns(df)
        filled = df[xy].notna().sum(axis=1) // 2
        subject = path.stem
        frames.append(pd.DataFrame({
            "sample_id": subject + "/" + df["ID"].astype(str),
            "n_filled": filled,
        }))
    lm = pd.concat(frames, ignore_index=True)
    dup_total = int(lm.duplicated("sample_id").sum())
    lm = lm.drop_duplicates("sample_id")
    lm["bucket"] = np.where(lm["n_filled"] >= (PROFILE_POINTS + FRONTAL_POINTS) / 2,
                            FRONTAL_POINTS, PROFILE_POINTS)
    return lm, dup_total


def census_polygons(bucket_of: dict) -> pd.DataFrame:
    """(a) pontos por região no JSON de polígonos, por grupo."""
    rows = []
    for path in sorted(DATA_DIR.glob("S*_polygonal_masks.json")):
        subject = path.stem.replace("_polygonal_masks", "")
        with open(path, encoding="utf-8") as fh:
            polygons = json.load(fh)
        for img_id, regions in polygons.items():
            bucket = bucket_of.get(f"{subject}/{img_id}")
            if bucket is None:
                continue          # imagem não presente no conjunto processado
            for region, pts in regions.items():
                rows.append({"bucket": bucket, "regiao": region, "n_pontos": len(pts)})
    return pd.DataFrame(rows)


def census_masks(config, meta: pd.DataFrame) -> tuple[dict, dict]:
    """(b) presença e área por classe nas máscaras rasterizadas, por grupo."""
    n_classes = config.NUM_CLASSES
    present = {PROFILE_POINTS: np.zeros(n_classes), FRONTAL_POINTS: np.zeros(n_classes)}
    area = {PROFILE_POINTS: np.zeros(n_classes), FRONTAL_POINTS: np.zeros(n_classes)}
    counts = Counter()

    for bucket, sub in meta.groupby("bucket"):
        for path in sub["mask_path"]:
            mask = cv2.imread(str(config.DATA_DIR.parent / path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise FileNotFoundError(f"máscara ausente: {path}")
            counts[bucket] += 1
            cnt = np.bincount(mask.ravel(), minlength=n_classes)[:n_classes]
            present[bucket] += (cnt > 0)
            area[bucket] += cnt
    return {"present": present, "area": area}, dict(counts)


def main() -> None:
    config = load_config()
    lm, dup_total = build_bucket_index()
    meta = pd.read_csv(config.PROCESSED_DIR / "metadata.csv")
    meta = meta.merge(lm[["sample_id", "bucket"]], on="sample_id", how="left")

    unmatched = int(meta["bucket"].isna().sum())
    if unmatched:
        raise ValueError(
            f"{unmatched} imagens processadas sem linha de marcos correspondente — "
            f"o censo seria parcial."
        )
    meta["bucket"] = meta["bucket"].astype(int)

    n_total = len(meta)
    n_profile = int((meta["bucket"] == PROFILE_POINTS).sum())
    n_frontal = int((meta["bucket"] == FRONTAL_POINTS).sum())
    print(f"Imagens processadas: {n_total}  "
          f"(perfil {n_profile} = {100*n_profile/n_total:.1f}% | "
          f"frontal {n_frontal} = {100*n_frontal/n_total:.1f}%)")
    print(f"Linhas brutas com sample_id duplicado, descartadas: {dup_total}")

    print("\n=== (a) pontos por região no JSON de polígonos ===")
    poly = census_polygons(dict(zip(meta["sample_id"], meta["bucket"])))
    pts = poly.groupby(["regiao", "bucket"])["n_pontos"].agg(["min", "max", "mean", "size"])
    print(pts.to_string())

    print("\n=== (b) presença e área por classe nas máscaras ===")
    stats, counts = census_masks(config, meta)
    rows = []
    print(f"{'classe':<28}{'pres% frontal':>14}{'pres% perfil':>14}"
          f"{'px/img frontal':>16}{'px/img perfil':>15}")
    for i, name in enumerate(config.REGION_NAMES):
        pf = 100 * stats["present"][FRONTAL_POINTS][i] / counts[FRONTAL_POINTS]
        pp = 100 * stats["present"][PROFILE_POINTS][i] / counts[PROFILE_POINTS]
        af = stats["area"][FRONTAL_POINTS][i] / counts[FRONTAL_POINTS]
        ap = stats["area"][PROFILE_POINTS][i] / counts[PROFILE_POINTS]
        print(f"{name:<28}{pf:>13.1f}%{pp:>13.1f}%{af:>16.0f}{ap:>15.0f}")
        rows.append({"classe": name, "class_index": i,
                     "presenca_pct_frontal": pf, "presenca_pct_perfil": pp,
                     "px_por_img_frontal": af, "px_por_img_perfil": ap,
                     "px_total_frontal": int(stats["area"][FRONTAL_POINTS][i]),
                     "px_total_perfil": int(stats["area"][PROFILE_POINTS][i])})

    census_path = OUT_DIR / "mask_defect_census.csv"
    pd.DataFrame(rows).to_csv(census_path, index=False)

    summary = {
        "n_imagens_processadas": n_total,
        "n_perfil_43": n_profile, "pct_perfil_43": 100 * n_profile / n_total,
        "n_frontal_73": n_frontal, "pct_frontal_73": 100 * n_frontal / n_total,
        "linhas_brutas_duplicadas_descartadas": dup_total,
        "n_mascaras_lidas": {str(k): int(v) for k, v in counts.items()},
        "classes_ausentes_em_perfil": [
            r["classe"] for r in rows if r["presenca_pct_perfil"] < 1.0
        ],
    }
    summary_path = OUT_DIR / "mask_defect_summary.json"
    json.dump(summary, open(summary_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nClasses efetivamente ausentes em perfil (<1% de presença): "
          f"{summary['classes_ausentes_em_perfil']}")
    print(f"Salvo: {census_path}\nSalvo: {summary_path}")


if __name__ == "__main__":
    main()
