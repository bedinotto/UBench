"""A4 — Grade visual de inferência (qualitativa) + figura da anomalia lateral.

O rascunho original era um esqueleto com placeholders nunca implementados
(``AMOSTRAS = [...]``, ``carregar_tres_modelos_do_fold_correspondente()``,
``carregar_amostra()``, ``preparar()`` — nenhum definido, daí o
``NameError`` em outputs.txt). Este script implementa cada peça:

* ``carregar_amostra`` → lê ``data/processed/images/*.npy`` (Celsius) e
  ``data/processed/masks/*.png`` (rótulo inteiro) via `metadata.csv`.
* ``preparar`` → mesma normalização de carga usada no treino
  (``codes.utils.apply_normalization`` sobre ``config.PREPROCESSING``).
* ``carregar_tres_modelos_do_fold_correspondente`` → para CADA amostra
  usa-se o modelo do fold em que o SUJEITO daquela amostra estava em
  VALIDAÇÃO (nunca o modelo que viu o sujeito em treino — é exatamente o
  aviso do rascunho: "usar um modelo que viu o sujeito em treino invalida
  a figura"). O fold de cada sujeito vem de
  ``_common.fold_subject_splits`` (mesmo split do treino, não uma tabela
  copiada à mão).

Seleção de amostras (Tabela do documento-fonte, condição → critério
mecânico usado aqui):

    1 Frontal, curta distância        → menor `Distance` entre imagens frontais (73 marcos)
    2 Frontal, longa distância        → maior `Distance` entre imagens frontais
    3 Temp. ambiente baixa            → menor `env-temp`
    4 Temp. ambiente alta             → maior `env-temp`
    5 Perfil (43 marcos)              → amostra de mIoU mediana entre as de 43 marcos
                                         (evita tanto o melhor quanto o pior caso de perfil)
    6 Cabeça inclinada                → **não há coluna de pose de cabeça verificável nos
                                         CSVs brutos** — não inventado (R10/R4). Ver
                                         MANUAL_OVERRIDE abaixo: preencha 'cabeca_inclinada'
                                         com um sample_id escolhido visualmente.
    7 Caso de falha clara             → menor mIoU médio (entre as 3 arquiteturas) do
                                         conjunto de validação inteiro
    8 Sujeito de fold difícil (2 ou 3)→ amostra de mIoU mediana entre sujeitos do fold 2/3

Qualquer entrada pode ser sobrescrita manualmente em MANUAL_OVERRIDE — é lá
que fica o julgamento visual que o documento-fonte pede ("é onde está o
valor, não no script").

Pré-requisito: rode `a01_landmark_scheme_census.py` e `a03_per_subject_wilcoxon_best.py` antes (usa seus CSVs de saída).
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from _common import IMG_DIR, MODELS, OUT_DIR, fold_subject_splits, get_device, load_config, load_raw_subject_csv, \
    resolve_run_dir, load_fold_model_epoch

CLASSES = ['Fundo', 'Contorno', 'Sobr. esq.', 'Sobr. dir.', 'Nariz',
           'Olho esq.', 'Olho dir.', 'Boca', 'Lábios', 'Testa']
CORES = ['#00000000', '#8dd3c7', '#ffffb3', '#bebada', '#fb8072',
         '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9']
CMAP = ListedColormap(CORES)
NORM = BoundaryNorm(range(11), 10)

# Preencha manualmente (sample_id, ex. 'S3/R211045') para sobrepor a seleção
# automática de qualquer linha — em especial 'cabeca_inclinada', que o
# script não consegue escolher sozinho.
MANUAL_OVERRIDE: dict[str, str | None] = {
    "frontal_curta_distancia": None,
    "frontal_longa_distancia": None,
    "temp_baixa": None,
    "temp_alta": None,
    "perfil_43_marcos": None,
    "cabeca_inclinada": "S4/R411826",  # OBRIGATÓRIO escolher visualmente — ver docstring
    "caso_falha": None,
    "fold_dificil": None,
}


def load_sample(sample_id: str, metadata: pd.DataFrame, config):
    row = metadata.loc[metadata["sample_id"] == sample_id].iloc[0]
    img_path = config.DATA_DIR.parent / row["image_path"]
    mask_path = config.DATA_DIR.parent / row["mask_path"]
    thermal_c = np.load(img_path).astype(np.float32)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED).astype(np.int64)
    return thermal_c, mask


def to_model_input(thermal_c: np.ndarray, config, device) -> torch.Tensor:
    from codes.utils import apply_normalization
    img = apply_normalization(thermal_c, config.PREPROCESSING.normalization,
                              config.PREPROCESSING.fixed_range_celsius)
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)


def build_candidate_table(config) -> tuple[pd.DataFrame, dict]:
    """Assemble one row per (sample_id) with mIoU (per arch + mean), Distance,
    env-temp, landmark bucket, subject, and the fold whose model must be used.
    """
    img_metrics_path = OUT_DIR / "per_image_metrics_final_epoch.csv"
    landmarks_path = OUT_DIR / "landmark_classification.csv"
    if not img_metrics_path.exists():
        raise FileNotFoundError(f"{img_metrics_path} não existe — rode a9 primeiro.")
    if not landmarks_path.exists():
        raise FileNotFoundError(f"{landmarks_path} não existe — rode a01_landmark_scheme_census.py primeiro.")

    img_metrics = pd.read_csv(img_metrics_path)
    piv = img_metrics.pivot_table(index="sample_id", columns="arquitetura", values="mIoU")
    piv["mIoU_medio"] = piv.mean(axis=1)

    landmarks = pd.read_csv(landmarks_path)[["sample_id", "landmark_bucket"]]

    raw = pd.concat([load_raw_subject_csv(s) for s in sorted({sid.split("/")[0] for sid in piv.index})],
                    ignore_index=True)
    raw = raw[["sample_id", "Distance", "env-temp"]]

    fold_lookup = {}
    for f in fold_subject_splits(config):
        for subj in f["val_subjects"]:
            fold_lookup[subj] = f["fold"]

    table = piv.reset_index().merge(landmarks, on="sample_id", how="left").merge(raw, on="sample_id", how="left")
    table["subject"] = table["sample_id"].str.split("/").str[0]
    table["fold"] = table["subject"].map(fold_lookup)
    table = table.dropna(subset=["fold"])
    table["fold"] = table["fold"].astype(int)
    return table, fold_lookup


def auto_select(table: pd.DataFrame) -> dict[str, str]:
    frontal = table[table["landmark_bucket"] == 73]
    profile = table[table["landmark_bucket"] == 43]

    def median_row(df: pd.DataFrame) -> str:
        return df.iloc[(df["mIoU_medio"] - df["mIoU_medio"].median()).abs().argsort().iloc[0]]["sample_id"]

    picks = {
        "frontal_curta_distancia": frontal.loc[frontal["Distance"].idxmin(), "sample_id"],
        "frontal_longa_distancia": frontal.loc[frontal["Distance"].idxmax(), "sample_id"],
        "temp_baixa": table.loc[table["env-temp"].idxmin(), "sample_id"],
        "temp_alta": table.loc[table["env-temp"].idxmax(), "sample_id"],
        "perfil_43_marcos": median_row(profile) if len(profile) else None,
        "cabeca_inclinada": None,  # não determinável mecanicamente — ver MANUAL_OVERRIDE
        "caso_falha": table.loc[table["mIoU_medio"].idxmin(), "sample_id"],
        "fold_dificil": median_row(table[table["fold"].isin([2, 3])]) if (table["fold"].isin([2, 3])).any() else None,
    }
    return picks


def main() -> None:
    device = get_device()
    config = load_config()
    run_dir = resolve_run_dir(config)
    metadata = pd.read_csv(config.PROCESSED_DIR / "metadata.csv")

    table, fold_lookup = build_candidate_table(config)
    picks = auto_select(table)
    for slot, override in MANUAL_OVERRIDE.items():
        if override:
            picks[slot] = override

    print("Seleção de amostras (sobrescreva em MANUAL_OVERRIDE se necessário):")
    missing = []
    for slot, sid in picks.items():
        print(f"  {slot:26} -> {sid}")
        if sid is None:
            missing.append(slot)
    if missing:
        print(f"\n⚠️  Sem seleção automática para: {missing}. Preencha "
              f"MANUAL_OVERRIDE com um sample_id escolhido visualmente antes "
              f"de gerar a figura final.")

    amostras = [(slot, sid) for slot, sid in picks.items() if sid is not None]
    if not amostras:
        print("Nenhuma amostra selecionada — nada a plotar.")
        return

    # Carrega, por fold necessário, os 3 modelos (uma vez cada).
    needed_folds = sorted({fold_lookup[sid.split("/")[0]] for _, sid in amostras})
    models_by_fold: dict[int, dict[str, torch.nn.Module]] = {}
    for fold in needed_folds:
        models_by_fold[fold] = {
            chave: load_fold_model_epoch(chave, fold, 99, run_dir, config, device).eval()
            for _, chave in MODELS
        }

    fig, axes = plt.subplots(len(amostras), 5, figsize=(15, 3 * len(amostras)), squeeze=False)
    for i, (slot, sid) in enumerate(amostras):
        subject = sid.split("/")[0]
        fold = fold_lookup[subject]
        thermal_c, gt = load_sample(sid, metadata, config)

        axes[i, 0].imshow(thermal_c, cmap="inferno")
        axes[i, 1].imshow(thermal_c, cmap="gray")
        axes[i, 1].imshow(gt, cmap=CMAP, norm=NORM, alpha=.6)

        x = to_model_input(thermal_c, config, device)
        for j, (nome, chave) in enumerate(MODELS, 2):
            with torch.no_grad():
                pred = models_by_fold[fold][chave](x).argmax(1)[0].cpu().numpy()
            axes[i, j].imshow(thermal_c, cmap="gray")
            axes[i, j].imshow(pred, cmap=CMAP, norm=NORM, alpha=.6)

        axes[i, 0].set_ylabel(f"{slot}\n{sid} (fold {fold})", fontsize=8)
        for a in axes[i]:
            a.set_xticks([])
            a.set_yticks([])

    for j, t in enumerate(["Térmica (°C)", "Referência", "U-Net", "TransUNet", "Swin-UNet++"]):
        axes[0, j].set_title(t, fontsize=11)

    fig.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(CORES[1:], CLASSES[1:])],
              loc="lower center", ncol=5, frameon=False)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = IMG_DIR / "fig_qualitativa.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSalvo: {out_path}")

    table.to_csv(OUT_DIR / "qualitative_candidates.csv", index=False)
    print(f"Salvo: {OUT_DIR / 'qualitative_candidates.csv'} (todas as candidatas, para conferência manual)")


def lateral_anomaly_figure() -> None:
    """Figura extra: sobrepõe as regiões esquerda/direita de UMA imagem de
    perfil (ex. sobrancelha/olho esq. vs dir.) no mesmo referencial, para
    tornar visível a assimetria discutida em §6.5.1 — usa diretamente as
    classes já rasterizadas na máscara (Sombrancelha/Olho esquerdo=2/5,
    direito=3/6 em REGION_NAMES), sem precisar reabrir o JSON de polígonos.
    """
    config = load_config()
    metadata = pd.read_csv(config.PROCESSED_DIR / "metadata.csv")
    landmarks_path = OUT_DIR / "landmark_classification.csv"
    if not landmarks_path.exists():
        print("landmark_classification.csv ausente — rode a01_landmark_scheme_census.py primeiro; pulando figura de anomalia lateral.")
        return
    profile_ids = pd.read_csv(landmarks_path)
    profile_ids = profile_ids[profile_ids["landmark_bucket"] == 43]["sample_id"]
    profile_ids = [s for s in profile_ids if s in set(metadata["sample_id"])]
    if not profile_ids:
        print("Nenhuma amostra de perfil encontrada; pulando figura de anomalia lateral.")
        return

    sid = MANUAL_OVERRIDE.get("perfil_43_marcos") or profile_ids[len(profile_ids) // 2]
    thermal_c, gt = load_sample(sid, metadata, config)

    # índices em REGION_NAMES (config.yaml): 2=Sombrancelha esquerda,
    # 3=Sombrancelha direita, 5=Olho esquerdo, 6=Olho direito.
    left = np.isin(gt, [2, 5])
    right = np.isin(gt, [3, 6])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(thermal_c, cmap="gray")
    overlay = np.zeros((*gt.shape, 4))
    overlay[left] = (0.2, 0.6, 1.0, 0.6)   # azul = esquerda
    overlay[right] = (1.0, 0.3, 0.2, 0.6)  # vermelho = direita
    ax.imshow(overlay)
    ax.set_title(f"Assimetria lateral — {sid} (43 marcos, perfil)")
    ax.legend(handles=[Patch(facecolor=(0.2, 0.6, 1.0), label="Esquerda"),
                       Patch(facecolor=(1.0, 0.3, 0.2), label="Direita")], loc="lower right")
    ax.set_xticks([])
    ax.set_yticks([])
    out_path = IMG_DIR / "fig_anomalia_lateral.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {out_path}")


if __name__ == "__main__":
    main()
    lateral_anomaly_figure()
