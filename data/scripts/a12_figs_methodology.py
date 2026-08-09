"""A12 — Figuras do Capítulo 5 (metodologia) a partir dos dados já processados.

Gera as três figuras que o capítulo declarava pendentes por falta de dados
pré-processados nesta máquina. Nada é regenerado: lê os `.npy` (Celsius) e as
máscaras `.png` que a execução relatada de fato usou.

* **amostras_mascaras** — grade de amostras com sobreposição de máscara em três
  distâncias de captura (próxima, média, distante), imagem em °C.
* **pixels_por_classe** — distribuição de pixels por classe, escala logarítmica.
  A assimetria entre pares laterais é consequência do defeito UB-27 e está
  anotada na própria figura, para que a barra não seja lida como propriedade
  anatômica.
* **aumento_dados** — antes/depois do aumento físico, com barra de cores em °C
  evidenciando que a deriva desloca a escala em quantidade fisicamente
  plausível. Usa a mesma autoridade de augmentação do treino
  (`codes.augmentation.build_thermal_transform`).
"""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from _common import IMG_DIR, OUT_DIR, load_config, load_raw_subject_csv

CORES = ['#00000000', '#8dd3c7', '#ffffb3', '#bebada', '#fb8072',
         '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9']
CMAP = ListedColormap(CORES)
NORM = BoundaryNorm(range(11), 10)
CURTOS = ['Fundo', 'Contorno', 'Sobr. esq.', 'Sobr. dir.', 'Nariz',
          'Olho esq.', 'Olho dir.', 'Boca', 'Lábios', 'Testa']
# Nomes de exibição, na ordem de config.REGION_NAMES. As chaves reais do
# config.yaml trazem "Sombrancelha" (grafia incorreta de "Sobrancelha") e
# "Labios" sem acento; a dissertação declara essa divergência em nota e usa a
# grafia correta na tabela de classes. As figuras seguem a mesma convenção.
EXIBICAO = ['Plano de fundo', 'Contorno inferior do rosto', 'Sobrancelha esquerda',
            'Sobrancelha direita', 'Nariz', 'Olho esquerdo', 'Olho direito',
            'Boca', 'Lábios', 'Testa']


def _load(row, config):
    img = np.load(config.DATA_DIR.parent / row["image_path"]).astype(np.float32)
    msk = cv2.imread(str(config.DATA_DIR.parent / row["mask_path"]), cv2.IMREAD_UNCHANGED)
    return img, msk


def fig_amostras(config, meta: pd.DataFrame) -> None:
    """Grade 3x2: três distâncias de captura, imagem térmica e sobreposição."""
    subjects = sorted({s.split("/")[0] for s in meta["sample_id"]})
    raw = pd.concat([load_raw_subject_csv(s) for s in subjects], ignore_index=True)
    m = meta.merge(raw[["sample_id", "Distance"]], on="sample_id", how="inner").dropna(subset=["Distance"])

    alvos = [("Próxima", m["Distance"].min()), ("Média", m["Distance"].median()),
             ("Distante", m["Distance"].max())]
    fig, axes = plt.subplots(len(alvos), 2, figsize=(7.2, 3.3 * len(alvos)), squeeze=False)
    for i, (rotulo, alvo) in enumerate(alvos):
        row = m.iloc[(m["Distance"] - alvo).abs().argsort().iloc[0]]
        img, msk = _load(row, config)
        im = axes[i, 0].imshow(img, cmap="inferno")
        fig.colorbar(im, ax=axes[i, 0], fraction=0.046).set_label("°C")
        axes[i, 1].imshow(img, cmap="gray")
        axes[i, 1].imshow(msk, cmap=CMAP, norm=NORM, alpha=0.6)
        axes[i, 0].set_ylabel(f"{rotulo}\n{row['Distance']:.1f} m", fontsize=9)
        for a in axes[i]:
            a.set_xticks([]); a.set_yticks([])
    axes[0, 0].set_title("Imagem térmica", fontsize=10)
    axes[0, 1].set_title("Sobreposição da máscara", fontsize=10)
    fig.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(CORES[1:], CURTOS[1:])],
               loc="lower center", ncol=5, frameon=False, fontsize=8)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    out = IMG_DIR / "fig_amostras_mascaras.pdf"
    plt.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"Salvo: {out}")


def fig_pixels(config) -> None:
    """Barras horizontais em escala log da contagem de pixels por classe."""
    counts = pd.read_csv(OUT_DIR / "pixels_per_class.csv", index_col="classe")["pixel_count"]
    counts = counts.reindex(config.REGION_NAMES)
    if len(EXIBICAO) != len(config.REGION_NAMES):
        raise ValueError(
            f"EXIBICAO tem {len(EXIBICAO)} nomes para {len(config.REGION_NAMES)} "
            "classes de config.REGION_NAMES — atualize o mapa de exibição."
        )
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    cores = [CORES[i] if i else "#bbbbbb" for i in range(len(counts))]
    ax.barh(range(len(counts)), counts.values, color=cores, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(EXIBICAO, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Pixels no conjunto pré-processado (escala logarítmica)")
    ax.invert_yaxis()
    for i, v in enumerate(counts.values):
        ax.text(v * 1.15, i, f"{v:,.0f}".replace(",", "."), va="center", fontsize=7.5)
    ax.set_xlim(right=counts.max() * 6)
    for par in (("Sombrancelha esquerda", "Sombrancelha direita"),
                ("Olho esquerdo", "Olho direito")):
        i_e, i_d = counts.index.get_loc(par[0]), counts.index.get_loc(par[1])
        razao = counts.iloc[i_e] / counts.iloc[i_d]
        ax.annotate("", xy=(counts.max() * 3, i_d), xytext=(counts.max() * 3, i_e),
                    arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.1))
        ax.text(counts.max() * 3.4, (i_e + i_d) / 2, f"{razao:.1f}$\\times$",
                color="crimson", fontsize=8, va="center")
    plt.tight_layout()
    out = IMG_DIR / "fig_pixels_por_classe.pdf"
    plt.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"Salvo: {out}  (razões laterais anotadas em vermelho)")


def fig_aumento(config, meta: pd.DataFrame) -> None:
    """Original + três versões aumentadas, todas com barra de cores em °C."""
    from codes.augmentation import build_thermal_transform

    row = meta.iloc[len(meta) // 2]
    img, msk = _load(row, config)
    transform = build_thermal_transform(config.PREPROCESSING.augmentation,
                                        seed=config.RANDOM_SEED)
    paineis = [("Original", img, msk)]
    for k in range(3):
        aug = transform(image=img, mask=msk)
        paineis.append((f"Aumentada {k + 1}", aug["image"], aug["mask"]))

    vmin = min(p[1].min() for p in paineis)
    vmax = max(p[1].max() for p in paineis)
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.0))
    for ax, (titulo, im_c, mk) in zip(axes.ravel(), paineis):
        h = ax.imshow(im_c, cmap="inferno", vmin=vmin, vmax=vmax)
        ax.imshow(mk, cmap=CMAP, norm=NORM, alpha=0.35)
        ax.set_title(f"{titulo}  [{im_c.min():.1f}--{im_c.max():.1f} °C]", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(h, ax=ax, fraction=0.046).set_label("°C", fontsize=8)
    plt.tight_layout()
    out = IMG_DIR / "fig_aumento_dados.pdf"
    plt.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"Salvo: {out}  (escala comum {vmin:.1f}--{vmax:.1f} °C)")


def main() -> None:
    config = load_config()
    meta = pd.read_csv(config.PROCESSED_DIR / "metadata.csv")
    fig_amostras(config, meta)
    fig_pixels(config)
    fig_aumento(config, meta)


if __name__ == "__main__":
    main()
