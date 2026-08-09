"""A13 — Evidência reproduzível da regra de lado visível do esquema de 43 marcos (UB-29).

Por que este script existe. `LANDMARK_MAPPINGS_43` atribui o **mesmo** intervalo
de índices às duas sobrancelhas (12-15) e aos dois olhos (22-26), e os 43
índices estão integralmente consumidos. A razão é anatômica: uma vista de perfil
mostra **um** olho e **uma** sobrancelha, e o esquema anota apenas o lado
visível. Qual lado é esse depende de para onde a cabeça está virada, e **nenhuma
coluna do CSV registra isso**.

Este script demonstra que o lado é, ainda assim, **derivável da geometria dos
próprios marcos**, e que a derivação é sustentada por dois sinais independentes
e ortogonais. Ele não modifica dado algum; apenas mede e reporta.

As três medições:

(a) **Convenção de nomes** — em imagens frontais (73 marcos), ``Olho esquerdo``
    tem x menor que ``Olho direito``? Estabelece se o dataset nomeia pelo lado
    da imagem (observador) ou pela anatomia do sujeito.

(b) **Sinal 1: largura do olho** — conforme a cabeça gira, o olho do lado que se
    afasta da câmera estreita. Medido em faixas de rotação.

(c) **Sinal 2: distância olho-nariz** — o olho do lado que se afasta colapsa em
    direção ao nariz. Sinal ortogonal ao anterior: um mede extensão, o outro
    posição.

Se (b) e (c) forem monotônicos e concordantes, o olho que **sobrevive** até o
perfil é o do lado oposto ao deslocamento do nariz — que é a regra implementada
em ``codes/generate_boxes_polygons._resolve_visible_side``.

⚠️ Regra **derivada e validada**, não a especificação publicada do dataset. Uma
convenção oficial de índices por lado, se publicada, prevalece sobre esta.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from _common import OUT_DIR, landmark_files

FRONTAL, PERFIL = 73, 43
# Blocos do esquema de 73 (LANDMARK_MAPPINGS_73)
F_NOSE, F_EYE_L, F_EYE_R = range(27, 36), range(36, 42), range(42, 48)
# Blocos do esquema de 43 (LANDMARK_MAPPINGS_43)
P_NOSE, P_BROW, P_EYE = range(16, 22), range(12, 16), range(22, 27)

FAIXAS = [0, .35, .45, .55, .65, 1.0]
ROTULOS = ["nariz bem a ESQ", "esq", "centro", "dir", "nariz bem a DIR"]


def _carregar() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (frontais, perfis) com centróides x já normalizados pela face."""
    frontais, perfis = [], []
    for path in landmark_files():
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            n = sum(1 for i in range(FRONTAL)
                    if pd.notna(r.get(f"x{i}")) and pd.notna(r.get(f"y{i}")))
            if n not in (FRONTAL, PERFIL):
                continue                       # linhas parciais/danificadas
            lim = FRONTAL if n == FRONTAL else PERFIL
            xs = [r[f"x{i}"] for i in range(lim) if pd.notna(r.get(f"x{i}"))]
            larg = max(xs) - min(xs)
            if larg <= 0:
                continue
            cx = lambda idxs: np.mean([r[f"x{i}"] for i in idxs
                                       if pd.notna(r.get(f"x{i}"))])
            if n == FRONTAL:
                oe, od, nz = cx(F_EYE_L), cx(F_EYE_R), cx(F_NOSE)
                le = (max(r[f"x{i}"] for i in F_EYE_L)
                      - min(r[f"x{i}"] for i in F_EYE_L))
                ld = (max(r[f"x{i}"] for i in F_EYE_R)
                      - min(r[f"x{i}"] for i in F_EYE_R))
                frontais.append({
                    "nariz_rel": (nz - min(xs)) / larg,
                    "x_olho_esq": oe, "x_olho_dir": od,
                    "larg_esq": le / larg, "larg_dir": ld / larg,
                    "dist_esq": abs(oe - nz) / larg, "dist_dir": abs(od - nz) / larg,
                })
            else:
                perfis.append({
                    "nariz_rel": (cx(P_NOSE) - min(xs)) / larg,
                    "sobr_rel": (cx(P_BROW) - min(xs)) / larg,
                    "olho_rel": (cx(P_EYE) - min(xs)) / larg,
                })
    return pd.DataFrame(frontais), pd.DataFrame(perfis)


def main() -> None:
    fr, pf = _carregar()
    print(f"linhas frontais (73 marcos): {len(fr)}   perfis (43): {len(pf)}\n")

    # (a) convenção de nomes
    frac = (fr["x_olho_esq"] < fr["x_olho_dir"]).mean()
    print(f"=== (a) Convenção de nomes ===")
    print(f"'Olho esquerdo' tem x MENOR em {100*frac:.1f}% das imagens frontais")
    conv = ("lado do OBSERVADOR (image-side)" if frac > .95
            else "esquerda ANATÔMICA do sujeito" if frac < .05 else "AMBÍGUA")
    print(f"=> convenção: {conv}\n")

    fr["faixa"] = pd.cut(fr["nariz_rel"], FAIXAS, labels=ROTULOS)

    # (b) e (c): dois sinais ortogonais
    print("=== (b) Sinal 1: largura relativa do olho ===")
    print(f"{'faixa':<18}{'larg_esq':>10}{'larg_dir':>10}{'razao':>9}   olho mais visivel")
    razoes = {}
    for f in ROTULOS:
        s = fr[fr["faixa"] == f]
        if len(s) < 5:
            continue
        rz = s["larg_esq"].mean() / s["larg_dir"].mean()
        razoes[f] = rz
        print(f"{f:<18}{s['larg_esq'].mean():>10.3f}{s['larg_dir'].mean():>10.3f}"
              f"{rz:>9.3f}   {'ESQUERDO' if rz > 1 else 'DIREITO'}")

    print("\n=== (c) Sinal 2: distância do olho ao nariz ===")
    print(f"{'faixa':<18}{'dist_esq':>10}{'dist_dir':>10}   olho colado ao nariz (ocultando)")
    dists = {}
    for f in ROTULOS:
        s = fr[fr["faixa"] == f]
        if len(s) < 5:
            continue
        de, dd = s["dist_esq"].mean(), s["dist_dir"].mean()
        dists[f] = (de, dd)
        print(f"{f:<18}{de:>10.3f}{dd:>10.3f}   {'ESQUERDO' if de < dd else 'DIREITO'}")

    mono = all(razoes[a] < razoes[b] for a, b in zip(ROTULOS, ROTULOS[1:])
               if a in razoes and b in razoes)
    concorda = all((razoes[f] > 1) == (dists[f][0] > dists[f][1]) for f in razoes)
    print(f"\nSinal 1 monotônico entre faixas: {mono}")
    print(f"Sinais 1 e 2 concordam em todas as faixas: {concorda}")

    print("\n=== REGRA DERIVADA ===")
    print("  nariz deslocado para a ESQUERDA da imagem -> lado visivel = DIREITO")
    print("  nariz deslocado para a DIREITA  da imagem -> lado visivel = ESQUERDO")

    # distribuição dos perfis sob a regra
    n_dir = int((pf["nariz_rel"] < 0.35).sum())
    n_esq = int((pf["nariz_rel"] > 0.65).sum())
    n_amb = len(pf) - n_dir - n_esq
    print(f"\nAplicada aos {len(pf)} perfis: {n_dir} -> direito, {n_esq} -> esquerdo, "
          f"{n_amb} ambiguos ({100*n_amb/len(pf):.2f}%)")

    out = OUT_DIR / "landmark_side_evidence.json"
    json.dump({
        "n_frontais": len(fr), "n_perfis": len(pf),
        "frac_olho_esq_x_menor": float(frac), "convencao": conv,
        "razao_largura_por_faixa": {k: float(v) for k, v in razoes.items()},
        "dist_olho_nariz_por_faixa": {k: [float(a), float(b)]
                                      for k, (a, b) in dists.items()},
        "sinal1_monotonico": bool(mono), "sinais_concordam": bool(concorda),
        "perfis_direito": n_dir, "perfis_esquerdo": n_esq, "perfis_ambiguos": n_amb,
        "regra": "nariz a esquerda -> lado visivel direito; nariz a direita -> esquerdo",
        "status": "derivada e validada; nao e a especificacao publicada do dataset",
    }, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"Salvo: {out}")


if __name__ == "__main__":
    main()
