"""A1 — Conta imagens no caminho de 43 vs 73 marcos faciais (landmarks).

Bug do rascunho original (ver a saída real em outputs.txt):

    for csv in sorted(glob.glob('S*.csv', recursive=True)):
        df = pd.read_csv(csv)
        n = len([x for x in df.columns if x.lower().startswith(('x', 'y'))]) // 2
        ...

Isso produziu ``Counter({73: 8454, 0: 8449})`` — nunca um único "43". Dois
problemas, confirmados lendo os CSVs reais em ``data/``:

1. ``glob('S*.csv')`` também casa ``S{n}_bounding_boxes.csv`` (colunas
   ``ID,min_x,min_y,max_x,max_y`` — nenhuma começa com 'x'/'y' isoladamente,
   então cada um desses arquivos contribuía como "0 marcos", inflando
   artificialmente o balde 0 com ~8449 linhas que não são imagens de marcos.
2. Cada ``S{n}.csv`` real tem uma largura de esquema FIXA — 73 pares (x,y),
   146 colunas — em TODA linha, frontal ou de perfil. Contar colunas do
   arquivo é portanto uma constante (=73) e não discrimina nada por imagem.
   O sinal real está por LINHA: imagens de perfil só preenchem ~43 dos 73
   pares e deixam o resto como NaN (verificado: ``S1.csv`` tem linhas com
   146 e com 86 valores não-nulos entre as colunas x/y — 73 e 43 marcos).

Correção: excluir os arquivos de bounding boxes do glob e contar, por linha,
quantos pares (x,y) estão de fato preenchidos.
"""

from __future__ import annotations

from collections import Counter

from _common import OUT_DIR, classify_landmarks, landmark_files


def main() -> None:
    files = landmark_files()
    if not files:
        raise FileNotFoundError("Nenhum S{n}.csv de marcos encontrado em data/")

    print(f"Arquivos de marcos (landmarks) considerados: {[p.name for p in files]}")

    frames = [classify_landmarks(p) for p in files]
    import pandas as pd
    all_df = pd.concat(frames, ignore_index=True)

    counts = Counter(int(b) for b in all_df["landmark_bucket"])
    total = sum(counts.values())
    print(Counter(counts))
    for bucket in sorted(counts):
        print(f"{bucket} marcos: {counts[bucket]} imagens ({100 * counts[bucket] / total:.1f}%)")

    print("\nPor sujeito:")
    per_subject = (
        all_df.groupby(["subject", "landmark_bucket"]).size().unstack(fill_value=0)
    )
    print(per_subject)

    detail_path = OUT_DIR / "landmark_classification.csv"
    summary_path = OUT_DIR / "landmark_by_subject.csv"
    all_df.to_csv(detail_path, index=False)
    per_subject.to_csv(summary_path)
    print(f"\nSalvo: {detail_path}")
    print(f"Salvo: {summary_path}")


if __name__ == "__main__":
    main()
