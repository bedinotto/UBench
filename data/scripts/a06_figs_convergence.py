"""A6 — Painéis de convergência (uma figura por arquitetura, 5 curvas de fold).

Não havia rascunho para este script (pendente) — implementado a partir da
especificação do documento-fonte. Lê ``val_ious`` de cada
``logs/<run>/<model_key>_fold_<n>_metrics.json`` (já existem, um por
modelo/fold, gravados pelo `UnifiedTrainer` — ver `trainer.save_metrics()`
em `codes/unified_training.py`) em vez de recompor nada por inferência: é
puro replot de artefatos já existentes, sem tocar GPU.
"""

from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

from _common import IMG_DIR, MODELS, load_config, resolve_run_dir


def main() -> None:
    config = load_config()
    run_dir = resolve_run_dir(config)
    log_dir = config.LOG_DIR / run_dir.name
    if not log_dir.exists():
        raise FileNotFoundError(f"Diretório de logs não encontrado: {log_dir}")

    for nome, chave in MODELS:
        fold_files = sorted(log_dir.glob(f"{chave}_fold_*_metrics.json"))
        if not fold_files:
            print(f"⚠️  Nenhum metrics.json para '{chave}' em {log_dir}; pulando {nome}.")
            continue

        curvas = []
        for p in fold_files:
            d = json.load(open(p))
            curvas.append(d["val_ious"])
        # Folds podem, em tese, ter contagens de época diferentes (ex. resume
        # interrompido); corta no comprimento comum em vez de fabricar
        # valores além do que foi de fato registrado.
        min_len = min(len(c) for c in curvas)
        if any(len(c) != min_len for c in curvas):
            print(f"⚠️  {nome}: folds com número de épocas diferente "
                  f"({[len(c) for c in curvas]}); truncando em {min_len}.")
        curvas = np.array([c[:min_len] for c in curvas])

        m, s = curvas.mean(0), curvas.std(0)
        plt.figure(figsize=(6, 4))
        for f in range(curvas.shape[0]):
            plt.plot(curvas[f], alpha=.3, lw=.8)
        plt.plot(m, lw=2, label="média")
        plt.fill_between(range(len(m)), m - s, m + s, alpha=.2)
        plt.xlabel("Época")
        plt.ylabel("mIoU de validação")
        plt.title(nome)
        plt.legend()
        out_path = IMG_DIR / f"conv_{chave}.pdf"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"Salvo: {out_path}  (min_val_iou={curvas.min():.4f}, "
              f"max_val_iou={curvas.max():.4f}, época final média={m[-1]:.4f}±{s[-1]:.4f})")


if __name__ == "__main__":
    main()
