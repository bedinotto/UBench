"""A7 — Fronteira de compromisso (velocidade x mIoU x VRAM).

Não havia rascunho para este script (pendente). Depende de artefatos de
outros scripts já corrigidos/criados nesta sessão:

* ``outputs/latency_fixed_batch.json`` — gerado por `a02_latency_fixed_batch.py` (lote fixo=4,
  com warm-up e `torch.cuda.synchronize()`).
* ``logs/<run>/<model_key>_fold_*_metrics.json`` — já existem (gravados
  pelo treino real), usados aqui só para o mIoU final ± dp entre folds.

A VRAM é medida ao vivo com a MESMA sonda do benchmark real
(``codes.benchmark_models.probe_peak_memory`` — lote fixo=4, UB-10), em vez
dos valores fixos citados no documento-fonte como referência manual
("917,1 / 586,7 / 937,1 MB"): hardcodar aqueles seria reportar um número que
este script não mediu (R10).
"""

from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

from _common import IMG_DIR, MODELS, OUT_DIR, build_model, get_device, load_config, resolve_run_dir


def load_latency() -> dict:
    p = OUT_DIR / "latency_fixed_batch.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} não existe — rode a02_latency_fixed_batch.py primeiro.")
    return json.load(open(p))


def load_final_miou(config, run_dir) -> dict:
    log_dir = config.LOG_DIR / run_dir.name
    out = {}
    for nome, chave in MODELS:
        fold_files = sorted(log_dir.glob(f"{chave}_fold_*_metrics.json"))
        if not fold_files:
            raise FileNotFoundError(f"Nenhum metrics.json para '{chave}' em {log_dir}")
        vals = [json.load(open(p))["final_val_iou"] for p in fold_files]
        # ddof=1 (desvio amostral) para bater com a Tab. 6.2 da dissertação,
        # que reporta o desvio ENTRE as cinco partições. np.std usa ddof=0 por
        # padrão, o que daria barras de erro menores que as da tabela.
        out[nome] = (float(np.mean(vals)), float(np.std(vals, ddof=1)))
    return out


def probe_vram(config, device) -> dict:
    """Peak VRAM per architecture at the fixed probe batch (UB-10, M3).

    Uses freshly-constructed (untrained) weights — VRAM depends on tensor
    shapes flowing through the architecture, not on specific trained weight
    values, so no checkpoint needs to be loaded here.
    """
    import torch
    from codes.benchmark_models import probe_peak_memory

    out = {}
    for nome, chave in MODELS:
        model = build_model(chave, config).to(device)
        out[nome] = probe_peak_memory(model, device, config.IMAGE_SIZE)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def main() -> None:
    config = load_config()
    device = get_device()
    run_dir = resolve_run_dir(config)

    latency = load_latency()
    miou = load_final_miou(config, run_dir)
    vram = probe_vram(config, device)

    plt.figure(figsize=(7, 5))
    for nome, _chave in MODELS:
        x = latency[nome]["imagens_por_s"]
        y_mean, y_std = miou[nome]
        v = vram[nome]
        plt.scatter(x, y_mean, s=(v or 50) / 3, alpha=.6)
        plt.errorbar(x, y_mean, yerr=y_std, fmt="none", capsize=4)
        label = f"{nome}\n{v:.1f} MB" if v is not None else f"{nome}\nn/a (CPU)"
        plt.annotate(label, (x, y_mean), xytext=(8, 8), textcoords="offset points")

    plt.axvline(30, ls="--", c="gray")  # limiar de referência p/ monitoramento contínuo — justifique no texto
    plt.xlabel("Imagens por segundo (lote fixo=4, A2)")
    plt.ylabel("mIoU (época final, média ± dp entre folds)")
    plt.title("Fronteira de compromisso: velocidade x acurácia x VRAM")
    out_path = IMG_DIR / "fig_pareto.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {out_path}")

    for nome, _chave in MODELS:
        v = vram[nome]
        print(f"{nome:14} img/s={latency[nome]['imagens_por_s']:.1f}  "
              f"mIoU_final={miou[nome][0]:.4f}±{miou[nome][1]:.4f}  "
              f"VRAM={'%.1f MB' % v if v is not None else 'n/a (CPU)'}")


if __name__ == "__main__":
    main()
