"""A10 — Re-medição de VRAM com a sonda isolada (UB-28).

Motivo. Os valores de VRAM da execução `2026-07-25_10-55-17`
(U-Net 586,7 / TransUNet 917,1 / Swin-UNet++ 937,1 MB) são **artefato da ordem
de sondagem**, não propriedades das arquiteturas. `probe_peak_memory` lia
`torch.cuda.max_memory_allocated()`, um contador do dispositivo inteiro, e
`reset_peak_memory_stats()` re-semeia o pico com o total **atualmente alocado**,
não com zero. Como `run_benchmark` mantém as três instâncias vivas em
`models_dict` e `load_model` move cada uma para a GPU sem nunca devolvê-la, cada
sondagem era inflada pelos pesos de todos os modelos sondados antes dela — a
inflação cresce monotonicamente com a ordem (+29 / +317 / +625 MB) e **inverte a
ordenação de VRAM entre as três arquiteturas**.

Este script produz a evidência da correção e os números para a Tab. 6.7:

* **isolado × N** — mesma arquitetura medida repetidamente sem nenhum outro
  modelo residente: comprova reprodutibilidade;
* **cumulativo** — arquiteturas sondadas na mesma ordem do benchmark real, com
  todas as anteriores ainda residentes na GPU: comprova invariância à residência
  (era exatamente a condição que produzia o artefato);
* **lastro** — 300 MB de tensor não relacionado residente: controle negativo,
  memória alheia que não é modelo também não pode contaminar a leitura.

Critério de aceitação (declarado antes de rodar): reprodutibilidade exata dentro
de um mesmo estado de residência, e desvio isolado↔cumulativo abaixo de 1 MB
(~0,2 %). Não se exige igualdade em bits entre estados de residência distintos
porque o alocador de cache do PyTorch arredonda blocos de forma dependente da
memória livre — um efeito real de alocador, três ordens de grandeza menor que o
defeito corrigido.
"""

from __future__ import annotations

import json

import torch

from _common import MODELS, OUT_DIR, build_model, get_device, load_config
from codes.benchmark_models import MEMORY_PROBE_BATCH_SIZE, probe_peak_memory

REPEATS = 3
BALLAST_MB = 300
# Tolerância isolado↔cumulativo (MB). Ver docstring: arredondamento de bloco do
# alocador, não contaminação. O defeito UB-28 produzia até 625 MB.
TOLERANCE_MB = 1.0

# Valores gravados pela execução relatada, para contraste (logs/<run>/*_benchmark.json).
REPORTED = {"U-Net": 586.75, "TransUNet": 917.15, "Swin-UNet++": 937.11}


def main() -> None:
    device = get_device()
    if device.type != "cuda":
        raise SystemExit(
            "A10 exige GPU: a sonda de VRAM retorna None em CPU por construção "
            "(R10 — reportar 0 fabricaria um número)."
        )
    config = load_config()

    # (a) isolado: um modelo por vez, nada mais residente.
    isolated: dict[str, list[float]] = {}
    for nome, chave in MODELS:
        model = build_model(chave, config).to(device)
        isolated[nome] = [
            probe_peak_memory(model, device, config.IMAGE_SIZE) for _ in range(REPEATS)
        ]
        del model
        torch.cuda.empty_cache()

    # (b) cumulativo: reproduz a situação real do benchmark (models_dict vivo).
    resident: dict[str, torch.nn.Module] = {}
    cumulative: dict[str, float] = {}
    for nome, chave in MODELS:
        resident[nome] = build_model(chave, config).to(device)
        cumulative[nome] = probe_peak_memory(resident[nome], device, config.IMAGE_SIZE)
    del resident
    torch.cuda.empty_cache()

    # (c) controle negativo: memória alheia que não é modelo.
    ballast = torch.zeros(BALLAST_MB * 1024 * 1024 // 4, device=device)
    with_ballast: dict[str, float] = {}
    for nome, chave in MODELS:
        model = build_model(chave, config).to(device)
        with_ballast[nome] = probe_peak_memory(model, device, config.IMAGE_SIZE)
        del model
        torch.cuda.empty_cache()
    del ballast
    torch.cuda.empty_cache()

    header = (f"{'modelo':<14}{'isolado (x%d)' % REPEATS:>24}{'cumulativo':>12}"
              f"{'+lastro':>10}{'relatado':>10}")
    print(header)
    print("-" * len(header))
    ok = True
    results = {}
    for nome, chave in MODELS:
        reps = isolated[nome]
        spread = max(reps) - min(reps)
        drift = abs(cumulative[nome] - reps[0])
        drift_ballast = abs(with_ballast[nome] - reps[0])
        ok &= spread < 0.01 and drift < TOLERANCE_MB and drift_ballast < TOLERANCE_MB
        print(f"{nome:<14}{'/'.join(f'{v:.2f}' for v in reps):>24}"
              f"{cumulative[nome]:>12.2f}{with_ballast[nome]:>10.2f}{REPORTED[nome]:>10.1f}")
        results[nome] = {
            "model_key": chave,
            "vram_mb": reps[0],
            "isolated_repeats_mb": reps,
            "repeat_spread_mb": spread,
            "cumulative_mb": cumulative[nome],
            "cumulative_drift_mb": drift,
            "with_ballast_mb": with_ballast[nome],
            "reported_in_run_mb": REPORTED[nome],
            "inflation_in_run_mb": REPORTED[nome] - reps[0],
            "probe_batch": MEMORY_PROBE_BATCH_SIZE,
            "device": torch.cuda.get_device_name(0),
        }

    order_iso = " < ".join(n for n, _ in sorted(MODELS, key=lambda t: isolated[t[0]][0]))
    order_run = " < ".join(n for n, _ in sorted(MODELS, key=lambda t: REPORTED[t[0]]))
    print(f"\nOrdenação isolada  : {order_iso}")
    print(f"Ordenação relatada : {order_run}")
    print(f"\nAC UB-28 (repetição exata; desvio < {TOLERANCE_MB:.1f} MB): "
          f"{'OK' if ok else 'FALHOU'}")

    out_path = OUT_DIR / "vram_isolated.json"
    json.dump({"results": results,
               "ordering_isolated": order_iso,
               "ordering_reported": order_run,
               "tolerance_mb": TOLERANCE_MB,
               "acceptance_passed": bool(ok)},
              open(out_path, "w"), indent=2)
    print(f"Salvo: {out_path}")


if __name__ == "__main__":
    main()
