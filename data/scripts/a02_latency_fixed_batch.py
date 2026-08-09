"""A2 — Reinstrumenta latência de inferência com lote fixo (batch=4).

Bugs do rascunho original (ver outputs.txt):

1. ``from model_registry import create_model`` nunca importa os módulos de
   modelo (``codes.unet_v2``, ``codes.transunet``,
   ``codes.swin_unet_plus_plus``) — é a importação desses módulos que
   executa o decorator ``@register_model(...)`` e povoa o registry. Sem
   isso, ``create_model('unet', ...)`` falha com "Available models: []"
   mesmo o registry existindo — não é um bug de dado, é ordem de import.
2. ``sys.path.insert(0, '../codes/')`` + ``from model_registry import
   create_model`` quebra com os imports absolutos ``codes.`` do módulo
   pós-T3.6 (UB-21): ao colocar ``codes/`` (não a raiz do repo) no path,
   ``codes/unet_v2.py``'s próprio ``from codes.model_registry import
   register_model`` não resolve. A raiz do repo precisa estar no
   ``sys.path``, não ``codes/``.
3. A chave ``'swin'`` não existe no ``model_registry`` — é só o alias de
   CLI do ``main_pipeline.py``; a chave real é
   ``'swin_unet_plus_plus'`` (ver ``codes/model_registry.py`` +
   ``codes/main_pipeline.py:_MODEL_ALIASES``).
4. ``create_model(chave, num_classes=10)`` sozinho falha para TransUNet e
   Swin-UNet++: seus construtores exigem ``img_size`` (ver
   ``codes/transunet.py``/``codes/swin_unet_plus_plus.py``); só o U-Net não
   usa esse kwarg.

Correção: usar ``_common.build_model`` (mesmos kwargs que
``main_pipeline.py:train_model`` monta) sobre a chave canônica do registry.
"""

from __future__ import annotations

import json
import time

import torch

from _common import MODELS, OUT_DIR, build_model, get_device, load_config

BATCH, WARMUP, ITERS = 4, 20, 200  # lote igual ao da sonda de VRAM (A7/UB-10)


def main() -> None:
    device = get_device()
    if device.type != "cuda":
        print(
            "⚠️  Nenhuma GPU CUDA disponível — a latência medida em CPU não é "
            "comparável às números de VRAM/latência do benchmark real "
            "(M3: nunca compare hardware distinto). Prosseguindo mesmo assim "
            "para não bloquear o desenvolvimento do script; rode de novo na GPU."
        )

    config = load_config()
    x = torch.zeros((BATCH, 1, *config.IMAGE_SIZE), device=device)
    res = {}

    for nome, chave in MODELS:
        m = build_model(chave, config).to(device).eval()
        with torch.no_grad():
            for _ in range(WARMUP):
                m(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(ITERS):
                m(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        ms_img = dt * 1000 / (ITERS * BATCH)
        res[nome] = {
            "model_key": chave,
            "ms_por_imagem": ms_img,
            "imagens_por_s": 1000 / ms_img,
            "lote": BATCH,
            "warmup": WARMUP,
            "iters": ITERS,
            "device": str(device),
        }
        print(f"{nome:14} {ms_img:6.3f} ms/img   {1000 / ms_img:7.1f} img/s")
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_path = OUT_DIR / "latency_fixed_batch.json"
    json.dump(res, open(out_path, "w"), indent=2)
    print(f"\nSalvo: {out_path}")


if __name__ == "__main__":
    main()
