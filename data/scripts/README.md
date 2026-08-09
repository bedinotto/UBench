# Scripts de análise da dissertação

Análises pontuais que produzem os números e as figuras citados pela
dissertação. **Não fazem parte do *pipeline***: nada aqui treina, re-processa
dados ou escreve em `outputs/`/`logs/` do projeto. Leem os artefatos da
execução relatada (`2026-07-25_10-55-17`) e gravam em `data/outputs/` e
`data/img/`.

## Como rodar

```bash
cd /home/doga/Documents/UBench/data/scripts
../../.venv/bin/python a03_per_subject_wilcoxon_best.py
```

Sempre com o `.venv` do projeto, nunca com o `base` do conda: o build de
`opencv-python` daquele ambiente está ligado a um `libtiff` de sistema sem
símbolos, que é a origem do crash registrado em `data/outputs.txt`.

`_common.py` é a autoridade compartilhada. Ele põe `codes.*` no `sys.path`,
resolve `DATA_DIR` como o diretório **pai** deste, e expõe os contratos que o
*pipeline* já possui (chaves do registro de modelos, nomes de *checkpoint*,
divisão por sujeito). Nenhum script pode re-derivar esses contratos por conta
própria, é exatamente a classe de defeito de UB-02.

## O que cada um faz

| Script | O que produz | Saídas em `data/` |
|---|---|---|
| `a01_landmark_scheme_census.py` | Conta imagens no esquema de 43 vs 73 marcos, por sujeito | `outputs/landmark_classification.csv`, `outputs/landmark_by_subject.csv` |
| `a02_latency_fixed_batch.py` | Latência de inferência sob lote fixo 4, com aquecimento descartado e sincronização de GPU | `outputs/latency_fixed_batch.json` |
| `a03_per_subject_wilcoxon_best.py` | Métricas por sujeito e Wilcoxon pareado sobre os pesos `best_*.pth`, a variante **secundária** | `outputs/per_subject_metrics.csv`, `outputs/per_image_metrics.csv`, `outputs/wilcoxon_per_subject.csv` |
| `a04_figs_qualitative_and_anomaly.py` | Grade visual de inferência e a figura da anomalia lateral | `img/fig_qualitativa.pdf`, `img/fig_anomalia_lateral.pdf`, `outputs/qualitative_candidates.csv` |
| `a05_distance_strata_and_pixels.py` | mIoU por faixa de distância, pixels por classe e o Spearman área↔IoU | `outputs/miou_by_distance.csv`, `outputs/pixels_per_class.csv` |
| `a06_figs_convergence.py` | Um painel de convergência por arquitetura, com as 5 curvas de partição | `img/conv_*.pdf` |
| `a07_fig_pareto_frontier.py` | Fronteira de compromisso entre velocidade, mIoU e VRAM | `img/fig_pareto.pdf` |
| `a08_boundary_metrics_hd95_nsd.py` | HD95 e NSD. **Não executado**: MONAI não instalado, e a limitação está declarada na dissertação em vez de fabricada | `outputs/boundary_metrics.csv` |
| `a09_per_subject_wilcoxon_final_epoch.py` | O mesmo que `a03`, mas sobre os *checkpoints* de época 99, a variante **primária** que a dissertação reporta | `outputs/per_subject_metrics_final_epoch.csv`, `outputs/per_image_metrics_final_epoch.csv`, `outputs/wilcoxon_per_subject_final_epoch.csv` |
| `a10_vram_isolated_probe.py` | Re-mede VRAM com a sonda isolada, a evidência de UB-28 | `outputs/vram_isolated.json` |
| `a11_mask_defect_census.py` | Censo do defeito de derivação de máscaras sobre as 8074 imagens, a evidência de UB-27 | `outputs/mask_defect_census.csv`, `outputs/mask_defect_summary.json` |
| `a12_figs_methodology.py` | As três figuras do capítulo de metodologia, a partir dos dados já processados | `img/fig_amostras_mascaras.pdf`, `img/fig_pixels_por_classe.pdf`, `img/fig_aumento_dados.pdf` |
| `a13_landmark_side_evidence.py` | Deriva e valida a regra de lado visível do esquema de 43 marcos, a evidência de UB-29 | `outputs/landmark_side_evidence.json` |

## Dependências entre eles

O prefixo numérico é a ordem de execução, não uma cadeia rígida. As
dependências reais são poucas:

- `a04` precisa de `a01` (classificação de marcos) e de `a09` (métricas por imagem);
- `a05` precisa de `a09` (métricas por imagem);
- `a07` precisa de `a02` (latência) e de `a10` (VRAM);
- `a09` importa `evaluate_subject` de `a03`;
- `a12` precisa de `a05` (contagem de pixels por classe).

Um script que não encontra a entrada de que precisa levanta `FileNotFoundError`
nomeando qual rodar antes, em vez de seguir com valor padrão.

## Regras

Os dados da execução relatada estão **congelados**. Nenhum script aqui
regenera máscaras, re-processa `data/processed/` ou retreina, porque isso
quebraria a correspondência entre os 15 *checkpoints* e os dados que os
produziram. Os defeitos UB-27, UB-28 e UB-29 são **divulgados** na
dissertação, não corrigidos retroativamente nos artefatos.

Número não medido não vira número escrito. Onde a medição não foi feita, o
script falha ou declara a ausência, como `a08` faz.
