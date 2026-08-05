# Ações pendentes — sessão de 2026-08-03 (UB-27/UB-28/UB-29 + revisão da dissertação)

**Por que este documento existe.** Nesta sessão, o *pipeline* e a dissertação
foram auditados a partir do padrão dos próprios resultados (não da leitura
isolada do código), o que revelou dois defeitos novos (UB-27/UB-29 na
derivação de máscaras, UB-28 na sonda de VRAM), corrigidos e propagados por
todo o texto. O trabalho de código e de redação está **fechado e verificado**
(suíte completa 156 passed / 2 skipped / 0 failed; build da dissertação com 0
erros, 195 páginas). O que resta são itens que só você pode decidir ou
executar — busca externa, julgamento visual, ou uma decisão de escopo. Cada
item abaixo tem local exato, o que fazer, e como verificar que funcionou.

Ordem sugerida: **2 → 3 → 5**; o item 1 virou confirmação opcional, não
bloqueio. Os demais são independentes entre si.

---

## 1. UB-29 — confirmar (ou refutar) a regra de lado visível derivada

**Status: não bloqueia mais.** Quando este documento foi escrito, faltava a
especificação de índices por lado do esquema de 43 marcos, e a reexecução
corrigida estava travada. Você forneceu o fato semântico que faltava — em uma
vista de perfil só um olho e uma sobrancelha ficam visíveis, e qual lado
depende de para onde o rosto está virado —, e com ele foi possível **derivar a
regra da própria geometria dos marcos** e implementá-la.

### O que foi feito

A regra implementada em `codes/generate_boxes_polygons._resolve_visible_side`:

```
nariz deslocado para a ESQUERDA da imagem -> lado visível = DIREITO
nariz deslocado para a DIREITA  da imagem -> lado visível = ESQUERDO
```

Ela repousa sobre três medições, todas reproduzíveis com
`python data/a13_landmark_side_evidence.py`:

1. **Convenção de nomes do dataset** — `Olho esquerdo` tem x menor que
   `Olho direito` em **99,9%** das 4225 linhas frontais, o que estabelece que
   o dataset nomeia pelo lado da **imagem** (observador), não pela anatomia do
   sujeito.
2. **Sinal 1 — largura relativa do olho**: conforme a cabeça gira, o olho do
   lado que se afasta da câmera estreita. Razão esq/dir por faixa de rotação:
   0,813 / 0,920 / 1,021 / 1,123 / 1,292. Monotônico.
3. **Sinal 2 — distância olho-nariz**: o olho do lado que se afasta colapsa em
   direção ao nariz. Nariz bem à esquerda: 0,140 vs 0,269; bem à direita:
   0,275 vs 0,144. Ortogonal ao sinal 1, e concordante com ele em todas as
   faixas.

Aplicada ao corpus completo: **4105 imagens de perfil passam a ter exatamente
um lado lateral** (1957 direito, 2148 esquerdo), as 4147 frontais seguem com os
dois, e 82 linhas são puladas com motivo declarado (23 de direção ambígua, 59
com contagem de marcos que não corresponde a nenhum esquema — anotações
parciais ou danificadas). Uma taxa de descarte acima de 5% levanta erro, para
que uma falha sistêmica não passe como um arquivo de máscaras magro (R4).

### O que ainda cabe a você

A regra é **derivada e validada**, não a especificação publicada do dataset.
Isso está declarado no código, nos testes, no ledger e na dissertação. O que
resta é uma confirmação, não um desbloqueio:

- **Se puder**, confira no artigo original — Ashrafi, R.; Azarbayjani, M.;
  Tabkhi, H. *Charlotte-ThermalFace*, Infrared Physics & Technology, v. 124,
  p. 104209, 2022, DOI `10.1016/j.infrared.2022.104209` — se há um diagrama
  numerado dos 43 pontos do esquema de perfil. Se a convenção publicada
  divergir da derivada, ela prevalece: ajuste `LANDMARK_MAPPINGS_43` e
  `_resolve_visible_side`, e os testes de `test_landmark_mapping.py` vão
  apontar exatamente o que mudou.
- **Se não puder**, a regra se sustenta pelo argumento de consistência: é a
  única atribuição que faz `Olho direito` designar a mesma estrutura anatômica
  em quadros frontais e de perfil. Qualquer outra faria a mesma classe
  significar coisas diferentes em metade do corpus.

### Para gerar as máscaras corrigidas e reexecutar

⚠️ **Não sobrescreva os artefatos da execução relatada.** O congelamento é o
que mantém a dissertação reprodutível: `_generate_annotations` pula datasets
cujo `S{n}_polygonal_masks.json` já existe, e há um baseline de hashes em
`docs/.polygon_baseline_sha256.txt` para conferir a qualquer momento com
`sha256sum -c`.

Para uma reexecução corrigida, trabalhe sobre uma **cópia** dos dados:

```bash
# 1. copie data/ para um diretorio de trabalho novo
# 2. apague os dez S*_polygonal_masks.json dessa copia
# 3. gere as mascaras corrigidas
python -m codes.extract_data          # regenera anotacoes ausentes
# 4. force o repre-processamento e treine
python -m codes.main_pipeline --force-preprocess --models unet transunet swin
```

Considere setar `test_subjects` em `codes/config.yaml` nessa reexecução (ver
item 6), já que é o momento natural para fechar também aquela lacuna.

## 2. Escolher visualmente a amostra "cabeça inclinada" da Figura 6.5

A grade qualitativa (`fig_qualitativa.pdf`, Figura 6.5 da dissertação) tem
oito linhas planejadas; sete são selecionadas automaticamente por critério
quantitativo (distância mínima/máxima, temperatura mínima/máxima, etc. — ver
docstring de `data/a4.py`). A oitava, **`cabeca_inclinada`**, não pode ser
selecionada por código: **não existe coluna de pose de cabeça verificável
nos CSVs brutos**, e inventar um critério substituto fabricaria uma seleção
(R10 do `CLAUDE.md`). Por isso essa linha está atualmente ausente da figura
publicada — a grade tem sete linhas, não oito, e o texto não afirma o
contrário.

### Como escolher

1. Abra `data/outputs/qualitative_candidates.csv` (8442 linhas, colunas
   `sample_id, Swin-UNet++, TransUNet, U-Net, mIoU_medio, landmark_bucket,
   Distance, env-temp, subject, fold`). Ele não tem coluna de pose, então a
   triagem é visual: escolha uma dúzia de `sample_id` de diferentes sujeitos
   e inspecione as imagens térmicas correspondentes em
   `data/processed/images/<sample_id>.npy` (Celsius, use algo como
   `matplotlib.pyplot.imshow(np.load(...), cmap='inferno')` para visualizar).
2. Escolha uma amostra em que a inclinação da cabeça seja visualmente clara
   e que não seja um caso extremo de falha (para não duplicar o papel da
   linha `caso_falha`, que já existe).
3. Anote o `sample_id` escolhido (formato `S{n}/R{...}`) e abra
   `data/a4.py`; preencha:
   ```python
   MANUAL_OVERRIDE: dict[str, str | None] = {
       ...
       "cabeca_inclinada": "S3/R211045",   # substitua pelo sample_id escolhido
       ...
   }
   ```
4. Rode `python data/a4.py` a partir da raiz do repositório UBench. Ele
   reimprime a tabela de seleção (confirme que `cabeca_inclinada` agora
   aponta para o seu `sample_id`) e regrava `data/img/fig_qualitativa.pdf`
   com as oito linhas.
5. Copie o PDF atualizado para a dissertação e recompile:
   ```bash
   cp /home/doga/Documents/UBench/data/img/fig_qualitativa.pdf \
      /home/doga/Documents/Dissertac_a_o_FelipeBedinottoFava/img/resultados/
   cd /home/doga/Documents/Dissertac_a_o_FelipeBedinottoFava
   latexmk -pdf -interaction=nonstopmode dissertacao.tex
   ```
6. Ajuste a legenda da Figura 6.5 em `textuais/6_Resultados.tex` (procure por
   `fig:qualitativa_comparacao`) se a frase sobre os critérios de seleção
   precisar mencionar explicitamente a oitava linha.

---

## 3. Decidir se versiona os scripts de análise (`data/*.py`)

Todo o trabalho de remedição desta sessão — `a2` a `a12`, mais
`_common.py` — vive em `data/`, que está listado no `.gitignore` do UBench
(`data/` na linha 5). Isso é correto para os **dados** (não devem ir para o
Git), mas como consequência colateral, os **scripts** que produzem os
números agora citados na dissertação (Tabela 6.7, Figura 6.8, Apêndice D,
etc.) também não estão versionados — hoje eles só existem nesta máquina.

### Recomendação

Adicionar uma exceção ao `.gitignore` para versionar os `.py` sem incluir os
dados:

```gitignore
data/
!data/*.py
!data/img/
```

(a segunda exceção é opcional — decida se as figuras `.pdf`/`.png` geradas
também devem ir para o Git, ou se você prefere regená-las sempre a partir dos
scripts).

### Se decidir versionar

```bash
cd /home/doga/Documents/UBench
git add .gitignore data/*.py
git status --porcelain   # confira que NENHUM .npy/.csv/.json/.tiff aparece
git commit -m "chore: version the post-run analysis scripts (a1-a12, _common.py)"
```

Isso não é urgente — nada quebra por adiar —, mas sem isso, qualquer sessão
futura (ou outra pessoa) que precise reproduzir um número da dissertação
terá que recriar os scripts do zero.

---

## 4. Os 7 `\todo` remanescentes na dissertação

Dois `\todo` que já podiam ser fechados com dados desta sessão (percentual
de pixels de fundo, resultados de latência da Seção 6.7) **já foram
corrigidos e commitados** (`2dbc6b4`). Os 7 que restam genuinamente
precisam de você — nenhum é fechável com o que já foi medido nesta sessão:

| # | Arquivo:linha | O que falta |
|---|---|---|
| 1 | `textuais/4_Revisao.tex:346` | Confirmar, no Capítulo 3 (Trabalhos Relacionados), se as bases de dados e *strings* de busca da busca complementar de literatura já estão detalhadas ali; se sim, substituir a caracterização genérica atual pelos dados específicos. |
| 2 | `textuais/4_Revisao.tex:358` | Confirmar a data exata (mês/ano) da busca complementar de literatura — o texto assume "primeiro semestre de 2024" para a RSL principal, mas não tem essa data para a busca complementar. |
| 3 | `textuais/4_Revisao.tex:359` | Confirmar quais bases foram de fato consultadas na busca complementar (a RSL principal usou ACM DL, IEEE DL, PubMed Central, Scopus, arXiv — confirmar se a complementar usou o mesmo conjunto). |
| 4 | `textuais/4_Revisao.tex:360` | Confirmar quantos trabalhos adicionais a busca complementar realmente discute no Capítulo 3, para preencher a célula da tabela. |
| 5 | `textuais/3_Metodologia.tex:349` | Localizar o artigo aceito no SBCAS 2026 (mencionado em outras partes do texto como já aceito) para expandir, a partir do texto dele, a divulgação de uso de IA generativa nesta dissertação. |
| 6 | `textuais/Discussao.tex:112` | Confirmar a resolução nativa exata da câmera usada na construção do Charlotte-ThermalFace (Ashrafi et al. 2022 — mesma referência do item 1 acima), caso se queira comparar a resolução do sensor Lepton 2.5 diretamente contra a resolução bruta de captura do dataset, e não apenas contra os 256×256 usados como referência prática nesta seção. |
| 7 | `postextuais/apendices.tex:708` | Tabela de IoU por classe × arquitetura × **dobra individual** (10×3×5 células) — hoje a Tabela 6.x do Capítulo 6 reporta só a média entre as cinco dobras. Os dados por dobra individual **já existem** em `data/outputs/per_image_metrics_final_epoch.csv` (colunas incluem `fold`), então isto é factível sem nova medição — apenas uma extração/agregação e formatação de tabela que não coube no escopo desta sessão. Se quiser, posso fazer esse pivô e preencher a tabela numa próxima sessão. |

Os itens 1–6 exigem uma checagem ou busca externa que só você pode fazer. O
item 7 é uma extensão de escopo, não uma busca externa — avise se quiser que
eu monte a tabela a partir do CSV que já existe.

---

## 5. Confirmar que os repositórios remotos realmente têm o trabalho desta sessão

Ao final da sessão rodei `git fetch` nos dois repositórios e ambos
apareceram **0 ahead / 0 behind** da origem — ou seja, os commits desta
sessão parecem já estar em `origin/pos-run-updates` (UBench) e
`origin/main` (dissertação). **Eu não executei `git push` em nenhum
momento desta sessão**, e não encontrei nenhum *hook* de auto-push instalado
(o `post-commit` do UBench é só o *boilerplate* padrão do Git LFS). O reflog
do repositório da dissertação registra uma atualização de
`origin/main` por *push* às 07:38:48 de hoje, coincidindo com o horário do
último commit da sessão — mas eu não fui a origem desse push, pelo que pude
verificar.

**Ação recomendada:** confira diretamente no GitHub
(`github.com/bedinotto/Dissertac_a_o_FelipeBedinottoFava` e o repositório do
UBench) se os commits `f8a76bd`..`2dbc6b4` (dissertação) e
`da23a47`..`d63afa0` (UBench) estão realmente visíveis no branch remoto,
antes de assumir que o trabalho está publicado. Se houver algum mecanismo de
sincronização automática no seu ambiente que eu não conheço, tudo bem — só
não quero que você confie nessa confirmação sem verificar, já que não fui eu
quem enviou.

---

## 6. Itens de maior escopo, sem ação imediata necessária

Estes já estão **corretamente declarados como limitação** no texto —
listados aqui só para você ter o quadro completo, não porque exigem ação
agora:

- **HD95 / NSD** (métricas sensíveis a fronteira): ausência já declarada
  como ameaça de validade de construto em `Discussao.tex` (linha ~226) e
  como limitação metodológica no Capítulo 3. Não há pendência de texto —
  só entraria em jogo se você decidir *implementar* essas métricas, o que é
  trabalho de código novo, não de revisão.
- **`docs/phase1_realdata_checklist.md`**: validação em dados reais na
  máquina com GPU. Continua sendo o único item que nenhuma sessão em
  ambiente sandbox consegue fechar — depende da máquina física. Note que o
  item 1 deste documento (UB-29) é agora um **pré-requisito** dessa
  validação, caso você quera que a reexecução real já saia com máscaras de
  perfil corrigidas.
- **T5.3** (`CLAUDE.md`, Fase 5): os scripts de remedição `a9`–`a12` já
  cobrem boa parte do que T5.3 previa (n=10 por época final, censo de
  máscaras, VRAM isolada, figuras do Capítulo 5). O que resta de T5.3 é
  majoritariamente o item 7 da tabela acima (IoU por dobra individual).

---

## Resumo executivo (se você só tem 2 minutos)

1. **UB-29 deixou de bloquear.** Com o fato semântico que você forneceu, a
   regra de lado visível foi derivada da geometria dos marcos, validada por
   dois sinais independentes e implementada. A reexecução corrigida agora
   depende só do custo de treinamento (186,9 h), não de informação faltante.
   Se puder, confirme a convenção contra o artigo Ashrafi et al. 2022 (DOI
   `10.1016/j.infrared.2022.104209`) — se divergir, ela prevalece.
2. Escolha visualmente 1 amostra para `cabeca_inclinada` em `data/a4.py` e
   regenere a Figura 6.5 — 15 minutos de trabalho.
3. Confirme no GitHub que os commits desta sessão realmente chegaram ao
   remoto — encontrei evidência de que sim, mas não fui eu quem enviou.
4. Os demais itens (versionar `data/*.py`, 7 `\todo` de busca externa, tabela
   de IoU por dobra) podem esperar; nenhum bloqueia o que já está pronto.

---

## 7. Redução da dissertação para ~100 páginas — em andamento

**Estado: 195 → 154 páginas.** Build limpo em todos os pontos (0 erros, 0
referências ou citações indefinidas). Faltam ~54 páginas.

### Concluído

- **Apêndices A–E removidos** (−31 págs), com o conteúdo redistribuído: o
  catálogo de defeitos virou a subseção 5.9.1 da Metodologia (as 23
  referências foram redirecionadas); as tabelas de D e E foram para
  Resultados, com duas duplicatas eliminadas; A e C saíram, e seus ponteiros
  no texto foram reescritos.
- **Capítulo 7 (Discussão): 24 → 16 páginas**, a meta. O maior ganho veio de
  remover a prosa que enumerava as quinze ameaças à validade já listadas na
  tabela, e a tabela de defeitos que duplicava a nova da Metodologia.
- **Subseção 6.5.1** (a anomalia): 3470 → ~1950 palavras.
- Seis parágrafos longos do Capítulo 2 reescritos.

### O que falta, por capítulo

| Capítulo | Hoje | Alvo | Falta |
|---|---|---|---|
| 1. Introdução | 10 | 6 | −4 |
| 2. Fundamentação Teórica | 20 | 11 | −9 |
| 3. Trabalhos Relacionados | 16 | 9 | −7 |
| 4. Revisão Sistemática | 10 | 5 | −5 |
| 5. Metodologia | 24 | 14 | −10 |
| 6. Resultados | 26 | 16 | −10 |
| 8. Conclusão | 8 | 5 | −3 |

### Onde estão os alvos mais fáceis

- **Cap. 2**: a seção "Segmentação Semântica de Imagens" tem 4001 palavras --
  metade do capítulo. As subseções de U-Net/UNet++/TransUNet/Swin descrevem
  arquiteturas que o leitor de um programa de Computação já conhece; podem
  cair para o essencial que a dissertação de fato usa. As três seções médicas
  de abertura (febre, história da termometria, medição de temperatura) somam
  617 palavras e são candidatas naturais a fundir em uma só.
- **Cap. 4**: dez tabelas de protocolo de RSL (geração de \textit{string},
  bibliotecas, bases, termos, fases de seleção e extração). Consolidá-las em
  duas -- uma de protocolo, uma de funil de seleção -- deve render as 5
  páginas sem perder rastreabilidade.
- **Cap. 5 e 6**: são os que carregam a contribuição, então o corte deve vir
  de reescrita de frase, não de remoção de conteúdo. Ambos têm muitos
  parágrafos acima de 150 palavras.

### Como continuar

Peça a continuação da redução; o trabalho é incremental e cada capítulo pode
ser feito isoladamente, com `latexmk` verificando ao final que nenhuma
referência quebrou. Nenhum número, tabela ou figura foi perdido até aqui, e
esse é o critério a manter.
