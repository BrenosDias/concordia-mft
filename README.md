# concordia-mft

Simulação de **difusão de informações em fórum de mídia social** com agentes generativos (Concordia) e perfis morais derivados do MFQ. Este repositório consome os scores produzidos pelo [`mfq-ai`](../mfq-ai) e executa experimentos assíncronos em que personas interagem via compartilhamento, voto e resposta.

O objetivo é investigar como fundamentos morais, afinidade temática e composição do grupo influenciam a disseminação de notícias em um ambiente de rede social simulada.

---

## Contexto acadêmico

Este código faz parte do Trabalho de Conclusão de Curso (TCC):

| | |
|---|---|
| **Título** | *Difusão de Informações em Redes Sociais: uma Simulação com Agentes Generativos e Fundamentos Morais* |
| **Autor** | Breno Souza Dias |
| **Curso** | Bacharelado em Ciência da Computação |
| **Instituição** | Universidade Federal Rural do Rio de Janeiro (UFRRJ) |
| **Orientadores** | Prof. Leandro Guimarães Marques Alvim, D.Sc. · Adriano Beringuy, M.Sc. |

Cada agente possui um vetor nos cinco fundamentos morais (Care, Fairness, Loyalty, Authority, Purity), obtido no pipeline MFQ, e age em um fórum assíncrono coordenado pelo Game Master do Concordia.

---

## Fluxo

```
mfq-ai: results.json  +  personas.yaml  +  mfq_mapping.json
        │
        ▼
  mft-social-media.ipynb  ──►  Ollama (deepseek-r1:8b) + Concordia
        │
        ▼
  Resultados/Experimento - N/   (JSONL, sumários, CSVs agregados)
        │
        ▼
  aggregated_all_batches/figures/  (PDFs para o TCC)
```

1. **Carregar perfis morais** — `results.json` e `personas.yaml` definem scores MFQ e identidade de cada persona.
2. **Montar agentes** — prompts incorporam fundamentos morais e regras de engajamento (share, upvote, reply, etc.).
3. **Simular fórum** — baterias de experimentos com tópicos de notícia, variantes de fórum e grupos de 5 personas.
4. **Analisar difusão** — métricas por ação, ressonância MFT e visualizações agregadas entre baterias.

---

## Visualizações

O notebook gera figuras em PDF (prontas para LaTeX) em `Resultados/aggregated_all_batches/figures/`:

- **Notícias mais disseminadas** — média de compartilhamentos no post-semente por execução
- **Personas que mais compartilharam** — contagem de ações `share` bem-sucedidas
- **Efeito do rótulo do fórum** — comparação entre Rural News, Voz Conservadora e Voz Progressista
- **Similaridade do grupo vs compartilhamento** — correlação entre coesão moral do grupo e taxa de share

A análise por experimento também exporta CSVs em `Experimento - N/aggregated/`.

---

## Estrutura do repositório

```
concordia-mft/
├── mft-social-media.ipynb   # Notebook principal (orquestra o pipeline)
├── concordia_mft/           # Lógica modular
│   ├── config.py            # Paths, modelo, parâmetros da simulação
│   ├── constants.py         # Tópicos de notícia, variantes de fórum, prompts
│   ├── embedder.py          # MpnetEmbedder (memória associativa)
│   ├── mfq.py               # Scores MFQ + load_mfq_dataframe
│   ├── agents.py            # build_concordia_agent
│   ├── mft_math.py          # Vetores MFT e similaridade cosseno
│   ├── simulation.py        # run_batch_simulation, compute_summary
│   ├── analysis.py          # analyze_diffusion (experimento atual)
│   └── viz/
│       ├── loaders.py       # Carrega todas as baterias
│       └── plots.py         # Gráficos agregados (PDF)
├── personas.yaml            # 40 personas sintéticas
├── mfq_mapping.json         # Mapeamento pergunta → fundamento moral
├── results.json             # Scores MFQ (copiado do mfq-ai)
└── social_media_prefab.py   # Prefab local de referência (Concordia upstream)
```

---

## Tecnologias

- **Python 3.10+** (Jupyter Notebook)
- **[Concordia](https://github.com/google-deepmind/concordia)** — motor de simulação multiagente e prefab de mídia social assíncrona
- **Ollama** — modelo `deepseek-r1:8b` para decisões dos agentes
- **sentence-transformers** — `all-mpnet-base-v2` para embedder do Game Master
- **Análise e visualização** — `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`

---

## Pré-requisitos e instalação

1. Repositório **[concordia](https://github.com/google-deepmind/concordia)** instalado e acessível no ambiente Python
2. Saída do **mfq-ai** copiada para este diretório: `results.json`, `personas.yaml`, `mfq_mapping.json`
3. [Ollama](https://ollama.com/) em execução (`http://localhost:11434`) com `deepseek-r1:8b`

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install concordia pandas numpy pyyaml sentence-transformers matplotlib seaborn scipy
```

Ajuste `RESULTS_BASE_DIR` em `concordia_mft/config.py` se os experimentos forem salvos em outro caminho.

---

## Uso

Abra `mft-social-media.ipynb` e execute as células em ordem:

1. **Imports e configuração** — modelo Ollama, parâmetros (`num_batches`, `batch_size`, `max_steps`)
2. **Dados e embedder** — personas, MFQ, SentenceTransformer
3. **Simulação** — `run_batch_simulation(...)` grava em `Resultados/Experimento - N/`
4. **Análise de difusão** — `analyze_diffusion(ANALYSIS_FOLDER)` no experimento atual
5. **Visualização agregada** — carrega todas as baterias e gera os 4 PDFs

Para regenerar o notebook a partir da estrutura modular:

```bash
python _rebuild_notebook.py
```

---

## Arquivos de saída

| Local | Descrição |
|-------|-----------|
| `Resultados/Experimento - N/all_actions.jsonl` | Registro completo de ações por bateria |
| `Resultados/Experimento - N/runs/run-*-summary.json` | Métricas de difusão por execução |
| `Resultados/Experimento - N/aggregated/*.csv` | Tabelas agregadas (persona_rates, correlations, etc.) |
| `Resultados/aggregated_all_batches/figures/*.pdf` | Figuras consolidadas para o TCC |

---

## Referências

- Graham, J., Haidt, J., & Nosek, B. A. (2009). *Liberals and conservatives rely on different sets of moral foundations.* Journal of Personality and Social Psychology, 96(5), 1029–1046.
- [Moral Foundations Questionnaire](https://moralfoundations.org)
- [Concordia — DeepMind](https://github.com/google-deepmind/concordia)
