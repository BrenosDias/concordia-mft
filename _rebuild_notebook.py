"""Rebuild mft-social-media.ipynb with modular imports and clean structure."""

from __future__ import annotations

import json
from pathlib import Path


def md(source: str) -> dict:
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': source if source.endswith('\n') else source + '\n',
    }


def code(source: str) -> dict:
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': source if source.endswith('\n') else source + '\n',
    }


cells = [
    md(
        '# Simulação de mídia social com Moral Foundations Theory (MFT)\n\n'
        'Notebook principal do experimento Concordia + MFQ. A lógica está em '
        '`concordia_mft/`; execute as células em ordem.'
    ),
    md('## 1. Imports'),
    code(
        'import os\n'
        'import yaml\n'
        'import glob\n'
        '\n'
        'import sentence_transformers\n'
        'from concordia.contrib.language_models.ollama import ollama_model\n'
        '\n'
        'from concordia_mft.config import (\n'
        '    DEFAULT_BATCH_SIZE,\n'
        '    DEFAULT_MAX_STEPS,\n'
        '    DEFAULT_MODEL_NAME,\n'
        '    DEFAULT_NUM_BATCHES,\n'
        '    DEFAULT_OLLAMA_HOST,\n'
        '    EMBEDDER_MODEL_NAME,\n'
        '    FIGURES_DIR_NAME,\n'
        '    FIGURES_SUBDIR,\n'
        '    PERSONAS_PATH,\n'
        '    RESULTS_BASE_DIR,\n'
        ')\n'
        'from concordia_mft.constants import FORUM_VARIANTS, MFT_KEYS, NEWS_TOPICS\n'
        'from concordia_mft.embedder import MpnetEmbedder\n'
        'from concordia_mft.mfq import load_mfq_dataframe\n'
        'from concordia_mft.simulation import run_batch_simulation\n'
        'from concordia_mft.analysis import analyze_diffusion\n'
        'from concordia_mft.viz import (\n'
        '    configure_matplotlib,\n'
        '    load_all_batches,\n'
        '    plot_forum_variant_effect,\n'
        '    plot_news_dissemination,\n'
        '    plot_persona_shares,\n'
        '    plot_similarity_vs_sharing,\n'
        ')'
    ),
    md('## 2. Configuração do modelo e parâmetros da simulação'),
    code(
        'os.environ["OLLAMA_HOST"] = DEFAULT_OLLAMA_HOST\n'
        '\n'
        'model = ollama_model.OllamaLanguageModel(model_name=DEFAULT_MODEL_NAME)\n'
        '\n'
        'num_batches = DEFAULT_NUM_BATCHES\n'
        'batch_size = DEFAULT_BATCH_SIZE\n'
        'max_steps = DEFAULT_MAX_STEPS'
    ),
    md('## 3. Carregar personas e scores MFQ'),
    code(
        'with open(PERSONAS_PATH, "r", encoding="utf-8") as f:\n'
        '    yaml_data = yaml.safe_load(f)["personas"]\n'
        '\n'
        'df_mft = load_mfq_dataframe()\n'
        'df_mft.head()'
    ),
    md('## 4. Embedder (memória associativa do game master)'),
    code(
        'st_model = sentence_transformers.SentenceTransformer(EMBEDDER_MODEL_NAME)\n'
        'embedder = MpnetEmbedder(st_model)'
    ),
    md('## 5. Executar bateria de experimentos'),
    code(
        'new_folder = run_batch_simulation(\n'
        '    model=model,\n'
        '    embedder=embedder,\n'
        '    df_mft=df_mft,\n'
        '    yaml_data=yaml_data,\n'
        '    num_batches=num_batches,\n'
        '    batch_size=batch_size,\n'
        '    max_steps=max_steps,\n'
        '    base_dir=RESULTS_BASE_DIR,\n'
        ')'
    ),
    md('## 6. Análise de difusão (experimento atual)'),
    code(
        '# Altere ANALYSIS_FOLDER para analisar uma bateria anterior.\n'
        'ANALYSIS_FOLDER = new_folder\n'
        '\n'
        'analysis_outputs = analyze_diffusion(ANALYSIS_FOLDER)'
    ),
    md(
        '## 7. Visualização agregada (todas as baterias)\n\n'
        'Carrega os JSONL de **todas** as pastas `Experimento - N` em `Resultados`, '
        'agrega os dados e gera quatro figuras:\n\n'
        '1. **Notícias mais disseminadas** — média de compartilhamentos no post-semente por execução.\n'
        '2. **Personas que mais compartilharam** — ações `share` com `status == success`.\n'
        '3. **Influência do nome do fórum** — comparação entre `forum_variant`.\n'
        '4. **Similaridade do grupo vs compartilhamento** — correlação por execução.\n\n'
        'Figuras salvas em PDF em `Resultados/aggregated_all_batches/figures/`.'
    ),
    code(
        'configure_matplotlib()\n'
        '\n'
        'FIGURES_DIR = os.path.join(RESULTS_BASE_DIR, FIGURES_DIR_NAME, FIGURES_SUBDIR)\n'
        'FIGURE_FORMAT = "pdf"\n'
        'os.makedirs(FIGURES_DIR, exist_ok=True)\n'
        '\n'
        'df_actions_all, df_runs_all, experiment_dirs = load_all_batches(RESULTS_BASE_DIR)\n'
        '\n'
        'n_runs_json = sum(\n'
        '    len(glob.glob(os.path.join(d, "runs", "run-*-summary.json")))\n'
        '    for d in experiment_dirs\n'
        ')\n'
        'print(f"Baterias carregadas: {len(experiment_dirs)}")\n'
        'print(f"Runs (summary.json): {len(df_runs_all)} (arquivos no disco: {n_runs_json})")\n'
        'print(f"Linhas de ações: {len(df_actions_all)}")\n'
        'print(f"Figuras em: {FIGURES_DIR}")'
    ),
    code(
        'plot_news_dissemination(df_runs_all, FIGURES_DIR, FIGURE_FORMAT)'
    ),
    code(
        'plot_persona_shares(df_actions_all, FIGURES_DIR, FIGURE_FORMAT)'
    ),
    code(
        'plot_forum_variant_effect(df_runs_all, FIGURES_DIR, FIGURE_FORMAT)'
    ),
    code(
        'plot_similarity_vs_sharing(df_runs_all, FIGURES_DIR, FIGURE_FORMAT)'
    ),
]

notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3',
        },
        'language_info': {
            'name': 'python',
            'pygments_lexer': 'ipython3',
        },
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

out_path = Path(__file__).parent / 'mft-social-media.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'Notebook escrito: {out_path} ({len(cells)} células)')
