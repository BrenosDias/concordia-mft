"""Default paths and simulation parameters."""

from __future__ import annotations

RESULTS_BASE_DIR = r'D:\breno\Documents\TCC\Resultados'
FIGURES_DIR_NAME = 'aggregated_all_batches'
FIGURES_SUBDIR = 'figures'

DEFAULT_NUM_BATCHES = 20
DEFAULT_BATCH_SIZE = 5
DEFAULT_MAX_STEPS = 20
DEFAULT_MODEL_NAME = 'deepseek-r1:8b'
DEFAULT_OLLAMA_HOST = 'http://localhost:11434'
EMBEDDER_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'

PERSONAS_PATH = 'personas.yaml'
MFQ_MAPPING_PATH = 'mfq_mapping.json'
RESULTS_PATH = 'results.json'
