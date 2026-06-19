"""MFQ scoring utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from concordia_mft.config import MFQ_MAPPING_PATH, RESULTS_PATH


def load_mapping(path: str | Path = MFQ_MAPPING_PATH) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def compute_mfq_scores(answers: dict, mapping: dict) -> dict:
    scores = {}
    for foundation, questions in mapping.items():
        values = []
        for q in questions:
            if q in answers:
                values.append(answers[q])
        scores[foundation] = np.mean(values)
    return scores


def analyze_results(results: list, mapping: dict) -> list:
    analysis = []
    for r in results:
        scores = compute_mfq_scores(r['answers'], mapping)
        analysis.append({'persona': r['persona'], **scores})
    return analysis


def load_mfq_dataframe(results_path: str | Path = RESULTS_PATH, mapping_path: str | Path = MFQ_MAPPING_PATH):
    """Load MFQ results and return a cleaned pandas DataFrame."""
    import pandas as pd

    with open(results_path, encoding='utf-8') as f:
        json_results = json.load(f)
    mapping = load_mapping(mapping_path)
    analysis = analyze_results(json_results, mapping)
    df_mft = pd.DataFrame(analysis)
    return df_mft.dropna(subset=['care', 'fairness', 'loyalty', 'authority', 'purity'])
