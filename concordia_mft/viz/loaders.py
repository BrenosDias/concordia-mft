"""Load experiment batches for cross-batch visualization."""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from concordia_mft.analysis import prepare_actions_dataframe


def discover_experiment_dirs(base_dir: str) -> list[str]:
    if not os.path.isdir(base_dir):
        return []
    dirs = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and name.startswith('Experimento'):
            dirs.append(path)
    return sorted(dirs, key=lambda p: (
        int(''.join(c for c in os.path.basename(p) if c.isdigit()) or '0'),
        os.path.basename(p),
    ))


def experiment_batch_num(folder_name: str) -> int:
    digits = ''.join(c for c in folder_name if c.isdigit())
    return int(digits) if digits else 0


def load_batch_metadata(exp_dir: str) -> dict:
    meta_path = os.path.join(exp_dir, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_actions_for_experiment(exp_dir: str) -> pd.DataFrame:
    batch_name = os.path.basename(exp_dir)
    batch_num = experiment_batch_num(batch_name)
    all_path = os.path.join(exp_dir, 'all_actions.jsonl')
    if os.path.exists(all_path):
        paths = [all_path]
    else:
        paths = sorted(glob.glob(os.path.join(exp_dir, 'runs', 'run-*-actions.jsonl')))
    if not paths:
        return pd.DataFrame()

    frames = []
    for path in paths:
        df = pd.read_json(path, lines=True)
        if df.empty:
            continue
        df['experiment_batch'] = batch_name
        df['experiment_batch_num'] = batch_num
        df['source_file'] = os.path.basename(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_run_summaries(exp_dir: str) -> pd.DataFrame:
    batch_name = os.path.basename(exp_dir)
    batch_num = experiment_batch_num(batch_name)
    meta = load_batch_metadata(exp_dir)
    batch_size = meta.get('batch_size')
    max_steps = meta.get('max_steps')
    rows = []
    for path in sorted(glob.glob(os.path.join(exp_dir, 'runs', 'run-*-summary.json'))):
        with open(path, encoding='utf-8') as f:
            summary = json.load(f)
        seed = summary.get('seed_post') or {}
        rows.append({
            'experiment_batch': batch_name,
            'experiment_batch_num': batch_num,
            'run_file': os.path.basename(path),
            'experiment_index': summary.get('experiment_index'),
            'forum_variant': summary.get('forum_variant'),
            'forum_name': summary.get('forum_name'),
            'news_topic_id': summary.get('news_topic_id'),
            'news_title': summary.get('news_title'),
            'group_cosine_similarity_pct': summary.get('group_cosine_similarity_pct'),
            'seed_shares': seed.get('shares', 0),
            'seed_replies': seed.get('reply_count', 0),
            'seed_upvotes': seed.get('upvotes', 0),
            'seed_downvotes': seed.get('downvotes', 0),
            'unique_engagers_count': summary.get('unique_engagers_count'),
            'derived_post_count': summary.get('derived_post_count'),
            'batch_size': batch_size,
            'max_steps': max_steps,
        })
    return pd.DataFrame(rows)


def attach_run_share_rates(df_runs: pd.DataFrame, df_actions: pd.DataFrame) -> pd.DataFrame:
    if df_runs.empty:
        return df_runs
    out = df_runs.copy()
    action_counts = (
        df_actions.groupby(['experiment_batch', 'experiment_index'])
        .size()
        .reset_index(name='total_actions_in_run')
    )
    out = out.merge(action_counts, on=['experiment_batch', 'experiment_index'], how='left')
    denom = out['batch_size'] * out['max_steps']
    out['share_rate_run'] = out['seed_shares'] / denom.where(denom.notna() & (denom > 0))
    fallback = out['total_actions_in_run'].replace(0, np.nan)
    out['share_rate_run'] = out['share_rate_run'].fillna(out['seed_shares'] / fallback)
    return out


def load_all_batches(results_base_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load actions and run summaries from all experiment folders."""
    experiment_dirs = discover_experiment_dirs(results_base_dir)
    if not experiment_dirs:
        raise FileNotFoundError(
            f"Nenhuma pasta 'Experimento - *' encontrada em: {results_base_dir}"
        )

    action_frames = []
    summary_frames = []
    for exp_dir in experiment_dirs:
        df_a = load_actions_for_experiment(exp_dir)
        if not df_a.empty:
            action_frames.append(df_a)
        df_s = load_run_summaries(exp_dir)
        if not df_s.empty:
            summary_frames.append(df_s)

    if not action_frames and not summary_frames:
        raise FileNotFoundError('Nenhum JSONL de ações ou sumário encontrado nas baterias.')

    df_actions_all = prepare_actions_dataframe(
        pd.concat(action_frames, ignore_index=True) if action_frames else pd.DataFrame()
    )
    df_runs_all = (
        pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    )
    if not df_actions_all.empty and not df_runs_all.empty:
        df_runs_all = attach_run_share_rates(df_runs_all, df_actions_all)
    elif not df_runs_all.empty:
        df_runs_all['share_rate_run'] = np.nan

    return df_actions_all, df_runs_all, experiment_dirs
