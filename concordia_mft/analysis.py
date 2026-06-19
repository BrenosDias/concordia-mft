"""Action-level diffusion analysis for a single experiment folder."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from IPython import display

from concordia_mft.constants import FORUM_VARIANTS, NEWS_TOPICS
from concordia_mft.mft_math import cos_sim, mft_dict_to_vec


def _bin_mft_value(v: float) -> str:
    if v < 2.5:
        return 'low'
    if v < 3.75:
        return 'mid'
    return 'high'


def _row_mft_vec(row) -> np.ndarray:
    return np.array([
        row['actor_mft_care'],
        row['actor_mft_fairness'],
        row['actor_mft_loyalty'],
        row['actor_mft_authority'],
        row['actor_mft_purity'],
    ], dtype=np.float64)


def prepare_actions_dataframe(df_actions: pd.DataFrame) -> pd.DataFrame:
    """Normalize action columns and derive engagement flags."""
    if df_actions.empty:
        return df_actions

    if 'counts_as_ignore' not in df_actions.columns:
        df_actions['counts_as_ignore'] = False
    if 'duplicate_interaction' not in df_actions.columns:
        df_actions['duplicate_interaction'] = False
    if 'attempted_action_type' not in df_actions.columns:
        df_actions['attempted_action_type'] = df_actions['action_type']
    if 'effective_action_type' not in df_actions.columns:
        df_actions['effective_action_type'] = df_actions['action_type']
    if 'consecutive_all_ignore_cycles' not in df_actions.columns:
        df_actions['consecutive_all_ignore_cycles'] = 0
    if 'target_node_kind' not in df_actions.columns:
        df_actions['target_node_kind'] = None
    if 'target_root_post_id' not in df_actions.columns:
        df_actions['target_root_post_id'] = df_actions.get('target_post_id')
    if 'parent_reply_id' not in df_actions.columns:
        df_actions['parent_reply_id'] = None

    df_actions['counts_as_ignore'] = df_actions['counts_as_ignore'].fillna(False).astype(bool)
    df_actions['duplicate_interaction'] = df_actions['duplicate_interaction'].fillna(False).astype(bool)
    df_actions['target_root_post_id'] = df_actions['target_root_post_id'].where(
        df_actions['target_root_post_id'].notna(),
        df_actions['target_post_id'],
    )

    df_actions['is_success'] = df_actions['status'].eq('success')
    df_actions['is_seed_action'] = df_actions['target_post_id'].eq(0)
    df_actions['is_seed_thread'] = df_actions['target_root_post_id'].eq(0)
    df_actions['is_reply_target'] = df_actions['target_node_kind'].eq('reply')
    df_actions['is_quote_reply'] = (
        df_actions['parent_reply_id'].notna()
        & df_actions['effective_action_type'].eq('reply')
        & df_actions['is_success']
    )
    df_actions['is_effective_ignore'] = (
        df_actions['effective_action_type'].eq('ignore') | df_actions['counts_as_ignore']
    )
    df_actions['is_engagement'] = df_actions['is_success'] & ~df_actions['is_effective_ignore']
    df_actions['is_real_failure'] = ~df_actions['is_success'] & ~df_actions['counts_as_ignore']
    return df_actions


def add_resonance_columns(df_actions: pd.DataFrame) -> pd.DataFrame:
    topic_mft_by_id = {t['id']: mft_dict_to_vec(t['topic_mft']) for t in NEWS_TOPICS}
    forum_mft_by_name = {f['name']: mft_dict_to_vec(f['forum_mft']) for f in FORUM_VARIANTS}

    df_actions['topic_resonance'] = df_actions.apply(
        lambda row: cos_sim(_row_mft_vec(row), topic_mft_by_id[row['news_topic_id']]),
        axis=1,
    )
    df_actions['forum_alignment'] = df_actions.apply(
        lambda row: cos_sim(_row_mft_vec(row), forum_mft_by_name[row['forum_variant']]),
        axis=1,
    )
    return df_actions


def analyze_diffusion(analysis_folder: str) -> dict[str, pd.DataFrame]:
    """Run the full diffusion analysis pipeline and export CSV tables."""
    all_actions_file = os.path.join(analysis_folder, 'all_actions.jsonl')
    if not os.path.exists(all_actions_file):
        raise FileNotFoundError(f'No action registry found at: {all_actions_file}')

    df_actions = pd.read_json(all_actions_file, lines=True)
    print(f'Carregadas {len(df_actions)} ações de {all_actions_file}\n')

    df_actions = prepare_actions_dataframe(df_actions)
    df_actions = add_resonance_columns(df_actions)

    print('=== Dataset principal ===')
    display.display(df_actions.head())

    print('\n=== 1) Distribuição de ações por tipo e status ===')
    action_distribution = df_actions.groupby(
        ['action_type', 'attempted_action_type', 'status']
    ).size().reset_index(name='n')
    display.display(action_distribution)

    persona_action_distribution = df_actions.groupby(
        ['actor', 'action_type', 'attempted_action_type', 'status']
    ).size().reset_index(name='n')

    print('\n=== 2) Taxa de compartilhamento por persona ===')
    persona_rates = df_actions.groupby('actor').agg(
        total_actions=('action_seq', 'count'),
        successful_actions=('is_success', 'sum'),
        shares=('attempted_action_type', lambda s: (s == 'share').sum()),
        successful_shares=(
            'attempted_action_type',
            lambda s: ((s == 'share') & df_actions.loc[s.index, 'is_success']).sum(),
        ),
        seed_shares=(
            'attempted_action_type',
            lambda s: (
                (s == 'share')
                & df_actions.loc[s.index, 'is_success']
                & df_actions.loc[s.index, 'is_seed_action']
            ).sum(),
        ),
        replies=(
            'attempted_action_type',
            lambda s: ((s == 'reply') & df_actions.loc[s.index, 'is_success']).sum(),
        ),
        posts=(
            'attempted_action_type',
            lambda s: ((s == 'post') & df_actions.loc[s.index, 'is_success']).sum(),
        ),
        duplicate_share_attempts=(
            'duplicate_interaction',
            lambda s: (s & df_actions.loc[s.index, 'attempted_action_type'].eq('share')).sum(),
        ),
        duplicate_vote_attempts=(
            'duplicate_interaction',
            lambda s: (
                s & df_actions.loc[s.index, 'attempted_action_type'].isin(['upvote', 'downvote'])
            ).sum(),
        ),
        counts_as_ignore=('counts_as_ignore', 'sum'),
        effective_ignores=('is_effective_ignore', 'sum'),
        failed_actions=('is_real_failure', 'sum'),
        avg_topic_resonance=('topic_resonance', 'mean'),
        avg_forum_alignment=('forum_alignment', 'mean'),
    ).reset_index()
    persona_rates['share_rate'] = persona_rates['successful_shares'] / persona_rates['total_actions'].replace(0, np.nan)
    persona_rates['seed_share_rate'] = persona_rates['seed_shares'] / persona_rates['total_actions'].replace(0, np.nan)
    display.display(persona_rates.sort_values('successful_shares', ascending=False))

    print('\n=== 3) Difusão do post-semente por tópico e fórum ===')
    seed_diffusion = df_actions[df_actions['is_seed_action'] & df_actions['is_success']].groupby(
        ['news_topic_id', 'forum_variant', 'action_type']
    ).size().reset_index(name='n')
    display.display(seed_diffusion)

    seed_summary = df_actions[df_actions['is_seed_action'] & df_actions['is_success']].groupby(
        ['news_topic_id', 'forum_variant']
    ).agg(
        seed_actions=('action_seq', 'count'),
        seed_shares=('action_type', lambda s: (s == 'share').sum()),
        seed_replies=('action_type', lambda s: (s == 'reply').sum()),
        seed_upvotes=('action_type', lambda s: (s == 'upvote').sum()),
        seed_downvotes=('action_type', lambda s: (s == 'downvote').sum()),
        unique_engagers=('actor', 'nunique'),
    ).reset_index()
    display.display(seed_summary)

    print('\n=== 4) Taxa de erro por persona (excluindo duplicatas) ===')
    error_rates = df_actions.groupby('actor').agg(
        total_actions=('action_seq', 'count'),
        failed_actions=('is_real_failure', 'sum'),
        duplicate_attempts=('duplicate_interaction', 'sum'),
    ).reset_index()
    error_rates['error_rate'] = error_rates['failed_actions'] / error_rates['total_actions'].replace(0, np.nan)
    display.display(error_rates.sort_values('error_rate', ascending=False))

    status_by_topic = df_actions.groupby(['news_topic_id', 'status']).size().reset_index(name='n')

    print('\n=== 4.1) Tentativas duplicadas (share/voto) por persona ===')
    duplicate_breakdown = df_actions[df_actions['duplicate_interaction']].groupby(
        ['actor', 'attempted_action_type']
    ).size().reset_index(name='n')
    display.display(duplicate_breakdown)

    print('\n=== 4.2) Encerramento por inatividade (ciclos all-ignore) ===')
    if 'experiment_index' in df_actions.columns:
        termination_summary = df_actions.groupby(
            ['experiment_index', 'forum_variant', 'news_topic_id']
        ).agg(
            total_actions=('action_seq', 'count'),
            counts_as_ignore_total=('counts_as_ignore', 'sum'),
            duplicate_share_attempts=(
                'duplicate_interaction',
                lambda s: (s & df_actions.loc[s.index, 'attempted_action_type'].eq('share')).sum(),
            ),
            duplicate_vote_attempts=(
                'duplicate_interaction',
                lambda s: (
                    s & df_actions.loc[s.index, 'attempted_action_type'].isin(['upvote', 'downvote'])
                ).sum(),
            ),
            consecutive_all_ignore_cycles_final=('consecutive_all_ignore_cycles', 'max'),
        ).reset_index()
        termination_summary['terminated_by_ignore_cycles'] = (
            termination_summary['consecutive_all_ignore_cycles_final'] >= 2
        )
        display.display(termination_summary)
    else:
        termination_summary = pd.DataFrame()
        print('(coluna experiment_index ausente — dataset antigo)')

    print('\n=== 5) Ressonância MFT vs engajamento ===')
    engagement_by_actor_topic = df_actions.groupby(['actor', 'news_topic_id', 'forum_variant']).agg(
        actions=('action_seq', 'count'),
        engagements=('is_engagement', 'sum'),
        shares=('action_type', lambda s: ((s == 'share') & df_actions.loc[s.index, 'is_success']).sum()),
        topic_resonance=('topic_resonance', 'mean'),
        forum_alignment=('forum_alignment', 'mean'),
    ).reset_index()

    corr_cols = ['topic_resonance', 'forum_alignment', 'actions', 'engagements', 'shares']
    correlations = engagement_by_actor_topic[corr_cols].corr()
    print('Correlação com engajamento/compartilhamento:')
    display.display(correlations.loc[['topic_resonance', 'forum_alignment'], ['engagements', 'shares']].round(3))

    print('\n=== 6) Compartilhamento por faixa de fundamento moral ===')
    mft_propensity_tables = []
    for foundation, column in {
        'care': 'actor_mft_care',
        'fairness': 'actor_mft_fairness',
        'loyalty': 'actor_mft_loyalty',
        'authority': 'actor_mft_authority',
        'purity': 'actor_mft_purity',
    }.items():
        tmp = df_actions.copy()
        tmp[f'{foundation}_bin'] = tmp[column].map(_bin_mft_value)
        table = tmp.groupby(f'{foundation}_bin').agg(
            actions=('action_seq', 'count'),
            engagements=('is_engagement', 'sum'),
            shares=('action_type', lambda s: ((s == 'share') & tmp.loc[s.index, 'is_success']).sum()),
        ).reset_index()
        table['foundation'] = foundation
        table['share_rate'] = table['shares'] / table['actions'].replace(0, np.nan)
        mft_propensity_tables.append(table)
        print(f'\n-- {foundation.upper()} --')
        display.display(table)

    mft_propensity = pd.concat(mft_propensity_tables, ignore_index=True)

    print('\n=== 7) Engajamento em respostas (reply-as-target) ===')
    reply_actions = df_actions[df_actions['is_reply_target']]
    if not reply_actions.empty:
        reply_engagement = reply_actions.groupby('actor').agg(
            reply_upvotes=(
                'effective_action_type',
                lambda s: ((s == 'upvote') & reply_actions.loc[s.index, 'is_success']).sum(),
            ),
            reply_downvotes=(
                'effective_action_type',
                lambda s: ((s == 'downvote') & reply_actions.loc[s.index, 'is_success']).sum(),
            ),
            quote_replies=('is_quote_reply', 'sum'),
            not_shareable_attempts=('status', lambda s: (s == 'not_shareable').sum()),
        ).reset_index()
        display.display(reply_engagement.sort_values(
            ['quote_replies', 'reply_upvotes', 'reply_downvotes'], ascending=False
        ))
    else:
        reply_engagement = pd.DataFrame(
            columns=['actor', 'reply_upvotes', 'reply_downvotes', 'quote_replies', 'not_shareable_attempts']
        )
        print('(nenhuma ação direcionada a respostas no dataset)')

    print('\n=== 8) Estrutura de threads por experimento ===')
    if 'experiment_index' in df_actions.columns:
        successful_replies = df_actions[
            df_actions['is_success'] & df_actions['effective_action_type'].eq('reply')
        ]
        thread_structure = successful_replies.groupby(
            ['experiment_index', 'forum_variant', 'news_topic_id']
        ).agg(
            total_replies=('action_seq', 'count'),
            top_level_replies=('parent_reply_id', lambda s: s.isna().sum()),
            quote_replies=('parent_reply_id', lambda s: s.notna().sum()),
            unique_repliers=('actor', 'nunique'),
        ).reset_index()
        thread_structure['quote_reply_share'] = (
            thread_structure['quote_replies']
            / thread_structure['total_replies'].replace(0, np.nan)
        )
        display.display(thread_structure)
    else:
        thread_structure = pd.DataFrame()
        print('(coluna experiment_index ausente — dataset antigo)')

    print('\n=== 9) Engajamento na thread do post-semente (root_post_id == 0) ===')
    seed_thread_engagement = df_actions[
        df_actions['is_seed_thread'] & df_actions['is_engagement']
    ].groupby(['actor', 'effective_action_type', 'target_node_kind']).size().reset_index(name='n')
    display.display(seed_thread_engagement)

    agg_dir = os.path.join(analysis_folder, 'aggregated')
    os.makedirs(agg_dir, exist_ok=True)
    df_actions.to_csv(os.path.join(agg_dir, 'all_actions.csv'), index=False)
    action_distribution.to_csv(os.path.join(agg_dir, 'action_distribution.csv'), index=False)
    persona_action_distribution.to_csv(os.path.join(agg_dir, 'persona_action_distribution.csv'), index=False)
    persona_rates.to_csv(os.path.join(agg_dir, 'persona_rates.csv'), index=False)
    seed_diffusion.to_csv(os.path.join(agg_dir, 'seed_diffusion.csv'), index=False)
    seed_summary.to_csv(os.path.join(agg_dir, 'seed_summary.csv'), index=False)
    error_rates.to_csv(os.path.join(agg_dir, 'error_rates.csv'), index=False)
    status_by_topic.to_csv(os.path.join(agg_dir, 'status_by_topic.csv'), index=False)
    engagement_by_actor_topic.to_csv(os.path.join(agg_dir, 'engagement_by_actor_topic.csv'), index=False)
    mft_propensity.to_csv(os.path.join(agg_dir, 'mft_propensity.csv'), index=False)
    correlations.to_csv(os.path.join(agg_dir, 'correlations.csv'))
    duplicate_breakdown.to_csv(os.path.join(agg_dir, 'duplicate_breakdown.csv'), index=False)
    if not termination_summary.empty:
        termination_summary.to_csv(os.path.join(agg_dir, 'termination_summary.csv'), index=False)
    reply_engagement.to_csv(os.path.join(agg_dir, 'reply_engagement.csv'), index=False)
    if not thread_structure.empty:
        thread_structure.to_csv(os.path.join(agg_dir, 'thread_structure.csv'), index=False)
    seed_thread_engagement.to_csv(os.path.join(agg_dir, 'seed_thread_engagement.csv'), index=False)
    print(f'\n💾 Tabelas agregadas salvas em: {agg_dir}')

    return {
        'df_actions': df_actions,
        'action_distribution': action_distribution,
        'persona_rates': persona_rates,
        'correlations': correlations,
    }
