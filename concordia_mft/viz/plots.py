"""Cross-batch visualization helpers."""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from scipy import stats

from concordia_mft.constants import FORUM_VARIANTS


def configure_matplotlib():
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        pass
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42


def save_figure(fig, filename: str, figures_dir: str, figure_format: str = 'pdf') -> str:
    stem, _ = os.path.splitext(filename)
    out_path = os.path.join(figures_dir, f'{stem}.{figure_format}')
    fig.savefig(out_path, bbox_inches='tight', format=figure_format)
    return out_path


def plot_news_dissemination(df_runs_all, figures_dir: str, figure_format: str = 'pdf'):
    if df_runs_all.empty:
        print('Sem dados de sumário por run — gráfico 1 ignorado.')
        return None

    news_agg = df_runs_all.groupby(['news_topic_id', 'news_title'], as_index=False).agg(
        total_seed_shares=('seed_shares', 'sum'),
        mean_seed_shares=('seed_shares', 'mean'),
        std_seed_shares=('seed_shares', 'std'),
        n_runs=('seed_shares', 'count'),
        mean_seed_replies=('seed_replies', 'mean'),
        mean_engagers=('unique_engagers_count', 'mean'),
    )
    news_agg['sem_seed_shares'] = news_agg['std_seed_shares'] / np.sqrt(
        news_agg['n_runs'].clip(lower=1)
    )
    news_agg = news_agg.sort_values('mean_seed_shares', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(news_agg))))
    y_labels = [
        f"{(row['news_title'][:58] + '…') if len(str(row['news_title'])) > 58 else row['news_title']} (n={int(row['n_runs'])})"
        for _, row in news_agg.iterrows()
    ]
    xerr = news_agg['sem_seed_shares'].fillna(0)
    ax.barh(y_labels, news_agg['mean_seed_shares'], xerr=xerr, capsize=3, color='#2a6f97')
    ax.set_xlabel('Média de compartilhamentos no post-semente por execução (± SEM)')
    ax.set_title('Notícias mais disseminadas (proporcional: média por run)')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    out_path = save_figure(fig, '01_news_dissemination', figures_dir, figure_format)
    plt.show()
    print(f'Salvo: {out_path}')
    display.display(
        news_agg.sort_values('mean_seed_shares', ascending=False)[
            ['news_topic_id', 'n_runs', 'total_seed_shares', 'mean_seed_shares', 'sem_seed_shares']
        ]
    )
    return out_path


def plot_persona_shares(df_actions_all, figures_dir: str, figure_format: str = 'pdf', top_n: int = 15):
    if df_actions_all.empty:
        print('Sem dados de ações — gráfico 2 ignorado.')
        return None

    df_share = df_actions_all[
        df_actions_all['is_success'] & df_actions_all['attempted_action_type'].eq('share')
    ]
    persona_shares = df_share.groupby('actor').size().reset_index(name='successful_shares')
    persona_totals = df_actions_all.groupby('actor').size().reset_index(name='total_actions')
    persona_plot = persona_shares.merge(persona_totals, on='actor', how='left')
    persona_plot['share_rate'] = (
        persona_plot['successful_shares'] / persona_plot['total_actions'].replace(0, np.nan)
    )
    persona_plot = persona_plot.sort_values('successful_shares', ascending=False)
    top_n = min(top_n, len(persona_plot))
    plot_df = persona_plot.head(top_n).sort_values('successful_shares', ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * top_n)))
    ax.barh(plot_df['actor'], plot_df['successful_shares'], color='#e76f51')
    ax.set_xlabel('Compartilhamentos bem-sucedidos (todos os experimentos)')
    ax.set_title(f'Top {top_n} personas por compartilhamento')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    out_path = save_figure(fig, '02_persona_shares', figures_dir, figure_format)
    plt.show()
    print(f'Salvo: {out_path}')
    display.display(persona_plot.sort_values('successful_shares', ascending=False).head(20))
    return out_path


def plot_forum_variant_effect(df_runs_all, figures_dir: str, figure_format: str = 'pdf'):
    if df_runs_all.empty or 'forum_variant' not in df_runs_all.columns:
        print('Sem forum_variant nos sumários — gráfico 3 ignorado.')
        return None

    forum_df = df_runs_all.dropna(subset=['forum_variant']).copy()
    forum_order = [f['name'] for f in FORUM_VARIANTS]
    present = [f for f in forum_order if f in forum_df['forum_variant'].unique()]
    extra = sorted(set(forum_df['forum_variant'].unique()) - set(present))
    forum_order = present + extra

    forum_stats = forum_df.groupby('forum_variant').agg(
        mean_seed_shares=('seed_shares', 'mean'),
        std_seed_shares=('seed_shares', 'std'),
        mean_share_rate=('share_rate_run', 'mean'),
        std_share_rate=('share_rate_run', 'std'),
        n_runs=('seed_shares', 'count'),
    ).reindex(forum_order)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(forum_order))

    sem_shares = forum_stats['std_seed_shares'].fillna(0) / np.sqrt(
        forum_stats['n_runs'].clip(lower=1)
    )
    axes[0].bar(x, forum_stats['mean_seed_shares'], yerr=sem_shares, capsize=4, color='#264653')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(forum_order, rotation=15, ha='right')
    axes[0].set_ylabel('Média de shares no post-semente por run')
    axes[0].set_title('Compartilhamentos por tipo de fórum')
    axes[0].grid(axis='y', alpha=0.3)

    sem_rate = forum_stats['std_share_rate'].fillna(0) / np.sqrt(
        forum_stats['n_runs'].clip(lower=1)
    )
    axes[1].bar(x, forum_stats['mean_share_rate'], yerr=sem_rate, capsize=4, color='#f4a261')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(forum_order, rotation=15, ha='right')
    axes[1].set_ylabel('Taxa de compartilhamento (shares / ações possíveis)')
    axes[1].set_title('Taxa média de compartilhamento por run')
    axes[1].grid(axis='y', alpha=0.3)

    fig.suptitle('Efeito do rótulo do fórum (forum_variant)', y=1.02)
    fig.tight_layout()
    out_path = save_figure(fig, '03_forum_variant_sharing', figures_dir, figure_format)
    plt.show()
    print(f'Salvo: {out_path}')
    display.display(forum_stats.reset_index())

    groups = [
        g['share_rate_run'].dropna().values
        for _, g in forum_df.groupby('forum_variant')
        if len(g) >= 2
    ]
    if len(groups) >= 2:
        h_stat, p_val = stats.kruskal(*groups)
        print(f'Kruskal-Wallis (share_rate_run ~ forum_variant): H={h_stat:.3f}, p={p_val:.4f}')
    return out_path


def plot_similarity_vs_sharing(df_runs_all, figures_dir: str, figure_format: str = 'pdf'):
    if df_runs_all.empty:
        print('Sem dados de sumário — gráfico 4 ignorado.')
        return None

    sim_df = df_runs_all.dropna(subset=['group_cosine_similarity_pct']).copy()
    sim_df = sim_df.dropna(subset=['share_rate_run'])

    def similarity_bin(pct):
        if pct < 82:
            return 'Polarizado (<82%)'
        if pct <= 94:
            return 'Intermediário (82–94%)'
        return 'Bolha (>94%)'

    sim_df['similarity_bin'] = sim_df['group_cosine_similarity_pct'].map(similarity_bin)
    bin_order = ['Polarizado (<82%)', 'Intermediário (82–94%)', 'Bolha (>94%)']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if len(sim_df) >= 2:
        x = sim_df['group_cosine_similarity_pct'].values
        y = sim_df['share_rate_run'].values
        axes[0].scatter(x, y, alpha=0.75, color='#6a4c93', edgecolors='white', linewidths=0.5)
        if len(sim_df) >= 3:
            coef = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 50)
            axes[0].plot(x_line, np.polyval(coef, x_line), '--', color='#333333', lw=1.5)
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        axes[0].set_title(
            f'Similaridade vs taxa de compartilhamento\n'
            f'Pearson r={pearson_r:.3f} (p={pearson_p:.3f}) | '
            f'Spearman ρ={spearman_r:.3f} (p={spearman_p:.3f})'
        )
        print(
            f'Correlação (n={len(sim_df)} runs): '
            f'Pearson r={pearson_r:.3f} (p={pearson_p:.4f}), '
            f'Spearman ρ={spearman_r:.3f} (p={spearman_p:.4f})'
        )
    else:
        axes[0].set_title('Dados insuficientes para scatter')

    axes[0].set_xlabel('Similaridade vetorial do grupo (%)')
    axes[0].set_ylabel('Taxa de compartilhamento por run')
    axes[0].grid(alpha=0.3)

    box_data = [
        sim_df.loc[sim_df['similarity_bin'] == b, 'share_rate_run'].values
        for b in bin_order
        if b in sim_df['similarity_bin'].values
    ]
    box_labels = [b for b in bin_order if b in sim_df['similarity_bin'].values]
    if box_data:
        axes[1].boxplot(box_data, tick_labels=box_labels)
        axes[1].set_ylabel('Taxa de compartilhamento por run')
        axes[1].set_title('Taxa de compartilhamento por faixa de similaridade')
        axes[1].tick_params(axis='x', rotation=12)
        axes[1].grid(axis='y', alpha=0.3)
    else:
        axes[1].set_title('Sem faixas de similaridade')

    fig.tight_layout()
    out_path = save_figure(fig, '04_similarity_vs_sharing', figures_dir, figure_format)
    plt.show()
    print(f'Salvo: {out_path}')
    return out_path
