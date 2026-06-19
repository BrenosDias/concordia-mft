"""Batch simulation orchestration."""

from __future__ import annotations

import datetime
import itertools
import json
import logging
import os
import random
import time

import concordia.prefabs.game_master as game_master_prefabs
import numpy as np
from concordia.associative_memory import basic_associative_memory
from concordia.environment.engines import asynchronous
from concordia.utils import async_measurements as async_measurements_lib

from concordia_mft.agents import build_concordia_agent
from concordia_mft.config import RESULTS_BASE_DIR
from concordia_mft.constants import FORUM_CALL_TO_ACTION, FORUM_VARIANTS, MFT_KEYS, NEWS_TOPICS
from concordia_mft.mft_math import cos_sim, mft_dict_to_vec


def compute_summary(
    action_rows,
    forum_state,
    persona_mft_by_name,
    news_topic,
    forum_variant,
    group_mean_mft,
    group_similarity,
):
    """Compute diffusion metrics from the append-only action registry."""
    state = forum_state.get_state()
    posts = sorted(state['posts'].values(), key=lambda p: p['post_id'])
    if not posts:
        return {}

    seed = posts[0]
    derived = posts[1:]
    successful = [a for a in action_rows if a.get('status') == 'success']

    seed_actions = [a for a in successful if a.get('target_post_id') == 0]
    seed_replies = [a for a in seed_actions if a.get('action_type') == 'reply']
    seed_upvotes = [a for a in seed_actions if a.get('action_type') == 'upvote']
    seed_downvotes = [a for a in seed_actions if a.get('action_type') == 'downvote']
    seed_shares = [a for a in seed_actions if a.get('action_type') == 'share']

    seed_post = {
        'post_id': seed['post_id'],
        'author': seed['author'],
        'upvotes': len(seed_upvotes),
        'downvotes': len(seed_downvotes),
        'votes': len(seed_upvotes) - len(seed_downvotes),
        'shares': len(seed_shares),
        'reply_count': len(seed_replies),
        'unique_repliers': sorted({a['actor'] for a in seed_replies}),
        'unique_sharers': sorted({a['actor'] for a in seed_shares}),
    }

    status_counts = {}
    action_type_counts = {}
    for a in action_rows:
        status_counts[a.get('status', 'unknown')] = status_counts.get(a.get('status', 'unknown'), 0) + 1
        action_type_counts[a.get('action_type', 'unknown')] = action_type_counts.get(a.get('action_type', 'unknown'), 0) + 1

    derived_engagement_rows = [
        a for a in successful
        if (a.get('target_post_id') is not None and a.get('target_post_id') != 0)
        or (a.get('action_type') == 'post' and a.get('created_post_id') not in (None, 0))
    ]

    cascade_depth = 0
    if seed_actions:
        cascade_depth = 1
    if derived_engagement_rows:
        cascade_depth = 2

    actions_per_persona = {
        name: {
            'post': 0, 'reply': 0, 'share': 0, 'upvote': 0, 'downvote': 0,
            'ignore': 0, 'duplicate_share': 0, 'duplicate_vote': 0,
            'reply_upvote': 0, 'reply_downvote': 0, 'quote_reply': 0, 'failed': 0,
        }
        for name in persona_mft_by_name
    }
    for a in action_rows:
        actor = a.get('actor')
        if actor not in actions_per_persona:
            continue
        effective = a.get('effective_action_type') or a.get('action_type')
        attempted = a.get('attempted_action_type') or a.get('action_type')
        counts_as_ignore = bool(a.get('counts_as_ignore'))
        duplicate = bool(a.get('duplicate_interaction'))
        target_kind = a.get('target_node_kind')
        if counts_as_ignore:
            actions_per_persona[actor]['ignore'] += 1
            if duplicate and attempted == 'share':
                actions_per_persona[actor]['duplicate_share'] += 1
            elif duplicate and attempted in ('upvote', 'downvote'):
                actions_per_persona[actor]['duplicate_vote'] += 1
            continue
        if a.get('status') == 'success' and effective in actions_per_persona[actor]:
            actions_per_persona[actor][effective] += 1
            if target_kind == 'reply':
                if effective == 'upvote':
                    actions_per_persona[actor]['reply_upvote'] += 1
                elif effective == 'downvote':
                    actions_per_persona[actor]['reply_downvote'] += 1
                elif effective == 'reply':
                    actions_per_persona[actor]['quote_reply'] += 1
        elif a.get('status') != 'success':
            actions_per_persona[actor]['failed'] += 1

    unique_engagers = sorted({
        a.get('actor') for a in successful
        if a.get('actor') in persona_mft_by_name
        and (a.get('effective_action_type') or a.get('action_type')) != 'ignore'
        and not a.get('counts_as_ignore')
    })

    duplicate_share_attempts = sum(
        1 for a in action_rows
        if a.get('duplicate_interaction') and a.get('attempted_action_type') == 'share'
    )
    duplicate_vote_attempts = sum(
        1 for a in action_rows
        if a.get('duplicate_interaction')
        and a.get('attempted_action_type') in ('upvote', 'downvote')
    )
    counts_as_ignore_total = sum(1 for a in action_rows if a.get('counts_as_ignore'))
    consecutive_all_ignore_cycles_final = forum_state.get_consecutive_all_ignore_cycles()
    ignore_cycle_threshold = forum_state.get_ignore_cycle_threshold()
    terminated_by_ignore_cycles = consecutive_all_ignore_cycles_final >= ignore_cycle_threshold

    seed_thread_engagement_rows = [
        a for a in successful
        if a.get('target_root_post_id') == 0
        and (a.get('effective_action_type') or a.get('action_type')) != 'ignore'
        and not a.get('counts_as_ignore')
    ]
    reply_target_rows = [a for a in successful if a.get('target_node_kind') == 'reply']
    reply_vote_total = sum(
        1 for a in reply_target_rows
        if (a.get('effective_action_type') or a.get('action_type')) in ('upvote', 'downvote')
    )
    quote_reply_total = sum(
        1 for a in reply_target_rows
        if (a.get('effective_action_type') or a.get('action_type')) == 'reply'
    )
    not_shareable_attempts = sum(1 for a in action_rows if a.get('status') == 'not_shareable')

    topic_vec = mft_dict_to_vec(news_topic['topic_mft'])
    forum_vec = mft_dict_to_vec(forum_variant['forum_mft'])

    persona_resonance = {}
    for name, vec in persona_mft_by_name.items():
        persona_resonance[name] = {
            'mft_vector': vec.tolist(),
            'topic_resonance': cos_sim(vec, topic_vec),
            'forum_alignment': cos_sim(vec, forum_vec),
            'actions': actions_per_persona[name],
        }

    return {
        'news_topic_id': news_topic['id'],
        'news_title': news_topic['title'],
        'topic_mft': news_topic['topic_mft'],
        'forum_variant': forum_variant['name'],
        'forum_mft': forum_variant['forum_mft'],
        'group_mean_mft': dict(zip(MFT_KEYS, [float(x) for x in group_mean_mft])),
        'group_cosine_similarity_pct': float(group_similarity),
        'seed_post': seed_post,
        'derived_post_count': len(derived),
        'derived_engagement_total': len(derived_engagement_rows),
        'cascade_depth': cascade_depth,
        'unique_engagers': unique_engagers,
        'unique_engagers_count': len(unique_engagers),
        'status_counts': status_counts,
        'action_type_counts': action_type_counts,
        'duplicate_share_attempts': duplicate_share_attempts,
        'duplicate_vote_attempts': duplicate_vote_attempts,
        'counts_as_ignore_total': counts_as_ignore_total,
        'consecutive_all_ignore_cycles_final': consecutive_all_ignore_cycles_final,
        'ignore_cycle_threshold': ignore_cycle_threshold,
        'terminated_by_ignore_cycles': terminated_by_ignore_cycles,
        'seed_thread_engagement_total': len(seed_thread_engagement_rows),
        'reply_vote_total': reply_vote_total,
        'quote_reply_total': quote_reply_total,
        'not_shareable_attempts': not_shareable_attempts,
        'persona_resonance': persona_resonance,
    }


def prepare_experiment_folder(
    base_dir: str = RESULTS_BASE_DIR,
    num_batches: int = 20,
    batch_size: int = 5,
    max_steps: int = 20,
    model=None,
):
    """Create output folder and write batch metadata."""
    os.makedirs(base_dir, exist_ok=True)

    current_folders = [
        f for f in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, f)) and f.startswith('Experimento ')
    ]
    next_num = 8
    if current_folders:
        numbers = []
        for folder in current_folders:
            try:
                numbers.append(int(folder.replace('Experimento -', '')))
            except ValueError:
                pass
        if numbers:
            next_num = max(max(numbers) + 1, 8)

    new_folder = os.path.join(base_dir, f'Experimento - {next_num}')
    os.makedirs(new_folder, exist_ok=True)
    runs_folder = os.path.join(new_folder, 'runs')
    os.makedirs(runs_folder, exist_ok=True)

    run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    all_actions_path = os.path.join(new_folder, 'all_actions.jsonl')
    metadata_path = os.path.join(new_folder, 'metadata.json')
    if os.path.exists(all_actions_path):
        os.remove(all_actions_path)

    batch_metadata = {
        'run_id': run_id,
        'created_at': datetime.datetime.now().isoformat(),
        'num_batches': num_batches,
        'batch_size': batch_size,
        'max_steps': max_steps,
        'model_name': getattr(model, 'model_name', 'deepseek-r1:8b'),
        'results_folder': new_folder,
        'runs_folder': runs_folder,
        'all_actions_path': all_actions_path,
        'news_topics': NEWS_TOPICS,
        'forum_variants': FORUM_VARIANTS,
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(batch_metadata, f, ensure_ascii=False, indent=2)

    return {
        'new_folder': new_folder,
        'runs_folder': runs_folder,
        'run_id': run_id,
        'all_actions_path': all_actions_path,
        'metadata_path': metadata_path,
    }


def run_batch_simulation(
    *,
    model,
    embedder,
    df_mft,
    yaml_data,
    num_batches: int = 20,
    batch_size: int = 5,
    max_steps: int = 20,
    base_dir: str = RESULTS_BASE_DIR,
):
    """Run the full batch of forum experiments and persist outputs."""
    print(f'\nIniciando bateria de {num_batches} experimentos aleatórios (max_steps={max_steps})...')

    paths = prepare_experiment_folder(
        base_dir=base_dir,
        num_batches=num_batches,
        batch_size=batch_size,
        max_steps=max_steps,
        model=model,
    )
    new_folder = paths['new_folder']
    runs_folder = paths['runs_folder']
    run_id = paths['run_id']
    all_actions_path = paths['all_actions_path']

    print(f'\n📂 Diretório criado para salvar resultados: {new_folder}')
    print(f'[RUN ID]: {run_id}')

    persona_list = []
    full_start_time = time.time()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
        force=True,
    )

    for batch_index in range(num_batches):
        print(f"\n{'=' * 60}")
        print(f' EXPERIMENTO {batch_index + 1} DE {num_batches} ')
        print(f"{'=' * 60}")
        batch_start_time = time.time()
        perf_t0 = time.perf_counter()

        news_topic = NEWS_TOPICS[batch_index % len(NEWS_TOPICS)]
        forum_variant = random.choice(FORUM_VARIANTS)
        forum_name = f"{forum_variant['name']} (Exp {batch_index + 1})"
        current_news = f"{news_topic['title']} — {news_topic['body']}"

        print(f"\n[TEMA]: {news_topic['title']}")
        print(f'[FÓRUM]: {forum_name}')

        if len(persona_list) < batch_size:
            print('\n[SISTEMA] Embaralhando as 40 personas para garantir participação de todos...')
            persona_list = list(yaml_data)
            random.shuffle(persona_list)

        batch_group = persona_list[:batch_size]
        persona_list = persona_list[batch_size:]

        print('\n--- COMPOSIÇÃO DO GRUPO ---')
        active_agents = []
        vetores_mft_do_grupo = []
        persona_mft_by_name = {}
        persona_metadata = {}

        for persona_yaml in batch_group:
            name = persona_yaml['name']
            mft_row = df_mft[df_mft['persona'] == name]
            if mft_row.empty:
                print(f' -> AVISO: Dados de MFT não encontrados para {name}')
                continue
            mft_scores = mft_row.iloc[0]
            vetor_mft = mft_scores[MFT_KEYS].astype(float).values
            vetores_mft_do_grupo.append(vetor_mft)
            persona_mft_by_name[name] = vetor_mft
            persona_metadata[name] = {
                'actor_mft_care': float(vetor_mft[0]),
                'actor_mft_fairness': float(vetor_mft[1]),
                'actor_mft_loyalty': float(vetor_mft[2]),
                'actor_mft_authority': float(vetor_mft[3]),
                'actor_mft_purity': float(vetor_mft[4]),
                'actor_occupation': persona_yaml.get('occupation', ''),
                'actor_location': persona_yaml.get('location', ''),
                'actor_age_range': persona_yaml.get('age_range', ''),
            }
            print(f" -> {name:<25} | Vetor: [{', '.join(f'{v:.1f}' for v in vetor_mft)}]")
            active_agents.append(
                build_concordia_agent(persona_yaml, mft_scores, current_news, model)
            )

        medias_grupo = np.zeros(len(MFT_KEYS))
        porcentagem_sim = float('nan')
        if len(vetores_mft_do_grupo) > 1:
            matriz_mft = np.array(vetores_mft_do_grupo)
            medias_grupo = np.mean(matriz_mft, axis=0)
            print('\n--- MÉDIA MORAL DO GRUPO (PERFIL DO FÓRUM) ---')
            print(
                f'Care: {medias_grupo[0]:.2f} | Fairness: {medias_grupo[1]:.2f} | '
                f'Loyalty: {medias_grupo[2]:.2f} | Authority: {medias_grupo[3]:.2f} | '
                f'Purity: {medias_grupo[4]:.2f}'
            )

            sims = [cos_sim(v1, v2) for v1, v2 in itertools.combinations(matriz_mft, 2)]
            sims = [s for s in sims if not np.isnan(s)]
            if sims:
                porcentagem_sim = float(np.mean(sims) * 100)
                tag = ''
                if porcentagem_sim > 94:
                    tag = '  (Forte Bolha Ideológica)'
                elif porcentagem_sim < 82:
                    tag = '  (Fórum Polarizado)'
                print(f'\n[MÉTRICA] Similaridade Vetorial do Grupo: {porcentagem_sim:.1f}%{tag}')

        print('\n--- INICIANDO FÓRUM ASSÍNCRONO ---')
        perf_t1 = time.perf_counter()

        gm_memory_bank = basic_associative_memory.AssociativeMemoryBank(
            sentence_embedder=embedder
        )

        gm_prefab = game_master_prefabs.async_social_media.GameMaster(
            entities=active_agents,
            params={
                'name': 'forum_rules',
                'forum_name': forum_name,
                'call_to_action': FORUM_CALL_TO_ACTION,
                'seed_posts': [{
                    'author': 'BREAKING NEWS',
                    'title': news_topic['title'],
                    'content': news_topic['body'],
                }],
                'ignore_cycle_threshold': 3,
                'measurements': async_measurements_lib.ReactiveMeasurements(),
            },
        )
        gm = gm_prefab.build(model=model, memory_bank=gm_memory_bank)
        perf_t2 = time.perf_counter()

        sim_engine = asynchronous.Asynchronous()
        sim_engine.run_loop(
            game_masters=[gm],
            entities=active_agents,
            premise='',
            max_steps=max_steps,
            verbose=True,
        )
        perf_t3 = time.perf_counter()

        print(f'\n--- FIM DO EXPERIMENTO {batch_index + 1} ---')

        forum_state = gm.get_component('__forum__')
        experiment_index = batch_index + 1
        run_prefix = f'run-{experiment_index:03d}'
        experiment_metadata = {
            'run_id': run_id,
            'experiment_index': experiment_index,
            'forum_name': forum_name,
            'forum_variant': forum_variant['name'],
            'news_topic_id': news_topic['id'],
            'news_title': news_topic['title'],
            'max_steps': max_steps,
            'batch_size': batch_size,
            'group_cosine_similarity_pct': porcentagem_sim,
        }

        actions_path = os.path.join(runs_folder, f'{run_prefix}-actions.jsonl')
        n_actions = forum_state.to_action_jsonl(
            actions_path,
            experiment_metadata=experiment_metadata,
            persona_metadata=persona_metadata,
        )
        forum_state.to_action_jsonl(
            all_actions_path,
            experiment_metadata=experiment_metadata,
            persona_metadata=persona_metadata,
            append=True,
        )
        action_rows = forum_state.get_action_log()
        print(f'💾 {n_actions} ações salvas em: {actions_path}')

        events_path = os.path.join(runs_folder, f'{run_prefix}-forum-snapshot.jsonl')
        n_events = forum_state.to_jsonl_events(events_path)
        print(f'💾 {n_events} eventos de estado salvos em: {events_path}')

        summary = compute_summary(
            action_rows=action_rows,
            forum_state=forum_state,
            persona_mft_by_name=persona_mft_by_name,
            news_topic=news_topic,
            forum_variant=forum_variant,
            group_mean_mft=medias_grupo,
            group_similarity=porcentagem_sim,
        )
        summary.update(experiment_metadata)
        summary_path = os.path.join(runs_folder, f'{run_prefix}-summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f'💾 Sumário salvo em: {summary_path}')

        if summary:
            sp = summary['seed_post']
            print(
                f"\n[DIFUSÃO] Post #0 → shares={sp['shares']}, votes={sp['votes']}, "
                f"replies={sp['reply_count']}, derived_posts={summary['derived_post_count']}, "
                f"engajamento_derivado={summary['derived_engagement_total']}, "
                f"cascade_depth={summary['cascade_depth']}, "
                f"unique_engagers={summary['unique_engagers_count']}"
            )
            print(
                f"[INATIVIDADE] ciclos_all_ignore={summary['consecutive_all_ignore_cycles_final']}/"
                f"{summary['ignore_cycle_threshold']} "
                f"(terminado por inatividade: {summary['terminated_by_ignore_cycles']}), "
                f"duplicates_share={summary['duplicate_share_attempts']}, "
                f"duplicates_vote={summary['duplicate_vote_attempts']}, "
                f"counts_as_ignore={summary['counts_as_ignore_total']}"
            )
            print(
                f"[REPLIES] seed_thread_engagement={summary['seed_thread_engagement_total']}, "
                f"reply_votes={summary['reply_vote_total']}, "
                f"quote_replies={summary['quote_reply_total']}, "
                f"not_shareable={summary['not_shareable_attempts']}"
            )

        perf_t4 = time.perf_counter()
        print(
            f'\n[PERF] experimento {batch_index + 1}: '
            f'grupo+agentes={perf_t1 - perf_t0:.2f}s, '
            f'memória_GM+prefab+build={perf_t2 - perf_t1:.2f}s, '
            f'run_loop={perf_t3 - perf_t2:.2f}s, '
            f'export+json={perf_t4 - perf_t3:.2f}s, '
            f'total_cpu_phases={perf_t4 - perf_t0:.2f}s'
        )

        batch_end_time = time.time()
        min_b, sec_b = divmod(batch_end_time - batch_start_time, 60)
        print(f'\n[TEMPO] ⏱️ Experimento {batch_index + 1} levou {int(min_b)}m {int(sec_b)}s.')

    full_end_time = time.time()
    min_t, sec_t = divmod(full_end_time - full_start_time, 60)
    print(f"\n{'=' * 60}")
    print(' SIMULAÇÃO COMPLETA FINALIZADA!')
    print(f' ⏱️ Tempo total: {int(min_t)} minutos e {int(sec_t)} segundos.')
    print(f' 📂 Resultados em: {new_folder}')
    print(f"{'=' * 60}")

    return new_folder
