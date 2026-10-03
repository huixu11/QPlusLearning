"""Generate and audit native-engine planning labels as instructor prework."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import tempfile
import time
import zipfile

from pacman_lab import ArcadeEngine, PACMAN_REVISION, ROOT, body, teacher

PREFIX = 'pacman-planner-v1'
COUNTS = {'train': 4096, 'development': 256, 'evaluation': 256}
RECIPE = {'epochs': 1, 'lr': '2e-5', 'batch': 4, 'accum': 2, 'row_budget': 0,
          'max_state': 4096, 'checkpointing': 1, 'replay': 2000,
          'p_none': 0, 'p_none_distract': 0, 'p_distract': 0, 'p_none_pair': 0, 'seed': 7}


def planner_rollout(seed=7, level=1, turns=512, planned=True):
    with ArcadeEngine() as engine:
        state = engine.request('reset', options={'seed': seed, 'level': level})
        initial = state['pellets_remaining']
        repeated, transitions, status = 0, 0, None
        for _ in range(turns):
            if planned:
                plan = engine.request('plan')
                move = plan['choice']; transitions += plan['transitions']
            else:
                move = teacher(state)
            result = engine.step(move)
            state, status = result['state'], result['outcome']
            repeated += state['visits_to_current_tile'] > 1
            if status:
                break
        return {'seed': seed, 'level': level, 'turns': state['turn'],
                'score': state['score'], 'dots_collected': initial-state['pellets_remaining'],
                'repeated_tiles': repeated, 'outcome': status or 'turn_limit',
                'simulation_frames': state['simulation_frames'], 'search_transitions': transitions}


def generate(directory, counts=COUNTS, seed=41):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    rng, seen, episode = random.Random(seed), set(), 0
    manifest = {'counts': dict(counts), 'groups': {}, 'labels': {}, 'coverage': {},
                'seed': seed, 'game_commit': PACMAN_REVISION, 'dataset': PREFIX,
                'teacher': {'algorithm': 'native-engine beam search', 'horizon_moves': 20,
                            'beam_width_per_initial_action': 8, 'future_rng_scenarios': [11117, 77717],
                            'ghost_prediction': 'Exact native deterministic personality/targeting rules in chase/scatter. RNG scenarios differ only when frightened ghosts choose turns (or native fruit duration is sampled).',
                            'optimality': 'Best surviving sequence found by bounded search; beam pruning and finite horizon preclude a full-game optimality claim.',
                            'information': 'Teacher sees native current timers/pixel offsets; learner sees structured board/ghost modes/history/elapsed frames. Future RNG is resampled independently of the episode seed. Labels are privileged-state imitation targets.'},
                'recipe': RECIPE,
                'expected_training_requests': counts['train'] + RECIPE['replay'],
                'expected_optimizer_steps': (counts['train'] + RECIPE['replay'] + 7)//8,
                'split_policy': 'Whole episodes and exact observable-state hashes are disjoint. Same classic maze; starting levels 1,2,3,5. No held-out-layout claim.',
                'collection_policy': '70% planner, 20% old maze heuristic, 10% random legal moves; snapshot every fourth move from turn 12 at two-or-more-option states; stop at first loss/clear or 768 moves.'}
    started = time.perf_counter()
    with ArcadeEngine() as engine:
        for split, count in counts.items():
            rows, groups, coverage = [], [], Counter()
            while len(rows) < count:
                episode += 1
                episode_seed = seed + episode*1009
                level = (1, 2, 3, 5)[(episode-1)%4]
                state = engine.request('reset', options={'seed': episode_seed, 'level': level})
                replay = {'seed': episode_seed, 'actions': []}
                if level != 1: replay['level'] = level
                group = f'planner-episode-{episode:05d}'
                for turn in range(768):
                    fingerprint = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
                    collect = turn >= 12 and turn%4 == 0 and len(state['legal_moves']) >= 2 and fingerprint not in seen
                    random_value = rng.random()
                    plan = engine.request('plan') if collect or random_value < .7 else None
                    if collect:
                        request = body(state); request.pop('model')
                        request['questions']['move'].update(label=plan['choice'], src='pacman_planner')
                        request['_meta'] = {'id': f'{split}-board-{len(rows):04d}', 'group_id': group,
                                            'source': 'classic_pacman_planner', 'variant': 'clean',
                                            'label_source': 'native-engine 20-move beam search',
                                            'replay': replay, 'teacher': plan}
                        rows.append(request); seen.add(fingerprint)
                        if group not in groups: groups.append(group)
                        coverage['frightened'] += state['frightened']
                        coverage['four_ghosts_outside'] += all(g['mode']=='outside' for g in state['ghosts'])
                        coverage['junctions'] += len(state['legal_moves']) >= 3
                        coverage['fruit_present'] += state['fruit'] is not None
                        coverage[state['ghost_phase']] += 1
                        coverage[f'level_{level}'] += 1
                        coverage['disagrees_with_old_heuristic'] += teacher(state) != plan['choice']
                        if len(rows) == count: break
                    move = plan['choice'] if random_value < .7 else teacher(state) if random_value < .9 else rng.choice(state['legal_moves'])
                    result = engine.step(move)
                    state, replay = result['state'], result['replay']
                    if result['outcome']: break
                print(f'{split}: {len(rows)}/{count} boards; episode {episode}; {(time.perf_counter()-started)/60:.1f} min', flush=True)
            path = directory / f'{PREFIX}-{split}.jsonl'
            path.write_text(''.join(json.dumps(row, separators=(',', ':'))+'\n' for row in rows))
            manifest['groups'][split], manifest['coverage'][split] = groups, dict(coverage)
            manifest['labels'][split] = dict(Counter(r['questions']['move']['label'] for r in rows))
    manifest['files'] = {f'{PREFIX}-{split}.jsonl': hashlib.sha256((directory/f'{PREFIX}-{split}.jsonl').read_bytes()).hexdigest() for split in counts}
    (directory/f'{PREFIX}-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def generate_parallel(directory, seed=41):
    """Instructor build: four independent collectors, then global deduplication."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pacman-collect-') as scratch:
        folders = [Path(scratch)/str(worker) for worker in range(4)]
        seeds = [seed+worker for worker in range(4)]
        sizes = {'train': 1280, 'development': 96, 'evaluation': 96}
        with ProcessPoolExecutor(max_workers=4) as pool:
            manifests = list(pool.map(generate, folders, [sizes]*4, seeds))
        manifest = {**manifests[0], 'counts': dict(COUNTS), 'groups': {}, 'labels': {}, 'coverage': {},
                    'generation_seeds': seeds, 'expected_training_requests': COUNTS['train']+RECIPE['replay'],
                    'expected_optimizer_steps': (COUNTS['train']+RECIPE['replay']+7)//8,
                    'generator_sha256': {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                         for name in ['planner_data.py', 'pacman_lab.py', 'games/arcade-planner.js', 'games/arcade-engine.js']}}
        seen = set()
        for split, count in COUNTS.items():
            sources = [[json.loads(line) for line in (folder/f'{PREFIX}-{split}.jsonl').read_text().splitlines()] for folder in folders]
            rows, coverage = [], Counter()
            for batch in zip(*sources):
                for worker, row in enumerate(batch):
                    fingerprint = hashlib.sha256(json.dumps(row['state'], sort_keys=True).encode()).hexdigest()
                    if fingerprint in seen: continue
                    seen.add(fingerprint)
                    row['_meta']['id'] = f'{split}-board-{len(rows):04d}'
                    row['_meta']['group_id'] = f'seed-{seeds[worker]}-'+row['_meta']['group_id']
                    rows.append(row)
                    state = row['state']
                    coverage['frightened'] += state['frightened']
                    coverage['four_ghosts_outside'] += all(g['mode']=='outside' for g in state['ghosts'])
                    coverage['junctions'] += len(state['legal_moves']) >= 3
                    coverage['fruit_present'] += state['fruit'] is not None
                    coverage[state['ghost_phase']] += 1
                    coverage[f"level_{state['level']}"] += 1
                    coverage['disagrees_with_old_heuristic'] += teacher(state) != row['questions']['move']['label']
                    if len(rows)==count: break
                if len(rows)==count: break
            if len(rows)!=count: raise RuntimeError(f'Insufficient unique {split} boards; collected {len(rows)}/{count}')
            (directory/f'{PREFIX}-{split}.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
            manifest['groups'][split] = sorted({r['_meta']['group_id'] for r in rows})
            manifest['coverage'][split] = dict(coverage)
            manifest['labels'][split] = dict(Counter(r['questions']['move']['label'] for r in rows))
        manifest['files'] = {f'{PREFIX}-{split}.jsonl': hashlib.sha256((directory/f'{PREFIX}-{split}.jsonl').read_bytes()).hexdigest() for split in COUNTS}
        (directory/f'{PREFIX}-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def pack(directory):
    directory = Path(directory)
    archive = directory/f'{PREFIX}.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for split in COUNTS:
            path = directory/f'{PREFIX}-{split}.jsonl'
            info = zipfile.ZipInfo(path.name, date_time=(2026, 10, 3, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            output.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return archive


def prepare_dataset(directory=ROOT/'data'):
    """Verify committed compact data and extract only the three declared files."""
    directory = Path(directory)
    manifest = json.loads((directory/f'{PREFIX}-manifest.json').read_text())
    with zipfile.ZipFile(directory/f'{PREFIX}.zip') as archive:
        if set(archive.namelist()) != set(manifest['files']):
            raise ValueError('Unexpected planner dataset archive contents')
        for name, sha in manifest['files'].items():
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != sha:
                raise ValueError('Planner dataset checksum mismatch: '+name)
            target = directory/name
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != sha:
                target.write_bytes(content)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'data')
    args = parser.parse_args()
    generate_parallel(args.output); print('Packaged:', pack(args.output))
