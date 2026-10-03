"""CPU checks of native planning, restoration, data isolation and task recipe."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cloud_runtime import CloudRuntime, BASE_MODEL, BASE_REVISION
from pacman_lab import ArcadeEngine, ROOT, body, destination, distances, position, teacher
from planner_data import COUNTS, PREFIX, RECIPE, prepare_dataset
from test_cloud_runtime import checkpoint_files


class PlannerTests(unittest.TestCase):
    def test_search_restores_hidden_state_and_episode_rng_before_real_move(self):
        rows = [json.loads(line) for line in (ROOT/'data/pacman-arcade-train.jsonl').read_text().splitlines()]
        # Include frightened, all-four-out and late-game states. A fresh replay
        # is independent of the planner's native rewind hooks.
        with ArcadeEngine() as planned, ArcadeEngine() as control:
            for record in rows[::4]:
                replay = record['_meta']['replay']
                before = planned.request('replay', replay=replay)
                control.request('replay', replay=replay)
                decision = planned.request('plan', options={'horizon': 8, 'width': 4})
                self.assertEqual(planned.request('observe'), before)
                self.assertEqual({c['action'] for c in decision['candidates']}, set(before['legal_moves']))
                move = teacher(before)
                self.assertEqual(planned.step(move), control.step(move))

    def test_exhaustive_short_search_matches_independent_replay_enumeration(self):
        with ArcadeEngine() as engine:
            initial = engine.reset(91)
            plan = engine.request('plan', options={'horizon': 4, 'width': 0})
            self.assertTrue(plan['exhaustive'])
            self.assertEqual(plan['pruned_nodes'], 0)
            best = {}
            def enumerate_paths(sequence, loops=0):
                state = engine.request('replay', replay={'seed': 91, 'actions': sequence})
                if len(sequence)==4:
                    distance = distances(state['maze'], position(state['player']))
                    nearest = min(distance.get((r,c),100) for r,line in enumerate(state['maze']) for c,ch in enumerate(line) if ch in '.o')
                    danger = [distance.get(position(g),16) for g in state['ghosts'] if g['mode']=='outside' and not g['frightened']]
                    value = state['score']+5*(244-state['pellets_remaining'])-4*loops-6*nearest+2*min([8,*danger])
                    best[sequence[0]] = max(best.get(sequence[0],-float('inf')),value)
                    return
                for move in state['legal_moves']:
                    engine.request('replay', replay={'seed': 91, 'actions': sequence})
                    result = engine.step(move)
                    self.assertIsNone(result['outcome'])
                    enumerate_paths(sequence+[move],loops+min(4,max(0,result['state']['visits_to_current_tile']-1)))
            enumerate_paths([])
            self.assertEqual(best, {c['action']:c['value'] for c in plan['candidates']})
            for candidate in plan['candidates']:
                self.assertEqual(candidate['scenarios'][0], candidate['scenarios'][1])

    def test_native_release_eventually_puts_all_four_ghosts_outside(self):
        with ArcadeEngine() as engine:
            state = engine.reset()
            self.assertEqual([g['mode'] for g in state['ghosts']], ['outside','pacing_home','pacing_home','pacing_home'])
            for _ in range(350):
                move = 'left' if state['player']['heading']=='right' else 'right'
                result = engine.step(move if move in state['legal_moves'] else state['legal_moves'][0])
                state = result['state']
                if all(g['mode']=='outside' for g in state['ghosts']): break
                if result['outcome']: break
            self.assertTrue(all(g['mode']=='outside' for g in state['ghosts']))

    def test_planner_recipe_preserves_lora_and_matches_intermediate_stage_scale(self):
        meta = {'base': BASE_MODEL, 'base_revision': BASE_REVISION, 'lora': 16,
                'head_dim': 256, 'option_isolation': False, 'special_embeddings': False,
                'weights_dtype': 'fp32', 'weights': 'lora'}
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder)
            parent = Path(folder)/'skills'; checkpoint_files(parent)
            with patch('cloud_runtime.subprocess.check_output', return_value=json.dumps(meta)):
                command = runtime.finetuning_command('reviewed.jsonl','new-planner',parent)
        flags = dict(zip(command[3::2],command[4::2]))
        for key,value in RECIPE.items(): self.assertEqual(flags['--'+key],str(value))
        self.assertEqual(flags['--suite'],'evals/v7/decision-v7')
        self.assertEqual(flags['--lora_targets'],'all')
        self.assertEqual(flags['--lora'],'16')
        self.assertEqual(flags['--head_dim'],'256')
        self.assertEqual(flags['--max_steps'],'0')
        self.assertEqual((COUNTS['train']+RECIPE['replay'])//8,762)

    def test_packaged_dataset_has_legal_targets_and_no_episode_or_input_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for name in [f'{PREFIX}.zip',f'{PREFIX}-manifest.json']:
                (target/name).write_bytes((ROOT/'data'/name).read_bytes())
            manifest = prepare_dataset(target)
            episodes, fingerprints = set(), set()
            for split,count in COUNTS.items():
                rows = [json.loads(line) for line in (target/f'{PREFIX}-{split}.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows),count)
                groups = {r['_meta']['group_id'] for r in rows}
                self.assertFalse(episodes & groups); episodes.update(groups)
                for row in rows:
                    state, q = row['state'],row['questions']['move']
                    self.assertIn(q['label'],state['legal_moves'])
                    self.assertEqual(q['label'],row['_meta']['teacher']['choice'])
                    self.assertEqual({c['action'] for c in row['_meta']['teacher']['candidates']},set(state['legal_moves']))
                    request = body(state)
                    self.assertEqual(q['instructions'],request['questions']['move']['instructions'])
                    self.assertNotIn('_meta',request)
                    self.assertNotIn('label',request['questions']['move'])
                    self.assertNotIn('teacher',state)
                    fingerprint = json.dumps(state,sort_keys=True)
                    self.assertNotIn(fingerprint,fingerprints); fingerprints.add(fingerprint)
            self.assertGreater(manifest['coverage']['train']['four_ghosts_outside'],0)
            self.assertGreater(manifest['coverage']['train']['frightened'],0)


if __name__=='__main__': unittest.main()
