"""Check the actual pinned arcade engine, replay datasets and adapter bridge."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch, Mock

from pacman_lab import *


def rule_predict(request):
    selected = teacher(request['state'])
    return {'answers': {'move': {'choice': selected,
            'probabilities': {k: float(k == selected) for k in request['questions']['move']['criteria']}}}}


class PlayerLabTests(unittest.TestCase):
    def test_original_classic_map_four_ghosts_pellets_lives_and_score(self):
        with ArcadeEngine() as engine:
            state = engine.reset()
            self.assertEqual((len(state['maze']), len(state['maze'][0])), (36, 28))
            self.assertEqual([g['name'] for g in state['ghosts']], ['blinky', 'pinky', 'inky', 'clyde'])
            self.assertEqual(state['pellets_remaining'], 244)
            self.assertEqual(sum(row.count('o') for row in state['maze']), 4)
            self.assertEqual(state['lives'], 3)
            self.assertEqual(state['legal_moves'], ['left', 'right'])
            with self.assertRaisesRegex(RuntimeError, 'Illegal arcade'):
                engine.step('up')
            for _ in range(6):
                result = engine.step('left')
            self.assertGreater(result['state']['score'], 0)
            self.assertLess(result['state']['pellets_remaining'], 244)
            self.assertGreater(result['action_frames'], 0)

    def test_generated_episode_splits_and_exact_upstream_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = make_data(directory)
            groups = [set(v) for v in manifest['groups'].values()]
            self.assertFalse(any(a & b for i, a in enumerate(groups) for b in groups[i+1:]))
            states = set()
            phases, powers, all_outside = set(), 0, 0
            with ArcadeEngine() as engine:
                for split, count in manifest['counts'].items():
                    content = (Path(directory) / f'{DATA_PREFIX}-{split}.jsonl').read_bytes()
                    self.assertEqual(content, (ROOT / 'data' / f'{DATA_PREFIX}-{split}.jsonl').read_bytes())
                    rows = [json.loads(line) for line in content.splitlines()]
                    self.assertEqual(len(rows), count)
                    for row in rows:
                        state = row['state']
                        self.assertEqual(engine.request('replay', replay=row['_meta']['replay']), state)
                        q = row['questions']['move']
                        self.assertIn(q['label'], state['legal_moves'])
                        request = inference_request(row)
                        self.assertNotIn('_meta', request)
                        self.assertNotIn('replay', request['state'])
                        self.assertEqual(set(request['questions']['move']), {'type', 'instructions', 'criteria'})
                        fingerprint = json.dumps(state, sort_keys=True)
                        self.assertNotIn(fingerprint, states)
                        states.add(fingerprint)
                        phases.add(state['ghost_phase'])
                        powers += state['frightened']
                        all_outside += all(g['mode'] == 'outside' for g in state['ghosts'])
            self.assertEqual(phases, {'scatter', 'chase'})
            self.assertGreater(powers, 0)
            self.assertGreater(all_outside, 0)

    def test_native_power_pellet_and_tunnel_mechanisms(self):
        hook = '''
globalThis.arcadeFixture = function(kind) {
 labArcade.reset({seed:7,skipReady:true});
 if(kind==='power') {pacman.setPos(1*tileSize+midTile.x, 7*tileSize+midTile.y); pacman.setDir(DIR_UP); return labArcade.step('up');}
 if(kind==='tunnel') {pacman.setPos(midTile.x, 17*tileSize+midTile.y); pacman.setDir(DIR_LEFT); return labArcade.step('left');}
};
'''
        script = engine_script()
        end = script.rfind('})();')
        script = script[:end] + hook + script[end:]
        worker = (ROOT / 'games/arcade-worker.js').read_text()
        prefix = worker[:worker.index('readline.createInterface')]
        harness = prefix + "process.stdout.write(JSON.stringify(['power','tunnel'].map(kind=>context.arcadeFixture(kind))));"
        with tempfile.TemporaryDirectory() as directory:
            game, runner = Path(directory) / 'engine.js', Path(directory) / 'fixture.js'
            game.write_text(script); runner.write_text(harness)
            power, tunnel = json.loads(subprocess.check_output(['node', str(runner), str(game)], text=True))
        self.assertTrue(power['state']['frightened'])
        self.assertEqual(power['state']['score'], 60)  # Starting pellet plus power pellet.
        self.assertEqual(power['state']['pellets_remaining'], 242)
        self.assertEqual(tunnel['state']['player']['column'], 27)

    def test_rollout_and_evaluation_use_shared_engine(self):
        evaluation = evaluate(ROOT / 'data/pacman-arcade-evaluation.jsonl', predict=rule_predict)
        self.assertEqual(evaluation['accuracy'], 1)
        first = rollout(predict=rule_predict)
        self.assertEqual(first, rollout(predict=rule_predict))
        self.assertEqual(first['engine_revision'], PACMAN_REVISION)
        self.assertGreater(first['dots_collected'], 0)
        self.assertGreater(first['score'], 0)
        self.assertIn(first['outcome'], ('turn_limit', 'life_lost', 'level_cleared'))

    def test_bridge_traces_verified_adapter_and_rejects_changes_and_old_boards(self):
        state = initial_state()
        response = rule_predict(body(state))
        identity = {'checkpoint': '/checkpoints/kev-4b-skills', 'stage': 'skills', 'adapter_sha256': 'abc'}
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / 'trace.jsonl'
            bridge = NotebookBridge(trace, lambda: dict(identity))
            with patch('pacman_lab.call', return_value=(copy.deepcopy(response), 12)):
                answer = bridge.decide(state)
                self.assertEqual(answer['active_checkpoint'], identity)
                self.assertEqual(json.loads(trace.read_text())['active_checkpoint'], identity)
                bridge.model_info = Mock(side_effect=[identity, {**identity, 'checkpoint': 'changed'}])
                with self.assertRaisesRegex(RuntimeError, 'changed during'):
                    bridge.decide(state)
                self.assertEqual(len(trace.read_text().splitlines()), 1)
                with self.assertRaisesRegex(ValueError, 'current classic'):
                    bridge.decide({'game': 'old'})

    def test_original_assets_and_browser_adapter_controls_are_embedded(self):
        files = upstream_files()
        self.assertIn(b'GNU GENERAL PUBLIC LICENSE', files['LICENSE'])
        browser = notebook_game()
        self.assertIn('data:font/ttf;base64,', browser)
        self.assertIn('data:audio/mpeg;base64,', browser)
        self.assertIn('pacman.model', browser)
        self.assertIn('Checkpoint not connected', browser)
        self.assertNotIn('askJev()', browser)
        self.assertIn('active_checkpoint', browser)

    def test_browser_native_clock_adapter_transport_and_failure_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / 'engine.js'
            script.write_text(engine_script())
            result = subprocess.check_output(['node', str(ROOT / 'games/test-arcade-browser.cjs'),
                                               str(script), str(ROOT / 'games/arcade-browser.js')], text=True)
            self.assertIn('API-error pause passed', result)
            with ArcadeEngine() as engine:
                engine.reset()
                self.assertEqual(json.loads(result)['firstState'], engine.step('left')['state'])


if __name__ == '__main__':
    unittest.main()
