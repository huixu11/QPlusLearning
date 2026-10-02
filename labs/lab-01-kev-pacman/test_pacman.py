"""Verify the browser and dataset use the same player/ghost dynamics."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from pacman_lab import *

ROOT = Path(__file__).resolve().parent
SOURCE = (ROOT / 'vendor/jev-pacman/index.html').read_text()


class PlayerLabTests(unittest.TestCase):
    def test_split_and_inference_without_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = make_data(SOURCE, directory)
            groups = [set(v) for v in manifest['groups'].values()]
            self.assertFalse(any(a & b for i, a in enumerate(groups) for b in groups[i+1:]))
            for name in manifest['counts']:
                rows = [json.loads(line) for line in (Path(directory) / f'pacman-{name}.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows), manifest['counts'][name])
                for row in rows:
                    q = row['questions']['move']
                    self.assertIn(q['label'], legal(row['state']['maze'], position(row['state']['player'])))
                    request = inference_request(row)
                    self.assertNotIn('_meta', request)
                    self.assertEqual(set(request['questions']['move']), {'type','instructions','criteria'})

    def test_teacher_avoids_capture_and_transition_preserves_input(self):
        maze = ['#######', '#.....#', '#.....#', '#######']
        state = {'maze':maze, 'player':actor((1,2)), 'ghosts':[actor((1,3)),actor((2,5))], 'dots':[[1,3],[2,1]], 'turn':0, 'ghost_move_every':2}
        original = copy.deepcopy(state)
        self.assertNotEqual(teacher(state), 'right')
        _, outcome = transition(state, 'right')
        self.assertEqual(outcome, 'caught')
        self.assertEqual(state, original)
        with self.assertRaises(ValueError):
            transition(state, 'up')

    def test_browser_and_python_turns_agree(self):
        controller = (ROOT / 'games/player-controller.js').read_text()
        html = notebook_game(SOURCE, controller, (ROOT / 'vendor/jev-pacman/LICENSE').read_text())
        self.assertNotIn('fetch(', html)
        self.assertNotIn('askJev()', html)
        script = re.search(r'<script>(.*?)</script>', html, re.S).group(1)
        rows = [json.loads(line) for line in (ROOT / 'data/pacman-train.jsonl').read_text().splitlines()]
        fixtures = []
        for row in rows[:32]:
            state = row['state']
            for direction in legal(state['maze'], position(state['player'])):
                fixtures.append({'state':state, 'direction':direction})
            moved = copy.deepcopy(state)
            moved['ghosts'] = ghost_step(moved['maze'], moved['ghosts'], position(moved['player']))
            moved['turn'] = 1
            for direction in legal(moved['maze'], position(moved['player'])):
                fixtures.append({'state':moved, 'direction':direction})
        harness = '''
const vm=require('node:vm'),fs=require('node:fs');
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const elements={};
const element=id=>elements[id] ||= {style:{},value:'human',className:'',innerHTML:'',textContent:'',addEventListener(){},appendChild(){},focus(){}};
const ctx={console,performance:{now:()=>0},setInterval:()=>1,clearInterval:()=>{},document:{getElementById:element,createElement:()=>element('cell'+Math.random()),querySelectorAll:()=>[]},window:{addEventListener(){}},google:{colab:{kernel:{invokeFunction:async()=>{throw new Error('mock API failure')}}}}};
vm.createContext(ctx);
vm.runInContext(input.script,ctx);
vm.runInContext(`globalThis.simulate=async input=>{
 const s=input.state;pac={r:s.player.row,c:s.player.column};dir=null;desiredDir=DIR_BY_NAME[input.direction];
 ghosts=s.ghosts.map(g=>({r:g.row,c:g.column,dir:DIR_BY_NAME[g.heading]||null}));
 dots=new Set(s.dots.map(d=>kkey(...d)));turn=s.turn;over=null;running=true;playerBusy=false;
 document.getElementById('lab-mode').value='human';await tick();return {state:observation(),outcome:over};
};
globalThis.failure=async()=>{newGame();document.getElementById('lab-mode').value='kev';running=true;await tick();return {turn,running,err};};
globalThis.stale=async()=>{newGame();document.getElementById('lab-mode').value='kev';running=true;let resolve;google.colab.kernel.invokeFunction=()=>new Promise(r=>resolve=r);const pending=tick();newGame();const before=JSON.stringify(observation());resolve({data:{'application/json':{answers:{move:{choice:'left',probabilities:{left:1}}}}}});await pending;return before===JSON.stringify(observation());};`,ctx);
(async()=>{const output=[];for(const fixture of input.fixtures)output.push(await ctx.simulate(fixture));process.stdout.write(JSON.stringify({output,failure:await ctx.failure(),stale:await ctx.stale()}));})();
'''
        with tempfile.TemporaryDirectory() as directory:
            js = Path(directory) / 'parity.js'
            js.write_text(harness)
            result = json.loads(subprocess.check_output(['node',str(js)], input=json.dumps({'script':script,'fixtures':fixtures}), text=True))
        for fixture, actual in zip(fixtures,result['output']):
            expected, outcome = transition(fixture['state'], fixture['direction'])
            self.assertEqual(actual['state'], expected)
            self.assertEqual(actual['outcome'], outcome)
        self.assertEqual(result['failure']['turn'], 0)
        self.assertFalse(result['failure']['running'])
        self.assertIn('mock API failure', result['failure']['err'])
        self.assertTrue(result['stale'])


if __name__ == '__main__':
    unittest.main()
