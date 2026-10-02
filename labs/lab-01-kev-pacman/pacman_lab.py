"""Kev controls Pac-Man; deterministic game code controls both ghosts."""
from collections import deque
import copy
import json
from pathlib import Path
import random
import re
import time

from api_client import call, distribution

PACMAN_REVISION = "8446fe74690cd61909bda91acfadbccb0f02b422"
POLICY = "Control Pac-Man to collect dots and avoid ghosts. First avoid being caught this turn, including the programmed ghost move on even-numbered turns. Among safe moves, minimize maze distance to a remaining dot without crossing a ghost's current tile. Then prefer greater distance from the nearest ghost. Break remaining ties in order: up, down, left, right."
DIRECTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


def maze_from_html(html):
    match = re.search(r"const MAZE = (\[.*?\]);", html, re.S)
    if not match:
        raise ValueError("Pinned game maze was not found")
    return json.loads(re.sub(r",\s*]", "]", match.group(1)))


def position(actor):
    return actor["row"], actor["column"]


def actor(pos, heading=None):
    return {"row": pos[0], "column": pos[1], "heading": heading}


def destination(pos, direction):
    dr, dc = DIRECTIONS[direction]
    return pos[0] + dr, pos[1] + dc


def legal(maze, pos):
    options = []
    for direction in DIRECTIONS:
        r, c = destination(pos, direction)
        if 0 <= r < len(maze) and 0 <= c < len(maze[0]) and maze[r][c] != "#":
            options.append(direction)
    return options


def distances(maze, target, blocked=()):
    found, queue = {target: 0}, deque([target])
    while queue:
        tile = queue.popleft()
        for direction in legal(maze, tile):
            nxt = destination(tile, direction)
            if nxt not in found and nxt not in blocked:
                found[nxt] = found[tile] + 1
                queue.append(nxt)
    return found


def ghost_step(maze, ghosts, target):
    """Community straight-ahead/Manhattan chase, with fixed ties (no RNG)."""
    result = []
    for ghost in ghosts:
        pos, heading = position(ghost), ghost.get("heading")
        options = legal(maze, pos)
        if heading in options:
            move = heading
        else:
            choices = [d for d in options if d != OPPOSITE.get(heading)] or options
            def manhattan(d):
                nxt = destination(pos, d)
                return abs(nxt[0] - target[0]) + abs(nxt[1] - target[1])
            move = min(choices, key=manhattan)
        result.append(actor(destination(pos, move), move))
    return result


def initial_state(maze):
    pac = next((r, c) for r, row in enumerate(maze) for c, ch in enumerate(row) if ch == "P")
    ghosts = [actor((r, c)) for r, row in enumerate(maze) for c, ch in enumerate(row) if ch == "G"]
    excluded = {pac, *(position(g) for g in ghosts)}
    return {"maze": maze, "player": actor(pac), "ghosts": ghosts, "dots": [list(tile) for tile in sorted(distances(maze, pac)) if tile not in excluded], "turn": 0, "ghost_move_every": 2, "objective": "Collect every dot without being caught. Ghosts are controlled by deterministic game code."}


def body(state):
    criteria = {d: f"Move {d} to row {destination(position(state['player']), d)[0]}, column {destination(position(state['player']), d)[1]}." for d in legal(state["maze"], position(state["player"]))}
    return {"state": state, "model": "kev-latest", "questions": {"move": {"type": "choice", "instructions": POLICY, "criteria": criteria}}}


def transition(state, direction):
    """One player turn, then a ghost turn every second player turn."""
    if direction not in legal(state["maze"], position(state["player"])):
        raise ValueError("Player action crosses a wall")
    nxt = copy.deepcopy(state)
    pac = destination(position(state["player"]), direction)
    nxt["player"] = actor(pac, direction)
    nxt["turn"] += 1
    nxt["dots"] = [dot for dot in nxt["dots"] if tuple(dot) != pac]
    caught = pac in [position(g) for g in nxt["ghosts"]]
    if not nxt["dots"]:
        return nxt, "win"  # community game checks the final dot first
    if not caught and nxt["turn"] % nxt["ghost_move_every"] == 0:
        nxt["ghosts"] = ghost_step(state["maze"], nxt["ghosts"], pac)
        caught = pac in [position(g) for g in nxt["ghosts"]]
    return nxt, "caught" if caught else None


def teacher(state):
    """A stated heuristic for supervised imitation, not optimal Pac-Man play."""
    current_ghosts = [position(g) for g in state["ghosts"]]
    def rank(direction):
        nxt, outcome = transition(state, direction)
        if outcome == "win":
            return (-1, 0, 0)
        pac = position(nxt["player"])
        routes = distances(state["maze"], pac, current_ghosts)
        dot_distance = min((routes.get(tuple(dot), 999) for dot in state["dots"]), default=0)
        separation = min((distances(state["maze"], pac).get(position(g), 999) for g in nxt["ghosts"]), default=999)
        return (int(outcome == "caught"), dot_distance, -separation)
    return min(legal(state["maze"], position(state["player"])), key=rank)


def make_data(html, directory, counts=(64, 16, 16), seed=7):
    """Synthetic positions on the community maze, not human recordings."""
    maze = maze_from_html(html)
    initial = initial_state(maze)
    tiles = list(distances(maze, position(initial["player"])))
    rng, used, rows = random.Random(seed), set(), []
    while len(rows) < sum(counts):
        pac, ghost1, ghost2 = rng.sample(tiles, 3)
        group = (pac, ghost1, ghost2)
        if group in used or len(legal(maze, pac)) < 2:
            continue
        used.add(group)
        state = copy.deepcopy(initial)
        state["player"] = actor(pac)
        state["ghosts"] = [actor(ghost1), actor(ghost2)]
        state["dots"] = [list(t) for t in sorted(tiles) if t not in group and rng.random() < 0.3]
        if not state["dots"]:
            continue
        state["turn"] = rng.randrange(2)
        request = body(state)
        request["questions"]["move"].update(label=teacher(state), src="pacman_teacher")
        request.pop("model")
        request["_meta"] = {"id": f"board-{len(rows):03d}", "group_id": f"positions-{group}", "source": "pacman_synthetic", "variant": "clean", "label_source": "deterministic safety/shortest-dot heuristic"}
        rows.append(request)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    offset, groups, labels = 0, {}, {}
    for name, count in zip(["train", "development", "evaluation"], counts):
        part = rows[offset:offset+count]
        offset += count
        (directory / f"pacman-{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in part), encoding="utf-8")
        groups[name] = [row["_meta"]["group_id"] for row in part]
        labels[name] = {d: sum(row["questions"]["move"]["label"] == d for row in part) for d in DIRECTIONS}
    manifest = {"counts": dict(zip(["train", "development", "evaluation"], counts)), "labels": labels, "seed": seed, "game_commit": PACMAN_REVISION, "label_policy": POLICY, "groups": groups, "interpretation": "Authored synthetic positions on one maze. Measures imitation of a teacher, not human-level play or generalization to new mazes."}
    (directory / "pacman-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def inference_request(record):
    return {"state": record["state"], "model": "kev-latest", "questions": {key: {field: q[field] for field in ["type", "instructions", "criteria"]} for key, q in record["questions"].items()}}


def evaluate(path, predict=None):
    if predict is None:
        predict = lambda request: call("/v1/systemone", request)[0]
    results = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        started = time.perf_counter()
        response = predict(inference_request(record))
        answer = response["answers"]["move"]
        distribution(answer, record["questions"]["move"]["criteria"])
        _, outcome = transition(record["state"], answer["choice"])
        results.append({"id": record["_meta"]["id"], "gold": record["questions"]["move"]["label"], "prediction": answer["choice"], "probabilities": answer["probabilities"], "caught_next_turn": outcome == "caught", "http_ms": (time.perf_counter() - started) * 1000})
    if not results:
        raise ValueError("No labelled decisions to evaluate")
    return {"n": len(results), "accuracy": sum(row["gold"] == row["prediction"] for row in results) / len(results), "caught_next_turn": sum(row["caught_next_turn"] for row in results), "rows": results}


def rollout(maze, predict=None, max_turns=24):
    if predict is None:
        predict = lambda request: call("/v1/systemone", request)[0]
    state = initial_state(maze)
    initial_dots, frames, outcome = len(state["dots"]), [copy.deepcopy(state)], None
    for _ in range(max_turns):
        request = body(state)
        answer = predict(request)["answers"]["move"]
        distribution(answer, request["questions"]["move"]["criteria"])
        state, outcome = transition(state, answer["choice"])
        frames.append(copy.deepcopy(state))
        if outcome:
            break
    return {"turns": state["turn"], "dots_collected": initial_dots - len(state["dots"]), "outcome": outcome or "turn_limit", "frames": frames}


class NotebookBridge:
    def __init__(self, trace_path):
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)

    def decide(self, state):
        request = body(state)
        response, elapsed = call("/v1/systemone", request)
        distribution(response["answers"]["move"], request["questions"]["move"]["criteria"])
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"request": request, "response": response, "http_ms": elapsed}, ensure_ascii=False) + "\n")
        return response


def notebook_game(source_html, controller_js, license_text):
    """Reuse the pinned community maze and renderer, replace its controller."""
    html = source_html
    html = re.sub(r'<meta name="description"[^>]*>', '<meta name="description" content="Train Kev to control Pac-Man. Collect dots and avoid deterministic ghosts in this community browser game adaptation." />', html, count=1)
    def replace(pattern, replacement):
        nonlocal html
        html, count = re.subn(pattern, lambda _: replacement, html, count=1, flags=re.S)
        if count != 1:
            raise ValueError("Game source no longer matches the pinned adapter")
    replace(r'<a class="live".*?</a>', '<p class="live">Pac-Man lab — train the player</p>')
    replace(r'<h1>.*?</h1>', '<h1>Kev plays Pac-Man</h1>')
    replace(r'<p class="lede">.*?</p>', '<p class="lede">Collect dots and avoid the ghosts. Try a turn yourself, then let Kev control Pac-Man. Ghosts follow fixed game code.</p>')
    replace(r'<details class="set".*?</details>', '<p><label>Player: <select id="lab-mode"><option value="human">Human — arrow keys / WASD</option><option value="kev">Kev — model decisions</option></select></label></p>')
    replace(r'<footer>.*?</footer>', '<footer>Adapted from <a href="https://github.com/codaaiteam/jev-pacman">codaaiteam/jev-pacman</a> (MIT). The model controls Pac-Man; ghosts use deterministic game code.</footer>')
    replace(r'const DEFAULT_BASE = .*?// ---------------------------------------------------------------------------\n// Game', '// ---------------------------------------------------------------------------\n// Game')
    replace(r'const JEV_COST_PER_MTOK = .*?;', 'const STEP_MS=600;')
    replace(r'let startAt,', 'let turn=0,gameGeneration=0,playerInfo=null,playerBusy=false;\nlet startAt,')
    replace(r'function resetState\(\)\{', 'function resetState(){\n  turn=0;gameGeneration++;playerInfo=null;playerBusy=false;')
    replace(r'const rows=\[0,1\].map.*?document.getElementById\("hunt-rows"\).innerHTML=rows;', 'document.getElementById("hunt-rows").textContent=playerInfo ? `Pac-Man → ${playerInfo.choice}; probability ${playerInfo.probability.toFixed(2)}` : "Waiting for a player decision";')
    replace(r'async function askJev\(\).*?function startLoop\(\)', controller_js + '\nfunction startLoop()')
    html = html.replace('function pause(){running=false;', 'function pause(){gameGeneration++;playerBusy=false;running=false;')
    html = html.replace('Math.random()*0.3', '0')
    html = html.replace("Jev's ghosts are hunting — one API call picks both directions", 'Pac-Man decisions — ghosts follow fixed code')
    html = html.replace(' · $${cost.toFixed(5)}', '')
    html = re.sub(r'<span>· <b>\$\$\{cost\.toFixed\(5\)\}</b></span>', '', html)
    html = html.replace('Jev calls', 'model calls').replace('Jev decisions', 'model decisions')
    html = html.replace("Run! Jev's ghosts are hunting you", 'Collect dots and avoid ghosts')
    html = html.replace('Jev caught you', 'The ghosts caught Pac-Man').replace('You outran Jev.', 'Maze cleared.').replace("outrun Jev's ghosts", 'collect dots and avoid ghosts')
    html = html.replace('<title>Jev Pac-Man — you drive, the AI ghosts hunt you</title>', '<title>Kev plays Pac-Man</title>')
    html += '\n<!--\n' + license_text + '\n-->\n'
    return html
