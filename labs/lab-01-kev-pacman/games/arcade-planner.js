// Offline model-predictive teacher over the native engine. GPL-3.0.
// Beam pruning makes the default search approximate, not globally optimal.
// width=0 exhausts the finite action tree (use a short horizon for audits).
function createArcadePlanner(api) {
    const names = ['up', 'left', 'down', 'right'], offsets = [-28, -1, 28, 1];
    let routes;
    function floor(ch) { return ch === ' ' || ch === '.' || ch === 'o'; }
    function graph(tiles) {
        const vertices = tiles.map((ch, i) => floor(ch) ? i : -1).filter(i => i >= 0);
        const edges = {};
        for (const i of vertices) {
            edges[i] = offsets.map((delta, d) => d % 2 ? Math.floor(i/28)*28 + (i%28+delta+28)%28 : i+delta)
                .filter(j => j >= 0 && j < tiles.length && floor(tiles[j]));
        }
        routes = {};
        for (const origin of vertices) {
            const queue = [origin], distance = {[origin]: 0};
            for (let head = 0; head < queue.length; head++) {
                for (const next of edges[queue[head]]) if (distance[next] === undefined) {
                    distance[next] = distance[queue[head]] + 1; queue.push(next);
                }
            }
            routes[origin] = distance;
        }
    }
    function features(base, loops, status) {
        const f = api.searchFeatures(), at = f.row*28 + f.column, distance = routes[at] || {};
        let nearest = 100, separation = 16;
        for (let i = 0; i < f.tiles.length; i++) if (f.tiles[i] === '.' || f.tiles[i] === 'o')
            nearest = Math.min(nearest, distance[i] === undefined ? 100 : distance[i]);
        for (const g of f.ghosts) if (g.danger)
            separation = Math.min(separation, distance[g.row*28+g.column] === undefined ? 16 : distance[g.row*28+g.column]);
        const score = f.score-base.score, pellets = base.pellets-f.pellets;
        return {survival: status === 'life_lost' ? 0 : status === 'level_cleared' ? 2 : 1,
            value: score + 5*pellets - 4*loops - 6*(f.pellets ? nearest : 0) + 2*Math.min(separation, 8),
            score_gain: score, pellets_collected: pellets, nearest_pellet: nearest,
            danger_distance: separation, repeated_visits: loops, outcome: status};
    }
    function rank(node) {
        return [Math.min(...node.metrics.map(m=>m.survival)),
            node.metrics.reduce((sum,m)=>sum+m.value, 0)/node.metrics.length];
    }
    function compare(a,b) { const x=rank(a),y=rank(b); return y[0]-x[0] || y[1]-x[1]; }
    return function plan(options={}) {
        const horizon = options.horizon === undefined ? 20 : options.horizon;
        const width = options.width === undefined ? 8 : options.width;
        const seeds = options.scenario_seeds || [11117, 77717];
        if (!Number.isInteger(horizon) || horizon < 1 || horizon > 32 || !Number.isInteger(width) || width < 0 || width > 64
                || !seeds.length || seeds.length > 8) throw new Error('Invalid planner budget');
        const before = api.observe(), replay = api.replayRecord(), root = api.searchSave(-1);
        if (!routes) graph(api.searchFeatures().tiles);
        const base = api.searchFeatures(), roots = api.legalMoves();
        let slot = 0, expanded = 0, pruned = 0;
        function advance(parent, direction) {
            const snapshots = [], metrics = [], loops = [];
            for (let i=0; i<seeds.length; i++) {
                api.searchRestore(parent.snapshots[i]);
                let status = parent.metrics[i].outcome;
                if (!status) { status = api.step(direction, true); expanded++; }
                const after = api.searchFeatures(), tile = after.row+','+after.column;
                const cost = parent.loops[i] + (status ? 0 : Math.min(4, Math.max(0, (after.visits[tile] || 1)-1)));
                snapshots.push(api.searchSave(slot++)); loops.push(cost);
                metrics.push(features(base, cost, status));
            }
            return {snapshots, metrics, loops, sequence:parent.sequence.concat(direction)};
        }
        try {
            const snapshots = seeds.map(value=>{api.searchRestore(root); api.searchRandom(value); return api.searchSave(slot++);});
            const start = {snapshots, metrics:seeds.map(()=>features(base,0,null)), loops:seeds.map(()=>0), sequence:[]};
            const candidates = [];
            for (const direction of roots) {
                let beam = [advance(start,direction)];
                for (let depth=1; depth<horizon; depth++) {
                    const next = [];
                    for (const node of beam) {
                        if (node.metrics.some(m=>m.outcome)) { next.push(node); continue; }
                        let legal = names.slice();
                        for (const snapshot of node.snapshots) {
                            api.searchRestore(snapshot);
                            const available = api.legalMoves();
                            legal = legal.filter(move=>available.includes(move));
                        }
                        // Ghost-eating pauses can give different pixel/tile
                        // endpoints across random scenarios. Only common legal
                        // open-loop continuations are feasible in this search.
                        if (!legal.length) next.push(node);
                        else for (const move of legal) next.push(advance(node,move));
                    }
                    next.sort(compare);
                    pruned += width ? Math.max(0,next.length-width) : 0;
                    beam = width ? next.slice(0,width) : next;
                }
                beam.sort(compare);
                const best = beam[0], value = rank(best);
                candidates.push({action:direction, survival:value[0], value:value[1],
                    sequence:best.sequence, scenarios:best.metrics});
            }
            candidates.sort((a,b)=>b.survival-a.survival || b.value-a.value);
            return {choice:candidates[0].action, candidates, horizon, width, scenario_seeds:seeds,
                transitions:expanded, pruned_nodes:pruned, exhaustive:pruned===0,
                objective:'worst-scenario survival, then mean(score + 5*pellets - 4*revisits - 6*nearest-pellet-distance + 2*min(ghost-distance,8))',
                information:'native current engine state/timers; independent future RNG scenarios; no episode RNG or future actions'};
        } finally {
            api.searchRestore(root);
            if (JSON.stringify(api.observe()) !== JSON.stringify(before) || JSON.stringify(api.replayRecord()) !== JSON.stringify(replay))
                throw new Error('Planner mutated the live game');
        }
    };
}
