// QPlusLearning integration, 2026. GPL-3.0, following the upstream arcade engine.
// Inserted inside the unchanged engine's closure; all physics, timers, ghost AI,
// collisions, score, lives, pellets, fruit, sounds and drawing remain upstream.
var labArcade = (function () {
    var names = ['up', 'left', 'down', 'right'];
    var modes = ['outside', 'eaten', 'going_home', 'entering_home', 'pacing_home', 'leaving_home'];
    var seed = 7, actions = [], history = [], visits = {}, frames = 0;
    var randomState = 7;
    var nativeRandom = Math.random;
    function seededRandom() {
        randomState ^= randomState << 13; randomState ^= randomState >>> 17; randomState ^= randomState << 5;
        return (randomState >>> 0) / 4294967296;
    }
    function key() { return pacman.tile.y + ',' + ((pacman.tile.x + map.numCols) % map.numCols); }
    function remember() {
        var tile = key(); visits[tile] = (visits[tile] || 0) + 1;
        history.push({row: pacman.tile.y, column: (pacman.tile.x + map.numCols) % map.numCols, heading: names[pacman.dirEnum]});
        if (history.length > 12) history.shift();
    }
    function legalMoves() {
        return names.filter(function (name, index) {
            var direction = {}; setDirFromEnum(direction, index);
            return isNextTileFloor(pacman.tile, direction);
        });
    }
    function observe() {
        var maze = [];
        for (var y = 0; y < map.numRows; y++) {
            var row = '';
            for (var x = 0; x < map.numCols; x++) {
                var tile = map.getTile(x, y);
                row += map.isFloorTileChar(tile) ? tile : '#';
            }
            maze.push(row);
        }
        return {
            game: 'classic-pacman', engine_revision: '7407174c1d6a38be8cd230577489e39e0873145b',
            maze: maze, legend: {'#': 'wall/ghost house', '.': '10-point pellet', 'o': '50-point power pellet', ' ': 'empty corridor'},
            player: {row: pacman.tile.y, column: (pacman.tile.x + map.numCols) % map.numCols, heading: names[pacman.dirEnum]},
            ghosts: ghosts.map(function(g) { return {name: g.name, row: g.tile.y,
                column: (g.tile.x + map.numCols) % map.numCols, heading: names[g.dirEnum],
                mode: modes[g.mode], frightened: !!g.scared}; }),
            legal_moves: legalMoves(), level: level, lives: extraLives + (state === newGameState ? 0 : 1), score: getScore(),
            pellets_remaining: map.dotsLeft(), frightened: energizer.isActive(),
            ghost_phase: ghostCommander.getCommand() === GHOST_CMD_SCATTER ? 'scatter' : 'chase',
            fruit: fruit.isPresent() ? {name: fruit.getCurrentFruit().name, points: fruit.getPoints(),
                row: Math.floor(fruit.pixel.y / tileSize), column: Math.floor(fruit.pixel.x / tileSize)} : null,
            turn: actions.length, simulation_frames: frames, recent_positions: history.slice(), visits_to_current_tile: visits[key()] || 0,
            objective: 'Collect pellets, use power pellets to eat frightened ghosts, avoid dangerous ghosts and repeated loops.'
        };
    }
    function reset(options) {
        options = options || {}; seed = options.seed || 7; randomState = seed;
        Math.random = options.seeded === false ? nativeRandom : seededRandom;
        actions = []; history = []; visits = {}; frames = 0;
        gameMode = GAME_PACMAN; practiceMode = false; turboMode = false;
        newGameState.setStartLevel(options.level || 1);
        switchState(newGameState);
        if (options.skipReady) {
            for (var i = 0; state !== playState && i < 600; i++) state.update();
            if (state !== playState) throw new Error('The upstream engine did not enter play state.');
        }
        remember(); return observe();
    }
    function atCenter() { return pacman.distToMid.x === 0 && pacman.distToMid.y === 0; }
    function tick() { if (state === playState) frames++; state.update(); }
    function setMove(direction) {
        if (legalMoves().indexOf(direction) === -1) throw new Error('Illegal arcade player direction: ' + direction);
        pacman.ai = false; pacman.setInputDir(names.indexOf(direction));
        actions.push(direction);
    }
    function finishMove() { remember(); }
    function outcome() {
        if (state === deadState || state === readyRestartState || state === overState) return 'life_lost';
        if (state === finishState || state === readyNewState) return 'level_cleared';
        return null;
    }
    function step(direction) {
        if (state !== playState) throw new Error('Arcade is not in play state.');
        setMove(direction);
        var moved = false, start = {x: pacman.pixel.x, y: pacman.pixel.y}, count = 0;
        do {
            tick(); count++;
            moved = moved || pacman.pixel.x !== start.x || pacman.pixel.y !== start.y;
            if (outcome()) break;
        } while ((!moved || !atCenter()) && count < 180);
        if (!outcome() && !moved) throw new Error('The arcade player did not move.');
        finishMove();
        return {state: observe(), outcome: outcome(), action_frames: count, replay: {seed: seed, actions: actions.slice()}};
    }
    function replay(record) {
        reset({seed: record.seed, skipReady: true});
        for (var i = 0; i < record.actions.length; i++) {
            var result = step(record.actions[i]);
            if (result.outcome && i !== record.actions.length - 1) throw new Error('Replay continues beyond a terminal state.');
        }
        return observe();
    }
    return {reset: reset, observe: observe, legalMoves: legalMoves, atCenter: atCenter,
        tick: tick, setMove: setMove, finishMove: finishMove, outcome: outcome, step: step, replay: replay,
        draw: function() {
            renderer.beginFrame(); state.draw();
            if (hud.isValidState()) renderer.renderFunc(hud.draw);
            renderer.endFrame();
        }, playing: function() { return state === playState; },
        replayRecord: function() { return {seed: seed, actions: actions.slice()}; },
        headless: function() { renderer = new Proxy({}, {get: function() { return function() {}; }}); }
    };
})();
globalThis.labArcade = labArcade;
