// CPU replay harness for the same engine embedded in the browser. GPL-3.0.
const fs = require('node:fs'), vm = require('node:vm'), readline = require('node:readline');
const script = fs.readFileSync(process.argv[2], 'utf8');
function Audio() { this.paused = true; this.load = this.pause = this.addEventListener = this.removeEventListener = function() {}; this.play = function() { return Promise.resolve(); }; }
const context = {console: {log() {}, error() {}}, Audio, localStorage: {},
    window: {addEventListener() {}, location: {hash: ''}}, document: {},
    setTimeout() {}, clearTimeout() {}, setInterval() {}, clearInterval() {},
    requestAnimationFrame() {}, cancelAnimationFrame() {}};
vm.createContext(context); vm.runInContext(script, context); context.labArcade.headless();
readline.createInterface({input: process.stdin}).on('line', line => {
    try {
        const input = JSON.parse(line), api = context.labArcade;
        let result;
        if (input.command === 'reset') result = api.reset({...input.options, skipReady: true});
        else if (input.command === 'observe') result = api.observe();
        else if (input.command === 'step') result = api.step(input.direction);
        else if (input.command === 'replay') result = api.replay(input.replay);
        else if (input.command === 'transition') { api.replay(input.replay); result = api.step(input.direction); }
        else throw new Error('Unknown engine command');
        process.stdout.write(JSON.stringify({result}) + '\n');
    } catch (error) { process.stdout.write(JSON.stringify({error: String(error.stack || error)}) + '\n'); }
});
