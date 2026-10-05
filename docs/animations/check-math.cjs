const assert=require('node:assert/strict');
const {softmax,update,effective,W0}=require('./kev-math.js');
const p=softmax([2,1,0]);
assert.ok(Math.abs(p[0]-.6652409557748218)<1e-12);
assert.ok(Math.abs(p.reduce((a,b)=>a+b,0)-1)<1e-12);
assert.deepEqual(softmax([1000,1000]),[.5,.5]);
assert.ok(softmax([-1000,0])[1]>.999999);
assert.deepEqual(effective(0),W0);
const full=update(1),half=update(.5),combined=effective(1);
for(let i=0;i<4;i++)for(let j=0;j<4;j++){
 assert.equal(half[i][j],full[i][j]*.5);
 assert.equal(combined[i][j],W0[i][j]+full[i][j]);
}
console.log('Probability normalization, numerical stability and LoRA identity passed.');
