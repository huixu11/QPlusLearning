/* Mathematical illustrations. No trained-model data. */
(function (root) {
  const W0=[[.6,-.4,.2,.1],[-.2,.5,-.6,.3],[.3,.1,.4,-.3],[-.5,.2,.2,.5]];
  const A=[[.5,-.4,.2,.6],[-.3,.5,.4,-.2]];
  const B=[[.4,-.2],[-.3,.5],[.6,.1],[-.2,.4]];
  function softmax(z){const high=Math.max(...z),e=z.map(v=>Math.exp(v-high)),sum=e.reduce((a,b)=>a+b,0);return e.map(v=>v/sum);}
  function update(t){return B.map(row=>A[0].map((_,j)=>t*row.reduce((sum,b,k)=>sum+b*A[k][j],0)));}
  function effective(t){return update(t).map((row,i)=>row.map((v,j)=>W0[i][j]+v));}
  const math={softmax,update,effective,W0,A,B};
  if(typeof module!=='undefined'&&module.exports)module.exports=math;
  if(!root.document)return;
  const document=root.document,$=id=>document.getElementById(id),names=['Up','Right','Down'];
  names.forEach((name,i)=>{
    const label=document.createElement('label');label.innerHTML=`${name}: <output id="z${i}"></output><br><input id="input${i}" type="range" min="-4" max="4" step="0.1" value="${2-i}" aria-label="${name} logit">`;$('logits').append(label);
    $('probabilities').insertAdjacentHTML('beforeend',`<span>${name}</span><div class="bar-track"><div class="bar" id="bar${i}"></div></div><output id="p${i}"></output>`);
  });
  let scoreAnimation=null,loraAnimation=null,scoreTick=0;
  function renderScores(){
    const z=names.map((_,i)=>+$(`input${i}`).value),p=softmax(z);
    names.forEach((_,i)=>{$(`z${i}`).textContent=z[i].toFixed(1);$(`p${i}`).textContent=(p[i]*100).toFixed(1)+'%';$(`bar${i}`).style.width=(p[i]*100)+'%';});
    $('total').textContent=p.reduce((a,b)=>a+b,0).toFixed(6);$('choice').textContent=names[p.indexOf(Math.max(...p))];
  }
  function stopScores(){clearInterval(scoreAnimation);scoreAnimation=null;$('softmax-play').textContent='Animate scores';}
  names.forEach((_,i)=>$(`input${i}`).addEventListener('input',()=>{stopScores();renderScores();}));
  $('softmax-play').onclick=()=>{
    if(scoreAnimation){stopScores();return;}$('softmax-play').textContent='Pause';
    scoreAnimation=setInterval(()=>{scoreTick+=.05;names.forEach((_,i)=>$(`input${i}`).value=(2-i+Math.sin(scoreTick+i)*.9).toFixed(1));renderScores();},80);
  };
  $('softmax-reset').onclick=()=>{stopScores();names.forEach((_,i)=>$(`input${i}`).value=2-i);renderScores();};
  ['Fixed base W0','Low-rank update','Effective matrix'].forEach((name,i)=>$('matrices').insertAdjacentHTML('beforeend',`<div class="matrix"><h3>${name}</h3><table id="matrix${i}" aria-label="${name}"></table></div>`));
  function renderMatrices(){
    const t=+$('update').value;$('fraction').textContent=t.toFixed(2);
    [W0,update(t),effective(t)].forEach((m,i)=>{
      $(`matrix${i}`).innerHTML=m.map(row=>'<tr>'+row.map(v=>{
        const light=95-Math.abs(v)*28,color=v>=0?`hsl(174 35% ${light}%)`:`hsl(30 43% ${light}%)`;
        return `<td style="background:${color}">${v>=0?'+':''}${v.toFixed(2)}</td>`;
      }).join('')+'</tr>').join('');
    });
  }
  function stopLoRA(){clearInterval(loraAnimation);loraAnimation=null;$('lora-play').textContent='Play';}
  $('update').oninput=()=>{stopLoRA();renderMatrices();};
  $('lora-play').onclick=()=>{
    if(loraAnimation){stopLoRA();return;}$('lora-play').textContent='Pause';
    loraAnimation=setInterval(()=>{$('update').value=(+$('update').value+.01)%1;renderMatrices();},80);
  };
  $('lora-reset').onclick=()=>{stopLoRA();$('update').value=0;renderMatrices();};
  renderScores();renderMatrices();
})(typeof globalThis!=='undefined'?globalThis:this);
