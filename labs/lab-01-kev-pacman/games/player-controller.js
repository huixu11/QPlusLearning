// Course adaptation: the model controls Pac-Man; ghosts use fixed code.
function observation(){
  return {
    maze: MAZE,
    player: {row:pac.r,column:pac.c,heading:dir?dir.name:null},
    ghosts: ghosts.map(g=>({row:g.r,column:g.c,heading:g.dir?g.dir.name:null})),
    dots: [...dots].map(k=>k.split(',').map(Number)).sort((a,b)=>a[0]-b[0]||a[1]-b[1]),
    turn, ghost_move_every:2,
    objective:'Collect every dot without being caught. Ghosts are controlled by deterministic game code.'
  };
}
function stepGhosts(){
  ghosts=ghosts.map(g=>{
    const legal=legalDirs(g);
    let move=g.dir&&legal.some(d=>d.name===g.dir.name)?g.dir:null;
    if(!move){
      const nonReverse=legal.filter(d=>d.name!==OPP[g.dir&&g.dir.name]);
      move=greedyToward(g,nonReverse.length?nonReverse:legal,pac);
    }
    return {r:g.r+move.dr,c:g.c+move.dc,dir:move};
  });
}
function finish(kind){running=false;over=kind;stopLoop();render();}
async function tick(){
  if(!running||playerBusy)return;
  const generation=gameGeneration;
  let move=null;
  const mode=document.getElementById('lab-mode').value;
  if(mode==='kev'){
    playerBusy=true;
    const t=performance.now();
    try{
      const result=await google.colab.kernel.invokeFunction('pacman.decide',[observation()],{});
      if(generation!==gameGeneration)return;
      const data=result.data['application/json'];
      const ans=data&&data.answers&&data.answers.move;
      if(!ans||!legalDirs(pac).some(d=>d.name===ans.choice))throw new Error('The model did not return a legal player move.');
      move=DIR_BY_NAME[ans.choice];
      playerInfo={choice:ans.choice,probability:ans.probabilities[ans.choice]};
      latency=Math.round(performance.now()-t);decisions++;err=null;
    }catch(e){
      if(generation!==gameGeneration)return;
      playerBusy=false;running=false;stopLoop();err=String(e.message||e);render();return;
    }
    playerBusy=false;
  }else{
    if(desiredDir&&!isWall(pac.r+desiredDir.dr,pac.c+desiredDir.dc))move=desiredDir;
    else if(dir&&!isWall(pac.r+dir.dr,pac.c+dir.dc))move=dir;
    if(!move)return;
    playerInfo=null;
  }
  pac={r:pac.r+move.dr,c:pac.c+move.dc};dir=move;turn++;
  dots.delete(kkey(pac.r,pac.c));
  if(dots.size===0){finish('win');return;}
  if(ghosts.some(g=>g.r===pac.r&&g.c===pac.c)){finish('caught');return;}
  if(turn%2===0)stepGhosts();
  if(ghosts.some(g=>g.r===pac.r&&g.c===pac.c)){finish('caught');return;}
  render();
}
