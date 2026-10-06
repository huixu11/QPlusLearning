"""Typeset equations and animate mathematical data; all values are illustrative."""
from pathlib import Path
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

W = Path(__file__).resolve().parent
ASSETS = W / 'assets'
ASSETS.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':16, 'text.color':'#1a2433',
                     'axes.labelcolor':'#1a2433', 'axes.edgecolor':'#d7dee5'})

def formula(name, text, width=12, height=1.3, font=29):
    fig=plt.figure(figsize=(width,height),dpi=160,facecolor='white')
    fig.text(.5,.5,text,fontsize=font,ha='center',va='center')
    fig.savefig(ASSETS/name,facecolor='white'); plt.close(fig)

formula('pointer-softmax.png',r'$z_j=\frac{q^{\mathsf{T}}k_j}{\sqrt{d_p}}\qquad p_j=\frac{e^{z_j}}{\sum_{k\in\mathcal{A}(s)}e^{z_k}}$',height=1.5)
formula('teacher-objective.png',r'$J=\Delta S+5P-4R-6D_p+2\min(D_g,8)$',height=1.3,font=29)
formula('cross-entropy.png',r'$\mathcal{L}_{\rm CE}=-\frac{1}{B}\sum_{i=1}^{B}\log p(y_i\mid s_i)$',height=1.5)
formula('lora.png',r'$W_{\rm eff}=W_0+\frac{\alpha}{r}BA$',height=1.3,font=32)
formula('accuracy.png',r'$A=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\!\left[\hat a_i=y_i\right]\qquad \hat a_i=\arg\max_j p(j\mid s_i)$',height=1.5,font=27)

def frame(fig):
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB')

z=np.array([2.,1.,0.]); e=np.exp(z-z.max()); p=e/e.sum()
assert np.allclose(p,[.6652409558,.2447284711,.0900305732]) and np.isclose(p.sum(),1)
frames=[]
for t in np.linspace(0,1,48):
    fig,axes=plt.subplots(1,3,figsize=(12,3.0),dpi=110)
    fig.subplots_adjust(left=.055,right=.99,top=.8,bottom=.23,wspace=.42)
    values=[z,np.exp(z),p]
    titles=['1  Pointer logits','2  Exponentiate','3  Normalize']
    for stage,(ax,v,title) in enumerate(zip(axes,values,titles)):
        phase=np.clip((t*4-stage),0,1)
        shown=v
        ax.bar(['Up','Right','Down'],shown,color=['#008b8b','#8fbdbb','#c3d5d5'],width=.62,alpha=.35+.65*phase)
        ax.set_title(title,fontsize=17,pad=10,color='#008b8b' if phase>.5 else '#1a2433')
        ax.set_ylim(0,[2.5,8.5,1][stage])
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(axis='both',labelsize=13)
        for i,value in enumerate(shown):
            label=f'{value:.0%}' if stage==2 else f'{value:.2f}'
            ax.text(i,value+.04*[2.5,8.5,1][stage],label,ha='center',fontsize=14)
    fig.text(.5,.015,'Illustrative logits [2, 1, 0]; three legal moves. These are not trained-model measurements.',
             fontsize=12,ha='center',color='#5e6e80')
    frames.append(frame(fig)); plt.close(fig)
frames=[frames[-1].copy(),*frames]
frames[0].save(ASSETS/'softmax.gif',save_all=True,append_images=frames[1:],duration=[1000]+[110]*47+[2100],loop=0)

# A tiny rank-two update, deliberately separate from Kev's rank-16 model.
w0=np.array([[.6,-.4,.2,.1],[-.2,.5,-.6,.3],[.3,.1,.4,-.3],[-.5,.2,.2,.5]])
a=np.array([[.5,-.4,.2,.6],[-.3,.5,.4,-.2]])
b=np.array([[.4,-.2],[-.3,.5],[.6,.1],[-.2,.4]])
assert np.linalg.matrix_rank(b@a)<=2
frames=[]
for t in np.linspace(0,1,40):
    fig,axes=plt.subplots(1,3,figsize=(12,3.0),dpi=110)
    fig.subplots_adjust(left=.035,right=.99,top=.78,bottom=.14,wspace=.2)
    for ax,m,title in zip(axes,[w0,t*(b@a),w0+t*(b@a)],['Fixed base W₀','Learned low-rank update','Effective matrix']):
        ax.imshow(m,cmap='BrBG',vmin=-1,vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title,fontsize=17,pad=10)
        for i in range(4):
            for j in range(4): ax.text(j,i,f'{m[i,j]:+.2f}',ha='center',va='center',fontsize=12,color='#1a2433')
    fig.text(.5,.02,'Illustrative 4 × 4 matrices, rank 2, α/r = 1. Kev uses rank 16 and α = 32.',
             fontsize=12,ha='center',color='#5e6e80')
    frames.append(frame(fig)); plt.close(fig)
frames[0].save(ASSETS/'lora.gif',save_all=True,append_images=frames[1:],duration=[120]*39+[2000],loop=0)
if (W/'image16.gif').is_file():
    shutil.copy2(W/'image16.gif',ASSETS/'clm-overview.gif')
elif not (ASSETS/'clm-overview.gif').is_file():
    raise FileNotFoundError('Retain the byte-preserved original CLM GIF in assets/clm-overview.gif.')
media=[]
for path in sorted(ASSETS.iterdir()):
    with Image.open(path) as im:
        media.append({'file':path.name,'width':im.width,'height':im.height,'frames':getattr(im,'n_frames',1),
                      'loop':im.info.get('loop')})
(W/'asset-checks.json').write_text(json.dumps({'media':media,'softmax':p.tolist(),
 'lora_update_rank':int(np.linalg.matrix_rank(b@a)), 'disclosure':'All new animation numbers are illustrative, not model results.'},indent=2)+'\n')
print(json.dumps(media,indent=2))
