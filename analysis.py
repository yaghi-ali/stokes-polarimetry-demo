import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def savefig(fig,out):
    fig.tight_layout()
    fig.savefig(out/'preview.png',dpi=160,bbox_inches='tight')
    plt.close(fig)

M=.5*np.array([[1,1,0,0],[1,-1,0,0],[1,0,1,0],[1,0,-1,0],[1,0,0,1],[1,0,0,-1]],float)
def reconstruct(intensities):
    y=np.asarray(intensities,float)
    if y.shape!=(6,) or not np.all(np.isfinite(y)) or np.any(y<0):raise ValueError('Require six finite nonnegative analyzer intensities.')
    s=np.linalg.lstsq(M,y,rcond=None)[0]
    if s[0]<=0:raise ValueError('Total intensity must be positive.')
    dop=float(np.linalg.norm(s[1:])/s[0])
    return s,dop
def example(out,seed):
    truth=np.array([1,.55,.3,.4]);y=M@truth+np.random.default_rng(seed).normal(0,.008,6);s,dop=reconstruct(y)
    fig,ax=plt.subplots(1,2,figsize=(10,4));x=np.arange(6)
    ax[0].bar(x-.18,y,.36,label='Noisy measurements');ax[0].bar(x+.18,M@s,.36,label='Reconstruction');ax[0].set_xticks(x,['H','V','+45','-45','R','L']);ax[0].set_ylabel('Intensity (a.u.)');ax[0].legend()
    ax[1].bar(np.arange(4)-.18,truth,.36,label='Truth');ax[1].bar(np.arange(4)+.18,s,.36,label='Estimate');ax[1].set_xticks(np.arange(4),['S0','S1','S2','S3']);ax[1].legend();ax[1].set_title('Stokes reconstruction')
    savefig(fig,out);np.savetxt(out/'channels.csv',np.column_stack([y,M@s]),delimiter=',',header='measured_au,reconstructed_au',comments='')
    return {'stokes':s.tolist(),'degree_of_polarization':dop,'physical_estimate':bool(dop<=1+1e-12),'stokes_rmse':float(np.sqrt(np.mean((s-truth)**2)))}

def run(out='results',seed=42):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    metrics=example(out,seed);metrics.update(seed=seed,data='synthetic')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2));return metrics
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results');p.add_argument('--seed',type=int,default=42);a=p.parse_args();print(json.dumps(run(a.out,a.seed),indent=2))
