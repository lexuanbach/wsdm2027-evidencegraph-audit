"""Analytical unit-variance information curves; no empirical model confidence."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib.pyplot as plt
import paperfig as P

def information(n, rho):
    return np.asarray(n, dtype=float) / (1 + (np.asarray(n, dtype=float) - 1) * rho)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'outputs/figures')
    parser.add_argument('--no-tex', action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    P.apply_camera_ready_style()
    if args.no_tex: plt.rcParams.update({'text.usetex': False, 'font.family': 'serif'})
    n=np.arange(1,21)
    fig,ax=plt.subplots(figsize=(3.34,1.95))
    for rho,label,color in [(0,'Independent',P.NAVY),(.5,r'Equicorrelation $\\rho=0.5$',P.OCHRE),(1,'Exact copies',P.PLUM)]:
        ax.plot(n,information(n,rho),label=label.replace('\\\\rho','\\rho'),color=color,lw=1.4)
    ax.set_xlabel('Number of observations');ax.set_ylabel(r'GLS information $J$');ax.set_xticks([1,5,10,15,20]);ax.set_ylim(0,21);ax.legend(frameon=False,fontsize=7,loc='upper left');fig.tight_layout(pad=.25)
    fig.savefig(args.output/'fig_information.pdf');plt.close(fig)
    data={'scope':'Analytical calculation under known unit-variance covariance, not empirical LLM results','n':n.tolist(),'curves':{str(r):information(n,r).tolist() for r in [0,.5,1]}}
    (args.output/'fig_information.json').write_text(json.dumps(data,indent=2)+'\n')
if __name__=='__main__':main()
