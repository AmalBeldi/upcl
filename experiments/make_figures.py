#!/usr/bin/env python3
"""Regenerates the RQ2 and RQ6 figures from the frozen values in
outputs/full_evaluation_log.txt (values copied below; update both together)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
L=[0,.25,.5,.75,1]; up=[1.0,.9977,.9886,.9735,.9514]; de=[.9398,.9398,.9398,.9398,.9243]
fig,ax=plt.subplots(figsize=(5.4,3.3),dpi=200)
ax.plot(L,up,"o-",label="UPCL-Pre"); ax.plot(L,de,"s--",label="Deterministic")
ax.set_xlabel("State-uncertainty intensity $\\lambda$"); ax.set_ylabel("NDCG@5 (mean over 3 seeds)")
ax.set_xticks(L); ax.grid(alpha=.3); ax.legend(frameon=False); fig.tight_layout(); fig.savefig("rq2_robustness.png")
K=[2,4,6,8,10]; ex=[0.10,0.96,17.38,355.96,7332.95]; mc=[4.48,6.04,7.00,8.55,10.52]
fig,ax=plt.subplots(figsize=(5.4,3.4),dpi=200)
ax.semilogy(K,ex,"o-",label="Exact enumeration"); ax.semilogy(K,mc,"s--",label="Monte Carlo ($N=2000$, no enumeration)")
ax.set_xlabel("Number of state variables $K$ ($|\\Omega|=4^K$)"); ax.set_ylabel("Wall-clock time (ms, log)")
ax.set_xticks(K); ax.grid(alpha=.3,which="both"); ax.legend(frameon=False,fontsize=8); fig.tight_layout(); fig.savefig("rq6_scalability.png")
