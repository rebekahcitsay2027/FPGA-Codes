#!/usr/bin/env python3
"""
Node B steering loop:  two-way Delta -> Kalman filter -> disciplined PPS_B'
=========================================================================
Closes the loop from the block diagram:
  measure Delta (two-way) -> Kalman estimate [offset, frequency]
  -> predict to the emission instant -> steer local timebase
  -> generate PPS_B' aligned to Node A ; report the RESIDUAL after steering.

Measurement model (validated by the two-way sim at C/N0=120 dB-Hz):
  z_k = x_k + v_k ,  v_k ~ N(0, sigma_meas^2) ,  sigma_meas = 1.2 ps

Clock model (FS725 rubidium):
  sigma_y(1 s) = 2e-11  ->  time wander sigma_x(tau) = sigma_y*tau/sqrt(3)
  so sigma_x(1 s) ~ 11.5 ps  (this is how far B drifts between corrections)

KEY RESULT: the disciplined-output residual is limited NOT by the 1.2 ps
measurement noise but by how far the free-running Rb wanders during one loop
update interval. Faster updates -> smaller residual, down to the meas. floor.
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="/mnt/user-data/outputs"; os.makedirs(OUT,exist_ok=True)
REPORT=[]; say=lambda s="":(print(s),REPORT.append(s))

# ---- physical constants of the problem ----
SIGMA_MEAS = 1.2e-12                 # two-way Delta measurement noise (from validated sim)
SIGY_1S    = 2.0e-11                 # FS725 sigma_y at 1 s
DX_1S      = (SIGY_1S/np.sqrt(3))**2 # phase random-walk diffusion [s^2/s]: sigma_x(1s)^2 = 11.5 ps^2
DY         = (5e-13)**2              # random-walk FM diffusion [1/s] (long-term, small)
X0         = 3.0e-9                  # initial B-vs-A offset (acquisition test)
Y0         = 1.0e-11                 # initial fractional frequency offset

def simulate(Tep, T_total, sigma_meas=SIGMA_MEAS, dx=DX_1S, dy=DY,
             x0=X0, y0=Y0, bias_amp=0.0, bias_period=3000.0, bias_const=0.0, seed=0):
    """Run the closed loop. Returns time, true offset x, steered residual r,
       freq estimate yhat, true freq y."""
    rng=np.random.default_rng(seed)
    Nk=int(round(T_total/Tep))
    t=np.arange(Nk)*Tep
    # --- generate true clock state ---
    x=np.empty(Nk); y=np.empty(Nk); x[0]=x0; y[0]=y0
    for k in range(1,Nk):
        y[k]=y[k-1]+np.sqrt(dy*Tep)*rng.standard_normal()
        x[k]=x[k-1]+y[k-1]*Tep+np.sqrt(dx*Tep)*rng.standard_normal()
    # measurement bias = slow hardware-delay drift (systematic, leaks through loop)
    bias=bias_amp*np.sin(2*np.pi*t/bias_period)+bias_const   # constant = static miscalibration
    # --- Kalman filter (2-state clock) ---
    F=np.array([[1,Tep],[0,1.0]]); H=np.array([[1.0,0.0]])
    Q=np.array([[dx*Tep,0],[0,dy*Tep]]); R=np.array([[sigma_meas**2]])
    xe=np.array([0.0,0.0]); P=np.diag([ (10e-9)**2,(1e-9)**2 ])   # init: unknown offset/freq
    r=np.empty(Nk); yhat=np.empty(Nk)
    for k in range(Nk):
        # PREDICT (this prediction is what steers the PPS' emitted this epoch)
        xp=F@xe; P=F@P@F.T+Q
        r[k]=x[k]-xp[0]                      # residual of disciplined output vs A
        # MEASURE + UPDATE
        z=x[k]+bias[k]+sigma_meas*rng.standard_normal()
        S=H@P@H.T+R; K=(P@H.T)@np.linalg.inv(S)
        xe=xp+(K@np.array([z-(H@xp)[0]])); P=(np.eye(2)-K@H)@P
        yhat[k]=xe[1]
    return t,x,r,yhat,y

# ======================================================================
say("="*70); say("NODE B STEERING LOOP (Kalman -> disciplined PPS_B')"); say("="*70)
say(f"measurement noise sigma_meas = {SIGMA_MEAS*1e12:.1f} ps (two-way, C/N0=120 dB-Hz)")
say(f"FS725: sigma_y(1s)={SIGY_1S:.0e} -> wander sigma_x(1s)={np.sqrt(DX_1S)*1e12:.1f} ps")
say("")

# ---- main run: 1 Hz update (PPS cadence), 600 s ----
TEP=1.0; T=600.0
t,x,r,yhat,y=simulate(TEP,T,seed=1)
ss=slice(int(0.1*len(t)),None)                         # drop acquisition transient
say(f"(A) 1 Hz loop, {T:.0f} s:")
say(f"    free-running drift (no steer): {np.std(x)*1e9:.3f} ns rms, up to {np.max(np.abs(x))*1e9:.2f} ns")
say(f"    steered residual (PPS_B' vs A): {np.std(r[ss])*1e12:.1f} ps rms")
say(f"    -> limited by Rb wander per 1 s update (~{np.sqrt(DX_1S)*1e12:.0f} ps), NOT by 1.2 ps meas.")
# acquisition
acq=np.argmax(np.abs(r)<50e-12)
say(f"    acquisition: pulls 3 ns offset to <50 ps in {acq*TEP:.0f} s")
say(f"    frequency lock: yhat -> {np.mean(yhat[ss]):.3e} (true ~{np.mean(y[ss]):.3e})")
say("")

# ---- (B) sweep update interval: the design curve ----
say("(B) Residual vs loop update interval (100 s runs):")
Teps=np.array([0.003,0.01,0.03,0.1,0.3,1.0,3.0])
res=[]
for tp in Teps:
    _,_,rr,_,_=simulate(tp,100.0,seed=2)
    ss2=slice(int(0.2*len(rr)),None); res.append(np.std(rr[ss2]))
res=np.array(res)
wander=np.sqrt(DX_1S*Teps)                              # Rb wander per update interval
for tp,rr,ww in zip(Teps,res,wander):
    say(f"    Tep={tp:6.3f}s : residual {rr*1e12:6.1f} ps | Rb wander {ww*1e12:6.1f} ps")
say(f"    measurement-noise floor ~ {SIGMA_MEAS*1e12:.1f} ps")
say("")

# ---- (C) hardware systematic leaks through (shown at fast 100 Hz update) ----
TEPF=0.01; TF=100.0
_,_,rf,_,_  = simulate(TEPF,TF,seed=4)                          # fast, no bias
_,_,rfb,_,_ = simulate(TEPF,TF,bias_const=8e-12,seed=4)         # fast, +8 ps static miscal
ssf=slice(int(0.2*len(rf)),None)
rms0=lambda a: np.sqrt(np.mean(a**2))                            # RMS about zero = accuracy
say("(C) Hardware systematic leaks through (100 Hz update, so clock wander is tiny):")
say(f"    no bias        : residual {rms0(rf[ssf])*1e12:.1f} ps  (near measurement floor)")
say(f"    +8 ps hw miscal: residual {rms0(rfb[ssf])*1e12:.1f} ps  (mean {np.mean(rfb[ssf])*1e12:+.1f} ps = the bias)")
say(f"    -> loop removes random NOISE and tracks the clock, but a measurement BIAS")
say(f"       passes straight into the disciplined output. Calibration stability = the floor.")
say("")

# ======================================================================
C0,C1,C2,C3="#1f77b4","#d62728","#2ca02c","#9467bd"
# Fig 1: free-running vs steered
fig,axL=plt.subplots(figsize=(9,4.4))
axL.plot(t,x*1e9,color=C1,lw=1,label="free-running B vs A (no steering)")
axL.set_xlabel("time [s]"); axL.set_ylabel("offset [ns]",color=C1); axL.tick_params(axis='y',labelcolor=C1)
axL.grid(alpha=.3)
axR=axL.twinx()
axR.plot(t,r*1e12,color=C0,lw=1,label="steered residual PPS_B' vs A")
axR.set_ylabel("residual [ps]",color=C0); axR.tick_params(axis='y',labelcolor=C0); axR.set_ylim(-60,60)
axL.set_title("Disciplining: free-running clock drifts (ns); steered output holds (ps)")
l1,la1=axL.get_legend_handles_labels(); l2,la2=axR.get_legend_handles_labels()
axL.legend(l1+l2,la1+la2,fontsize=8,loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/steer_fig1_lock.png",dpi=130); plt.close(fig)

# Fig 2: residual histogram (steady state), no-bias vs bias
fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].plot(t[ss],r[ss]*1e12,color=C0,lw=.8)
ax[0].axhline(0,color='k',lw=.6)
ax[0].set(title=f"Steered residual (1 Hz), RMS {np.std(r[ss])*1e12:.1f} ps",
          xlabel="time [s]",ylabel="residual [ps]"); ax[0].grid(alpha=.3)
ax[1].hist(rf[ssf]*1e12,bins=40,color=C0,alpha=.8,label=f"no bias ({np.std(rf[ssf])*1e12:.1f} ps)")
ax[1].hist(rfb[ssf]*1e12,bins=40,color=C1,alpha=.5,label=f"+8 ps hw bias ({np.std(rfb[ssf])*1e12:.1f} ps)")
ax[1].set(title="Residual @100 Hz: noise floor vs hardware-bias leak",xlabel="residual [ps]",ylabel="count")
ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/steer_fig2_residual.png",dpi=130); plt.close(fig)

# Fig 3: design curve residual vs update interval
fig,ax=plt.subplots(figsize=(7.5,4.6))
ax.loglog(Teps,res*1e12,"o-",color=C0,label="simulated residual")
ax.loglog(Teps,wander*1e12,"--",color=C1,label=r"Rb wander $\propto\sqrt{T_{ep}}$")
ax.axhline(SIGMA_MEAS*1e12,color=C2,ls=":",lw=1.3,label="measurement-noise floor 1.2 ps")
ax.set(title="Steered residual vs loop update interval (the design curve)",
       xlabel="update interval T_ep [s]",ylabel="residual RMS [ps]")
ax.grid(alpha=.3,which="both"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/steer_fig3_designcurve.png",dpi=130); plt.close(fig)

# Fig 4: frequency lock
fig,ax=plt.subplots(figsize=(8,4))
ax.plot(t,y*1e11,color=C2,lw=2,label="true frac. freq y (x1e-11)")
ax.plot(t,yhat*1e11,color=C0,lw=1,label="Kalman estimate yhat")
ax.set(title="Frequency discipline: Kalman locks to the true rate",
       xlabel="time [s]",ylabel="fractional frequency [x1e-11]")
ax.grid(alpha=.3); ax.legend(fontsize=9); ax.set_ylim(-1,3)
fig.tight_layout(); fig.savefig(f"{OUT}/steer_fig4_freqlock.png",dpi=130); plt.close(fig)

with open(f"{OUT}/node_b_steering_results.txt","w") as fp: fp.write("\n".join(REPORT))
say("Wrote: steer_fig1_lock.png, steer_fig2_residual.png, steer_fig3_designcurve.png, steer_fig4_freqlock.png, results.txt")
print("\nDONE")
