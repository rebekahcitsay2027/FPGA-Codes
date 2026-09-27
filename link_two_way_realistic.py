#!/usr/bin/env python3
"""
REALISTIC two-way link simulation:  NODE A <-> NODE B
=====================================================
Extends the ideal two-way sim with the impairments that actually limit a
44 km free-space microwave time-transfer link, and decomposes the error budget.

Key physics:
  Two-way cancels RECIPROCAL, SIMULTANEOUS effects -> they hit d, not Delta:
     * propagation delay (~147 us)          -> cancels
     * common-mode tropospheric delay        -> cancels (shows up in d_hat)
  NON-reciprocal effects are what limit Delta:
     * per-frequency hardware-delay DRIFT (f1 vs f2 chains)  <- usually dominant
     * multipath (ground bounce), differs per pol/freq
     * receiver noise (random)
     * tiny non-reciprocal troposphere residual

Impairments modeled:
  1. Stochastic rubidium-class clock (random-walk Delta_true, the SIGNAL)
  2. Tropospheric delay: slow common-mode OU process + small non-reciprocal part
  3. Thermal drift of hardware delays h_AB(t), h_BA(t) (different per channel)
  4. Multipath ground bounce with slowly wandering phase (per direction)
  5. Time-varying C/N0 (scintillation + occasional deep fades)

Outputs: error time series, tropo-wander-vs-clean-Delta, TDEV curve, budget table.
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng=np.random.default_rng(11)
OUT="/mnt/user-data/outputs"; os.makedirs(OUT,exist_ok=True)

# ---- params ----
FCHIP=100e6; SPS=8; FS=FCHIP*SPS; DEG=15; L=2**DEG-1; BETA=0.25; SPAN=10
FRF=5.8e9; N=L*SPS; TCODE=L/FCHIP; C=299_792_458.0; D_LINK=44e3
REPORT=[]; say=lambda s="":(print(s),REPORT.append(s))

# ---------------- LINK BUDGET from datasheet values ----------------
# PA  ZHL-1W-63-S+  : OP1dB +28 dBm (Psat +30)      LNA ZX60-83LN-S+ : NF 1.56 dB
# BPF ZVBP-5800-S+  : IL 0.8 dB, GD ~3.5 ns          Dish 0.9 m, eff 0.6
Pt_dBm=28.0; IL_BPF=0.8; NF_LNA=1.56; D_DISH=0.9; EFF=0.6; L_ATM=0.5; L_PT=2.0
lam=C/FRF
G_DISH=10*np.log10(EFF*(np.pi*D_DISH/lam)**2)
FSPL=92.45+20*np.log10(D_LINK/1e3)+20*np.log10(FRF/1e9)
Pr=Pt_dBm-IL_BPF+G_DISH-FSPL-L_ATM-L_PT+G_DISH-IL_BPF
NF_SYS=IL_BPF+NF_LNA                      # BPF ahead of LNA
N0=-174+NF_SYS
CN0=Pr-N0                                 # ~120 dB-Hz
GD_BPF=3.5e-9                             # ZVBP-5800-S+ passband group delay (per filter)

# ---- code/filter (from prior) ----
def lfsr(deg,taps):
    st=[1]*deg; out=np.empty(2**deg-1,np.int8)
    for i in range(out.size):
        out[i]=st[-1]; fb=0
        for t in taps: fb^=st[t-1]
        st=[fb]+st[:-1]
    return out
def rrc(beta,sps,span):
    Nt=span*sps; t=np.arange(-Nt/2,Nt/2+1)/sps; h=np.empty_like(t)
    for i,ti in enumerate(t):
        if abs(ti)<1e-12: h[i]=1-beta+4*beta/np.pi
        elif beta>0 and abs(abs(ti)-1/(4*beta))<1e-9:
            h[i]=(beta/np.sqrt(2))*((1+2/np.pi)*np.sin(np.pi/(4*beta))+(1-2/np.pi)*np.cos(np.pi/(4*beta)))
        else:
            num=np.sin(np.pi*ti*(1-beta))+4*beta*ti*np.cos(np.pi*ti*(1+beta))
            den=np.pi*ti*(1-(4*beta*ti)**2); h[i]=num/den
    return h/np.sqrt(np.sum(h**2))
H=rrc(BETA,SPS,SPAN)
def build(taps):
    b=1.0-2.0*lfsr(DEG,taps); up=np.zeros(N); up[::SPS]=b
    s=np.convolve(np.tile(up,2),H,mode="same")[:N]; s/=np.sqrt(np.mean(np.abs(s)**2))
    ref=np.convolve(np.tile(s,2),H[::-1],mode="same")[:N]; ref/=np.sqrt(np.mean(np.abs(ref)**2))
    return s.astype(complex),ref
S_AB,R_AB=build([15,14]); S_BA,R_BA=build([15,1])
Ps=np.mean(np.abs(S_AB)**2); RF_AB=np.fft.fft(R_AB); RF_BA=np.fft.fft(R_BA); _fb=np.fft.fftfreq(N)

def fdelay(x,tau): return np.fft.ifft(np.fft.fft(x)*np.exp(-1j*2*np.pi*_fb*tau))

def rx(s,tau_samp,ref_f,cn0,phi=0.0,mp_amp=0.0,mp_delay=0.0,mp_phase=0.0):
    y=fdelay(s,tau_samp)*np.exp(1j*phi)
    if mp_amp>0: y=y+mp_amp*np.exp(1j*mp_phase)*fdelay(s,tau_samp+mp_delay)  # ground bounce
    sig2=Ps*FS/10**(cn0/10)
    y=y+np.sqrt(sig2/2)*(rng.standard_normal(N)+1j*rng.standard_normal(N))
    R=np.fft.ifft(np.fft.fft(y)*np.conj(ref_f)); m=np.abs(R); k=int(np.argmax(m))
    a,b,c=m[(k-1)%N],m[k],m[(k+1)%N]; d=0.5*(a-c)/(a-2*b+c) if (a-2*b+c)!=0 else 0.0
    return ((k+d)%N)/FS

# ---- truth / impairment generators ----
d0=D_LINK/C
def ou(nn,rmsval,tau_c,dt=1.0):     # Ornstein-Uhlenbeck colored noise, given RMS + corr time
    a=np.exp(-dt/tau_c); x=np.zeros(nn); s=rmsval*np.sqrt(1-a*a)
    for i in range(1,nn): x[i]=a*x[i-1]+s*rng.standard_normal()
    return x

def make_truth(NE,Tep=1.0, on=dict(clock=1,tropo=1,hw=1,mp=1,cn0=1)):
    t=np.arange(NE)*Tep
    # 1. clock (signal): FS725 white-FM, sigma_y(1s)=2e-11 (-> 2e-12 @100s, matches datasheet)
    SY1=2e-11; YDRIFT=1e-11
    y=YDRIFT+(SY1/np.sqrt(Tep))*rng.standard_normal(NE)*on['clock']
    Delta=3e-9+np.cumsum(y*Tep)
    if not on['clock']: Delta=3e-9+0*t
    # 2. troposphere: common-mode slow fluctuation (RMS ~40 ps, corr ~60 s) + tiny non-recip
    trop_common=ou(NE,40e-12,60.0)*on['tropo']
    trop_nonrecip=ou(NE,1.5e-12,60.0)*on['tropo']    # ~2% residual (non-dispersive+duplex)
    # 3. hardware thermal drift: per-channel, differential is what matters (~5 ps/degC, diurnal)
    hAB=12.3e-9+(25e-12*np.sin(2*np.pi*t/1500)+ou(NE,8e-12,300.))*on['hw']
    hBA= 8.7e-9+(18e-12*np.sin(2*np.pi*t/1500+1.1)+ou(NE,8e-12,300.))*on['hw']
    # 4. multipath: ground bounce, ~ -30 dB (pointed 0.9 m dish), extra path ~9 m, bounded phase
    mp=dict(aAB=0.03*on['mp'], aBA=0.03*on['mp'], dAB=30e-9*FS, dBA=27e-9*FS,
            phAB=ou(NE,1.5,300.), phBA=ou(NE,1.5,300.))
    # 5. C/N0: scintillation + occasional fades
    cn0=CN0+ (ou(NE,1.2,40.)*on['cn0'])
    if on['cn0']:
        for _ in range(max(1,NE//60)):
            k=rng.integers(0,NE); cn0[k:k+3]-=rng.uniform(4,8)
    return dict(t=t,Delta=Delta,trop_c=trop_common,trop_nr=trop_nonrecip,
                hAB=hAB,hBA=hBA,mp=mp,cn0=cn0)

CAL_D=(12.3e-9-8.7e-9)/2; CAL_d=(12.3e-9+8.7e-9)/2   # one-time calibration at nominal T

def run(NE,Tep=1.0,on=dict(clock=1,tropo=1,hw=1,mp=1,cn0=1),tr=None):
    if tr is None: tr=make_truth(NE,Tep,on)
    Dhat=np.empty(NE); dhat=np.empty(NE)
    for k in range(NE):
        d_k=d0+tr['trop_c'][k]
        tauAB=((d_k+tr['trop_nr'][k]+tr['Delta'][k]+tr['hAB'][k])*FS)%N
        tauBA=((d_k-tr['trop_nr'][k]-tr['Delta'][k]+tr['hBA'][k])*FS)%N
        m=tr['mp']
        M_AB=rx(S_AB,tauAB,RF_AB,tr['cn0'][k],rng.uniform(0,2*np.pi),
                m['aAB'],m['dAB'],m['phAB'][k])
        M_BA=rx(S_BA,tauBA,RF_BA,tr['cn0'][k],rng.uniform(0,2*np.pi),
                m['aBA'],m['dBA'],m['phBA'][k])
        Dhat[k]=(M_AB-M_BA)/2-CAL_D; dhat[k]=(M_AB+M_BA)/2-CAL_d
    return tr,Dhat,dhat

# ---- TDEV ----
def tdev(x,tau0=1.0):
    Nn=len(x); ns=np.unique(np.round(np.logspace(0,np.log10(Nn//3),18)).astype(int)); ns=ns[ns>=1]
    taus=[]; td=[]
    for n in ns:
        M=Nn-3*n+1
        if M<1: continue
        acc=0.0
        for j in range(M):
            s=np.sum(x[j+2*n:j+3*n]-2*x[j+n:j+2*n]+x[j:j+n])
            acc+=s*s
        tv=acc/(6.0*n*n*M); taus.append(n*tau0); td.append(np.sqrt(tv))
    return np.array(taus),np.array(td)

# ======================================================================
say("="*70); say("REALISTIC TWO-WAY LINK  (error-budget mode, datasheet values)"); say("="*70)
say("LINK BUDGET (from datasheets):")
say(f"  PA out {Pt_dBm:.0f} dBm (ZHL-1W-63-S+) | LNA NF {NF_LNA} dB (ZX60-83LN-S+) | BPF IL {IL_BPF} dB")
say(f"  dish gain {G_DISH:.1f} dBi (0.9 m, eff {EFF}) x2 | FSPL {FSPL:.1f} dB @ 44 km")
say(f"  Rx power {Pr:.1f} dBm | N0 {N0:.1f} dBm/Hz | C/N0 = {CN0:.1f} dB-Hz")
say(f"  -> noise is negligible; error set by hardware/multipath/clock")
say(f"CLOCK: FS725 sigma_y(1s)=2e-11 -> 2e-12 @100s ; PN -130 dBc/Hz@10Hz")
say(f"NOTE: hardware-delay DRIFT (tempco) and multipath level are ASSUMPTIONS")
say(f"      (not on datasheets) - flagged below, need bench measurement.")
say("")
NE=150; TEP=40.0                               # 150 x 40 s = 6000 s (1.7 h) to expose thermal drift
tr,Dhat,dhat=run(NE,TEP)                       # all impairments ON
err=Dhat-tr['Delta']
say(f"epochs {NE} @ {TEP:.0f} s  ({NE*TEP/3600:.2f} h) | all impairments ON")
say(f"  total Delta error RMS      : {np.std(err)*1e12:6.1f} ps")
say(f"  Delta error mean (bias)    : {np.mean(err)*1e12:+6.1f} ps")
say(f"  d_hat wander (tropo+delay) : {np.std(dhat)*1e12:6.1f} ps rms  (this is weather, not clock error)")
say("")

# ---- budget: isolate each source ----
say("Error budget (RMS contribution to Delta):")
# noise only
_,Dn,_=run(NE,TEP,on=dict(clock=0,tropo=0,hw=0,mp=0,cn0=0)); en=Dn-3e-9
say(f"  receiver noise             : {np.std(en)*1e12:6.1f} ps   (C/N0 {CN0:.0f} dB-Hz -> negligible)")
# hardware drift (analytic: differential drift minus calibration)
hw_err=((tr['hAB']-tr['hBA'])/2 - CAL_D)
say(f"  hardware-delay drift [ASSUMED]: {np.std(hw_err)*1e12:5.1f} ps  bias {np.mean(hw_err)*1e12:+.1f} ps  <- DOMINANT, needs bench meas.")
# troposphere non-reciprocal residual (common part cancels)
say(f"  troposphere (non-recip)    : {np.std(tr['trop_nr'])*1e12:6.1f} ps   (common-mode cancels into d)")
# multipath only (vs clean)
_,Dmp,_=run(NE,TEP,on=dict(clock=0,tropo=0,hw=0,mp=1,cn0=0)); emp=Dmp-3e-9
mp_inc=np.sqrt(max(np.std(emp)**2-np.std(en)**2,0.0))   # remove the noise floor in quadrature
say(f"  multipath [ASSUMED -30 dB] : {mp_inc*1e12:6.1f} ps   (incremental; bias {np.mean(emp)*1e12:+.1f} ps)")
rss=np.sqrt(np.std(en)**2+np.std(hw_err)**2+np.std(tr['trop_nr'])**2+mp_inc**2)
say(f"  root-sum-square estimate   : {rss*1e12:6.1f} ps   (vs measured total {np.std(err)*1e12:.1f} ps)")
say("")

# ---- TDEV all-on vs noise-only ----
ta,tda=tdev(err,TEP); tn,tdn=tdev(en,TEP)
say("TDEV(tau):  averaging helps until the systematic floor")
for i in range(0,len(ta),max(1,len(ta)//6)):
    say(f"  tau={ta[i]:5.0f}s  all-on {tda[i]*1e12:6.1f} ps | noise-only {tdn[i]*1e12:6.1f} ps")
say("")

# ======================================================================
C0,C1,C2,C3,C4="#1f77b4","#d62728","#2ca02c","#9467bd","#ff7f0e"
# Fig 1: Delta error time series + dominant systematic
fig,ax=plt.subplots(figsize=(9,4.4))
ax.plot(tr['t'],err*1e12,color=C0,lw=1,label=f"total Delta error (RMS {np.std(err)*1e12:.0f} ps)")
ax.plot(tr['t'],hw_err*1e12,color=C1,lw=2,label="hardware-delay drift (slow systematic)")
ax.axhline(0,color="k",lw=.6)
ax.set(title="Two-way clock-offset error vs time (all impairments)",
       xlabel="time [s]",ylabel="Delta error [ps]"); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/real_fig1_error.png",dpi=130); plt.close(fig)

# Fig 2: d(t) wanders (weather) while Delta stays clean  -> cancellation benefit
fig,axL=plt.subplots(figsize=(9,4.4))
axL.plot(tr['t'],(dhat-d0)*1e12,color=C4,lw=1,label="d_hat residual (troposphere weather)")
axL.set_xlabel("time [s]"); axL.set_ylabel("one-way delay residual [ps]",color=C4)
axL.tick_params(axis='y',labelcolor=C4); axL.grid(alpha=.3)
axR=axL.twinx()
axR.plot(tr['t'],(Dhat-tr['Delta'])*1e12,color=C0,lw=1,label="Delta error (clock)")
axR.set_ylabel("Delta error [ps]",color=C0); axR.tick_params(axis='y',labelcolor=C0)
axL.set_title("Common-mode troposphere hits d, cancels out of Delta")
l1,la1=axL.get_legend_handles_labels(); l2,la2=axR.get_legend_handles_labels()
axL.legend(l1+l2,la1+la2,fontsize=8,loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/real_fig2_cancellation.png",dpi=130); plt.close(fig)

# Fig 3: TDEV
fig,ax=plt.subplots(figsize=(7.5,4.6))
ax.loglog(ta,tda*1e12,"o-",color=C0,label="all impairments")
ax.loglog(tn,tdn*1e12,"s--",color="#7f7f7f",label="noise only (~tau^-1/2)")
ax.axhline(np.std(hw_err)*1e12,color=C1,ls=":",lw=1.2,label="hardware systematic floor")
ax.set(title="Time deviation of two-way Delta",
       xlabel="averaging time tau [s]",ylabel="TDEV [ps]")
ax.grid(alpha=.3,which="both"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/real_fig3_tdev.png",dpi=130); plt.close(fig)

# Fig 4: C/N0 timeline (context)
fig,ax=plt.subplots(figsize=(9,2.8))
ax.plot(tr['t'],tr['cn0'],color=C2,lw=1); ax.axhline(CN0,color="k",ls=":",lw=.8)
ax.set(title="Link C/N0 over time (scintillation + fades)",xlabel="time [s]",ylabel="C/N0 [dB-Hz]")
ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/real_fig4_cn0.png",dpi=130); plt.close(fig)

with open(f"{OUT}/link_two_way_realistic_results.txt","w") as fp: fp.write("\n".join(REPORT))
say("Wrote: real_fig1_error.png, real_fig2_cancellation.png, real_fig3_tdev.png, real_fig4_cn0.png, results.txt")
print("\nDONE")
