#!/usr/bin/env python3
"""
TWO-WAY link simulation:  NODE A  <->  NODE B
=============================================
Bidirectional PRN time transfer (TWSTFT-style), complex baseband.

  A -> B  on f1/pol1  ->  Node B measures  M_AB = d + Delta + h_AB
  B -> A  on f2/pol2  ->  Node A measures  M_BA = d - Delta + h_BA

Two-way solve:
  Delta_hat = (M_AB - M_BA)/2  = Delta + (h_AB - h_BA)/2     <- clock offset
  d_hat     = (M_AB + M_BA)/2  = d     + (h_AB + h_BA)/2     <- one-way delay
The reciprocal propagation delay d (~147 us) CANCELS out of Delta_hat.
Constant hardware delays are removed by a one-time calibration.

Demonstrates:
  (1) single-epoch two-way solve: recover Delta to ps while d = 147 us,
  (2) hardware calibration,
  (3) multi-epoch: a drifting clock -> Delta(t) tracked, d(t) stays flat,
      fit the ramp to recover fractional frequency offset y.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
OUT = "/mnt/user-data/outputs"
os.makedirs(OUT, exist_ok=True)

# ---------------- parameters (mirror the TX/link models) ----------------
FCHIP=100e6; SPS=8; FS=FCHIP*SPS
DEG=15; L=2**DEG-1; BETA=0.25; SPAN=10
FRF=5.8e9; N=L*SPS; TCODE=L/FCHIP
C=299_792_458.0; D_LINK=44e3

# ---------------- LINK BUDGET from datasheet values ----------------
# PA ZHL-1W-63-S+ OP1dB +28 dBm | LNA ZX60-83LN-S+ NF 1.56 dB | BPF ZVBP-5800-S+ IL 0.8 dB
# 0.9 m dish, aperture efficiency 0.6
Pt_dBm=28.0; IL_BPF=0.8; NF_LNA=1.56; D_DISH=0.9; EFF=0.6; L_ATM=0.5; L_PT=2.0
lam=C/FRF
G_DISH=10*np.log10(EFF*(np.pi*D_DISH/lam)**2)
FSPL=92.45+20*np.log10(D_LINK/1e3)+20*np.log10(FRF/1e9)
Pr=Pt_dBm-IL_BPF+G_DISH-FSPL-L_ATM-L_PT+G_DISH-IL_BPF
N0=-174+IL_BPF+NF_LNA                       # -171.6 dBm/Hz
CN0=Pr-N0                                   # ~120 dB-Hz

REPORT=[]
def say(s=""):
    print(s); REPORT.append(s)

# ---------------- code + filter helpers ----------------
def lfsr(deg,taps):
    st=[1]*deg; out=np.empty(2**deg-1,np.int8)
    for i in range(out.size):
        out[i]=st[-1]; fb=0
        for t in taps: fb^=st[t-1]
        st=[fb]+st[:-1]
    return out

def is_maximal(deg,taps):
    """Definitive: LFSR returns to seed exactly at period 2^deg-1, not before."""
    st=[1]*deg; seed=tuple(st)
    for k in range(1,2**deg):
        fb=0
        for t in taps: fb^=st[t-1]
        st=[fb]+st[:-1]
        if tuple(st)==seed:
            return k==(2**deg-1)
    return False

# pick two distinct maximal-length codes (one per direction)
cands=[[15,14],[15,1],[15,4],[15,7],[15,13],[15,14,13,11]]
good=[t for t in cands if is_maximal(DEG,t)]
TAPS_AB, TAPS_BA = good[0], good[1]

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
    bpsk=1.0-2.0*lfsr(DEG,taps)
    up=np.zeros(N); up[::SPS]=bpsk
    s=np.convolve(np.tile(up,2),H,mode="same")[:N]
    s=s/np.sqrt(np.mean(np.abs(s)**2))
    ref=np.convolve(np.tile(s,2),H[::-1],mode="same")[:N]
    ref=ref/np.sqrt(np.mean(np.abs(ref)**2))
    return s.astype(complex), ref

S_AB,R_AB = build(TAPS_AB)     # A transmits this; B correlates
S_BA,R_BA = build(TAPS_BA)     # B transmits this; A correlates
Ps=np.mean(np.abs(S_AB)**2)
RF_AB=np.fft.fft(R_AB); RF_BA=np.fft.fft(R_BA)
_fb=np.fft.fftfreq(N)

# cross-isolation between the two direction codes (should be low)
xiso=np.abs(np.fft.ifft(np.fft.fft(S_AB)*np.conj(RF_BA)))
iso_db=20*np.log10(np.abs(np.fft.ifft(np.fft.fft(S_AB)*np.conj(RF_AB))).max()/xiso.max())

# ---------------- channel + receiver ----------------
def frac_delay(x,tau):
    return np.fft.ifft(np.fft.fft(x)*np.exp(-1j*2*np.pi*_fb*tau))

def link(s,tau_samp,ref_f,cn0=CN0,phi=0.0):
    y=frac_delay(s,tau_samp)*np.exp(1j*phi)
    sig2=Ps*FS/10**(cn0/10)
    y=y+np.sqrt(sig2/2)*(rng.standard_normal(N)+1j*rng.standard_normal(N))
    R=np.fft.ifft(np.fft.fft(y)*np.conj(ref_f)); mag=np.abs(R)
    k=int(np.argmax(mag)); a,b,c=mag[(k-1)%N],mag[k],mag[(k+1)%N]
    d=0.5*(a-c)/(a-2*b+c) if (a-2*b+c)!=0 else 0.0
    return ((k+d)%N)/FS, mag        # arrival time [s], correlator magnitude

# ---------------- truth ----------------
d_true   = D_LINK/C            # reciprocal one-way delay ~146.77 us
Delta0   = 3.0e-9             # clock offset B wrt A [s]
y_frac   = 1.0e-11           # fractional frequency offset (Rb-class)
h_AB     = 12.3e-9           # hardware delay on A->B measurement
h_BA     = 8.7e-9            # hardware delay on B->A measurement
CAL_D    = (h_AB-h_BA)/2      # calibration constants (measured once)
CAL_d    = (h_AB+h_BA)/2

def two_way(Delta, cn0=CN0):
    M_AB,magAB = link(S_AB,((d_true+Delta+h_AB)*FS)%N, RF_AB, cn0, rng.uniform(0,2*np.pi))
    M_BA,magBA = link(S_BA,((d_true-Delta+h_BA)*FS)%N, RF_BA, cn0, rng.uniform(0,2*np.pi))
    Delta_hat=(M_AB-M_BA)/2
    d_hat    =(M_AB+M_BA)/2
    return M_AB,M_BA,Delta_hat,d_hat,magAB,magBA

say("="*70); say("TWO-WAY LINK SIMULATION (A <-> B)"); say("="*70)
say(f"codes: A->B taps {TAPS_AB}, B->A taps {TAPS_BA} (both maximal-length)")
say(f"code cross-isolation : {iso_db:.1f} dB (auto-peak vs cross-peak)")
say(f"link budget: Pr {Pr:.1f} dBm | N0 {N0:.1f} dBm/Hz | C/N0 = {CN0:.1f} dB-Hz")
say(f"  (PA +{Pt_dBm:.0f} dBm, dish {G_DISH:.1f} dBi x2, FSPL {FSPL:.1f} dB, LNA NF {NF_LNA} dB)")
say(f"one-way delay d = {d_true*1e6:.3f} us")
say("")

# ---------------- (1) single epoch ----------------
M_AB,M_BA,Dh,dh,magAB,magBA = two_way(Delta0)
Dh_cal=Dh-CAL_D; dh_cal=dh-CAL_d
say("(1) Single two-way epoch:")
say(f"    M_AB              : {M_AB*1e6:.6f} us   (= d + Delta + h_AB)")
say(f"    M_BA              : {M_BA*1e6:.6f} us   (= d - Delta + h_BA)")
say(f"    Delta_hat (raw)   : {Dh*1e9:+.4f} ns   [= Delta + (h_AB-h_BA)/2]")
say(f"    Delta_hat (calib) : {Dh_cal*1e9:+.4f} ns   (true Delta {Delta0*1e9:+.3f} ns)")
say(f"    -> Delta error    : {(Dh_cal-Delta0)*1e12:+.1f} ps  "
    f"(while d = {d_true*1e6:.1f} us cancels out!)")
say(f"    d_hat (calib)     : {dh_cal*1e6:.6f} us   (true d {d_true*1e6:.6f} us, "
    f"err {(dh_cal-d_true)*1e12:+.0f} ps)")
say("")

# ---------------- (3) multi-epoch drift + frequency ----------------
NE=120; Tep=1.0                       # epochs at 1 s spacing (PPS cadence)
t=np.arange(NE)*Tep
Dtrue=Delta0 + y_frac*t
Dcal=np.empty(NE); dcal=np.empty(NE)
for k in range(NE):
    _,_,Dh_k,dh_k,_,_=two_way(Dtrue[k])
    Dcal[k]=Dh_k-CAL_D; dcal[k]=dh_k-CAL_d
p=np.polyfit(t,Dcal,1); y_est=p[0]; D0_est=p[1]
# slope uncertainty
resid=Dcal-np.polyval(p,t); s_slope=np.sqrt(np.sum(resid**2)/(NE-2))/np.sqrt(np.sum((t-t.mean())**2))
say("(3) Multi-epoch drift (clock drifting at y = 1e-11):")
say(f"    epochs            : {NE} at {Tep:.0f} s spacing")
say(f"    two-way Delta jitter (per epoch) : {np.std(resid)*1e12:.1f} ps rms")
say(f"    d(t) spread       : {np.std(dcal)*1e12:.1f} ps rms  (stays flat - delay cancels)")
say(f"    recovered y       : {y_est:.3e} +/- {s_slope:.1e}   (injected {y_frac:.1e})")
say(f"    recovered Delta0  : {D0_est*1e9:.4f} ns  (injected {Delta0*1e9:.3f} ns)")
say("")

# ---------------- plots ----------------
C0,C1,C2,C3="#1f77b4","#d62728","#2ca02c","#9467bd"

# Fig 1: two correlator peaks (the two measurements)
fig,ax=plt.subplots(figsize=(8,4))
lag=(np.arange(N)/FS)*1e6
kA=int(np.argmax(magAB)); kB=int(np.argmax(magBA)); w=200
def seg(mag,k):
    idx=np.arange(k-w,k+w); return lag[idx%N]-lag[k], mag[idx%N]/mag.max()
xA,yA=seg(magAB,kA); xB,yB=seg(magBA,kB)
ax.plot(xA*1e3,yA,color=C0,label=f"M_AB  (A->B, taps {TAPS_AB})")
ax.plot(xB*1e3,yB,color=C1,label=f"M_BA  (B->A, taps {TAPS_BA})")
ax.set(title="The two one-way measurements (peaks aligned for shape compare)",
       xlabel="lag - peak [ns]", ylabel="normalized |correlation|")
ax.grid(alpha=.3); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/tw_fig1_peaks.png",dpi=130); plt.close(fig)

# Fig 2: multi-epoch Delta(t) tracked, d(t) flat  (money plot)
fig,axL=plt.subplots(figsize=(9,4.6))
axL.plot(t,Dcal*1e9,"o",color=C0,ms=4,label="two-way Delta_hat (calibrated)")
axL.plot(t,Dtrue*1e9,"-",color=C2,lw=2,label="true Delta(t) = 3 ns + 1e-11 * t")
axL.plot(t,np.polyval(p,t)*1e9,"--",color=C1,lw=1.3,
         label=f"fit -> y={y_est:.2e}")
axL.set_xlabel("time [s]"); axL.set_ylabel("clock offset Delta [ns]", color=C0)
axL.tick_params(axis="y",labelcolor=C0); axL.grid(alpha=.3)
axR=axL.twinx()
axR.plot(t,(dcal-d_true)*1e12,".",color=C3,alpha=.5,label="d_hat - d [ps]")
axR.set_ylabel("one-way delay residual [ps]", color=C3)
axR.tick_params(axis="y",labelcolor=C3); axR.set_ylim(-400,400)
axL.set_title("Two-way tracks the drifting clock; the 147 us delay cancels")
l1,lab1=axL.get_legend_handles_labels(); l2,lab2=axR.get_legend_handles_labels()
axL.legend(l1+l2,lab1+lab2,fontsize=8,loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/tw_fig2_tracking.png",dpi=130); plt.close(fig)

# Fig 3: calibration bar (raw vs calibrated Delta)
fig,ax=plt.subplots(figsize=(6.5,4))
vals=[Dh*1e9, Dh_cal*1e9, Delta0*1e9]
labels=["raw\n(Delta+cal bias)","calibrated","true"]
bars=ax.bar(labels,vals,color=[C1,C0,C2],alpha=.85)
ax.axhline(Delta0*1e9,color=C2,ls="--",lw=1)
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+0.05,f"{v:.3f}",ha="center",fontsize=9)
ax.set(title="Hardware calibration recovers true clock offset",ylabel="Delta [ns]")
ax.grid(alpha=.3,axis="y")
fig.tight_layout(); fig.savefig(f"{OUT}/tw_fig3_calibration.png",dpi=130); plt.close(fig)

with open(f"{OUT}/link_two_way_results.txt","w") as fp: fp.write("\n".join(REPORT))
say("Wrote: tw_fig1_peaks.png, tw_fig2_tracking.png, tw_fig3_calibration.png, link_two_way_results.txt")
print("\nDONE")
