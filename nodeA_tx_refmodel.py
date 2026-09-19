#!/usr/bin/env python3
"""
Node A TX reference model
=========================
Dual-polarized microwave two-way time transfer (TWSTFT-style), RFSoC 4x2.

Node A transmit chain modeled here:
    PRN (m-sequence) -> BPSK -> RRC pulse-shape  [FPGA fabric @ 800 MSPS]
        -> RFDC interpolation x12 -> NCO fine mixer -> DAC @ 9.6 GSa/s
        -> 2nd-Nyquist image at 5.8 GHz -> PA -> ZVBP-5800 -> dish (pol1/f1)

This is a NUMERICAL REFERENCE MODEL. It:
  (a) validates the baseband DSP (PRN autocorrelation, RRC, occupied BW, eye),
  (b) fixes the clocking / RFDC placement math for 5.8 GHz, and
  (c) exports the two artifacts you load into the FPGA:
        - rrc_taps.coe   (Xilinx FIR Compiler coefficient file)
        - prn_bram.coe   (BRAM init: 32767 x 1-bit m-sequence)
The 5.8 GHz carrier is handled analytically (NCO + Nyquist imaging), NOT
time-simulated at >11 GSa/s.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/mnt/user-data/outputs"
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------
# 1. SYSTEM PARAMETERS  (single source of truth -> mirror into HDL params)
# ----------------------------------------------------------------------
FCHIP  = 100e6          # chip rate  [chips/s]
SPS    = 8              # samples/chip in FPGA fabric (RRC FIR runs here)
FS_FAB = FCHIP * SPS    # fabric sample rate = 800 MHz
DEG    = 15             # LFSR degree
L      = 2**DEG - 1     # code period = 32767 chips
BETA   = 0.25           # RRC roll-off
SPAN   = 10             # RRC span in chips  -> Ntap = SPAN*SPS + 1

# RF / DAC placement plan
FRF    = 5.8e9          # desired RF center
FS_DAC = 9.6e9          # DAC sample rate (<= 9.85 GSa/s max on RFSoC 4x2)
INTERP = int(round(FS_DAC / FS_FAB))     # RFDC interpolation factor
NZ     = 2 if FRF > FS_DAC/2 else 1       # DAC Nyquist zone for 5.8 GHz
# In-band NCO whose upper Nyquist image lands on FRF:
NCO    = FS_DAC - FRF                     # = 3.8 GHz  (image at fs-NCO = 5.8)

# Band-pass filter (Mini-Circuits ZVBP-5800-S+)
BPF_LO, BPF_HI = 5725e6, 5875e6

# Link geometry
D_LINK   = 44e3
C        = 299_792_458.0
T_OW     = D_LINK / C           # one-way delay
T_RT     = 2 * T_OW             # round-trip (ambiguity requirement)

REPORT = []
def say(s=""):
    print(s)
    REPORT.append(s)

say("="*70)
say("NODE A TX REFERENCE MODEL")
say("="*70)
say(f"Chip rate            : {FCHIP/1e6:.1f} Mcps")
say(f"Fabric sps / rate    : {SPS}  ->  {FS_FAB/1e6:.1f} MSPS")
say(f"Code (m-seq) degree  : {DEG}  ->  period L = {L} chips")
say(f"Code period          : {L/FCHIP*1e6:.2f} us")
say(f"Round-trip (44 km)   : {T_RT*1e6:.2f} us   (one-way {T_OW*1e6:.2f} us)")
say(f"Ambiguity margin     : code period {'>' if L/FCHIP>T_RT else '<'} round trip "
    f"({(L/FCHIP)/T_RT:.2f}x)")
say(f"RRC roll-off / span  : beta={BETA}, span={SPAN} chips")
say(f"DAC sample rate      : {FS_DAC/1e9:.2f} GSa/s")
say(f"RFDC interpolation   : x{INTERP}   ({FS_FAB/1e6:.0f} MHz -> {FS_DAC/1e9:.2f} GHz)")
say(f"RF center            : {FRF/1e9:.3f} GHz  (Nyquist zone {NZ} of DAC)")
say(f"NCO frequency        : {NCO/1e9:.3f} GHz  (image at fs-NCO = {(FS_DAC-NCO)/1e9:.3f} GHz)")
say("")

# ----------------------------------------------------------------------
# 2. m-SEQUENCE PRN  (PRBS15: x^15 + x^14 + 1, ITU-T O.150)
# ----------------------------------------------------------------------
def mseq(deg, taps):
    """Fibonacci LFSR, all-ones seed. Returns {0,1} of length 2**deg-1."""
    state = [1]*deg
    out = np.empty(2**deg - 1, dtype=np.int8)
    for i in range(out.size):
        out[i] = state[-1]
        fb = 0
        for t in taps:
            fb ^= state[t-1]
        state = [fb] + state[:-1]
    return out

TAPS = [15, 14]                        # PRBS15 feedback taps
chips = mseq(DEG, TAPS)                # {0,1}
ones = int(chips.sum())
assert ones == 2**(DEG-1), f"not maximal-length (ones={ones}, expected {2**(DEG-1)})"
bpsk = 1.0 - 2.0*chips                 # 0->+1, 1->-1  (antipodal)

# periodic (circular) autocorrelation via FFT
sp = np.fft.fft(bpsk)
acf = np.fft.ifft(sp*np.conj(sp)).real
peak = acf[0]
side = np.max(np.abs(acf[1:]))
say("PRN validation (PRBS15, taps x^15+x^14+1):")
say(f"  balance (ones/zeros)     : {ones}/{L-ones}  (ideal {2**(DEG-1)}/{2**(DEG-1)-1})")
say(f"  autocorr peak            : {peak:.0f}")
say(f"  max off-peak sidelobe    : {side:.0f}   (ideal -1)")
say(f"  peak/sidelobe ratio      : {20*np.log10(peak/max(abs(side),1e-9)):.1f} dB")
say("")

# ----------------------------------------------------------------------
# 3. RRC FILTER + BPSK PULSE SHAPING
# ----------------------------------------------------------------------
def rrc(beta, sps, span):
    N = span*sps
    t = np.arange(-N/2, N/2 + 1) / sps          # time in chips
    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1 - beta + 4*beta/np.pi
        elif beta > 0 and abs(abs(ti) - 1/(4*beta)) < 1e-9:
            h[i] = (beta/np.sqrt(2))*((1+2/np.pi)*np.sin(np.pi/(4*beta))
                                      + (1-2/np.pi)*np.cos(np.pi/(4*beta)))
        else:
            num = np.sin(np.pi*ti*(1-beta)) + 4*beta*ti*np.cos(np.pi*ti*(1+beta))
            den = np.pi*ti*(1 - (4*beta*ti)**2)
            h[i] = num/den
    return h / np.sqrt(np.sum(h**2))            # unit energy

h = rrc(BETA, SPS, SPAN)
NTAP = h.size
say(f"RRC filter: {NTAP} taps @ {FS_FAB/1e6:.0f} MSPS (span {SPAN} chips, sps {SPS})")

# upsample one code period and pulse-shape
up = np.zeros(L*SPS)
up[::SPS] = bpsk
tx = np.convolve(up, h, mode="same")            # baseband real waveform

# occupied bandwidth check (RRC -> Rc*(1+beta))
occ = FCHIP*(1+BETA)
say(f"Occupied BW (RRC)        : {occ/1e6:.1f} MHz  (+/-{occ/2e6:.1f} MHz)")
say(f"ZVBP-5800 passband       : {(BPF_HI-BPF_LO)/1e6:.0f} MHz  ({BPF_LO/1e6:.0f}-{BPF_HI/1e6:.0f})")
say(f"  -> fits in BPF          : {'YES' if occ <= (BPF_HI-BPF_LO) else 'NO'}")
say("")

# ----------------------------------------------------------------------
# 4. RF / NYQUIST PLACEMENT + DAC sinc rolloff
# ----------------------------------------------------------------------
def sinc_env(f, fs):
    x = f/fs
    return np.abs(np.sinc(x))                    # np.sinc = sin(pi x)/(pi x)
sinc_at_rf = sinc_env(FRF, FS_DAC)
say("RF placement:")
say(f"  DAC sinc |H| at {FRF/1e9:.2f} GHz : {sinc_at_rf:.3f}  ({20*np.log10(sinc_at_rf):.1f} dB)")
say(f"  (recovered by PA gain; expected for {NZ}nd-Nyquist operation)")
say("")

# ----------------------------------------------------------------------
# 5. PLOTS
# ----------------------------------------------------------------------
C0, C1, C2 = "#1f77b4", "#d62728", "#2ca02c"

# 5a. RRC impulse + magnitude response
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(np.arange(NTAP)-NTAP//2, h, color=C0)
ax[0].set(title="RRC impulse response", xlabel="tap (samples @ 800 MSPS)", ylabel="amp")
ax[0].grid(alpha=.3)
H = np.fft.fftshift(np.fft.fft(h, 4096))
fr = np.fft.fftshift(np.fft.fftfreq(4096, 1/FS_FAB))/1e6
ax[1].plot(fr, 20*np.log10(np.abs(H)/np.max(np.abs(H))+1e-9), color=C0)
for e in (-occ/2e6, occ/2e6):
    ax[1].axvline(e, color=C2, ls="--", lw=1)
ax[1].set(title=f"RRC magnitude (beta={BETA})", xlabel="baseband freq [MHz]",
          ylabel="dB", xlim=(-200, 200), ylim=(-80, 5))
ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_rrc.png", dpi=130); plt.close(fig)

# 5b. baseband PSD of shaped waveform
Nfft = 1 << 15
W = np.hanning(Nfft)
seg = tx[:Nfft]*W
P = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(seg)))+1e-9)
P -= P.max()
fb = np.fft.fftshift(np.fft.fftfreq(Nfft, 1/FS_FAB))/1e6
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(fb, P, color=C0, lw=.8, label="RRC-shaped BPSK")
for e, lab in [(-occ/2e6, "RRC edge"), (occ/2e6, None)]:
    ax.axvline(e, color=C2, ls="--", lw=1, label=lab)
for e, lab in [(-(BPF_HI-BPF_LO)/2e6, "BPF edge (mapped)"), ((BPF_HI-BPF_LO)/2e6, None)]:
    ax.axvline(e, color=C1, ls=":", lw=1.2, label=lab)
ax.set(title="Baseband PSD (one polarization, before upconversion)",
       xlabel="baseband freq [MHz]", ylabel="dB", xlim=(-200, 200), ylim=(-70, 5))
ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_baseband_psd.png", dpi=130); plt.close(fig)

# 5c. periodic autocorrelation (zoom)
lags = np.arange(-40, 41)
acf_shift = np.concatenate([acf[-40:], acf[:41]])
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(lags, acf_shift, color=C0, marker="o", ms=3)
ax.axhline(-1, color=C1, ls="--", lw=1, label="ideal floor -1")
ax.set(title=f"PRN periodic autocorrelation (L={L}) - zoom",
       xlabel="lag [chips]", ylabel="correlation")
ax.grid(alpha=.3); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_autocorr.png", dpi=130); plt.close(fig)

# 5d. eye diagram
fig, ax = plt.subplots(figsize=(7, 4))
span_s = 2*SPS
start = SPAN*SPS
for k in range(start, start+400*SPS, SPS):
    seg = tx[k:k+span_s]
    if seg.size == span_s:
        ax.plot(np.arange(span_s)/SPS, seg, color=C0, alpha=.06, lw=.8)
ax.set(title="Eye diagram (RRC-shaped BPSK)", xlabel="time [chips]", ylabel="amp")
ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_eye.png", dpi=130); plt.close(fig)

# 5e. RF placement at 5.8 GHz with BPF mask + DAC sinc envelope
frf = np.linspace(5.5e9, 6.1e9, 4000)
# one-sided baseband envelope (positive freqs, monotonic) reflected around FRF
pos = fb >= 0
xp = fb[pos]*1e6                 # Hz, increasing
fp = P[pos]
order = np.argsort(xp)
xp, fp = xp[order], fp[order]
bb_env = np.interp(np.abs(frf-FRF), xp, fp, left=-90, right=-90)
dac = 20*np.log10(sinc_env(frf, FS_DAC)/sinc_env(FRF, FS_DAC))   # normalized to FRF
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(frf/1e9, bb_env, color=C0, lw=.9, label="signal @ 5.8 GHz (NZ2 image)")
ax.plot(frf/1e9, dac, color="#7f7f7f", ls="-.", lw=1, label="DAC sinc (norm.)")
ax.axvspan(BPF_LO/1e9, BPF_HI/1e9, color=C1, alpha=.12, label="ZVBP-5800 passband")
ax.axvline(FRF/1e9, color=C2, ls="--", lw=1)
ax.set(title="RF placement: 5.8 GHz (DAC 2nd Nyquist zone image)",
       xlabel="RF [GHz]", ylabel="dB", ylim=(-70, 8))
ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower center")
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_rf_placement.png", dpi=130); plt.close(fig)

# ----------------------------------------------------------------------
# 6. EXPORT FPGA ARTIFACTS
# ----------------------------------------------------------------------
# 6a. RRC taps -> Xilinx FIR Compiler .coe (16-bit signed, radix 10)
COEFW = 16
scale = (2**(COEFW-1) - 1) / np.max(np.abs(h))
q = np.round(h*scale).astype(int)
with open(f"{OUT}/rrc_taps.coe", "w") as f:
    f.write(f"; RRC FIR taps  beta={BETA} span={SPAN} sps={SPS} Ntap={NTAP}\n")
    f.write(f"; {COEFW}-bit signed, scale={scale:.4f}\n")
    f.write("radix=10;\ncoefdata=\n")
    f.write(",\n".join(str(v) for v in q))
    f.write(";\n")

# 6b. PRN -> BRAM init .coe (1-bit wide, depth L)
with open(f"{OUT}/prn_bram.coe", "w") as f:
    f.write(f"; PRBS15 m-sequence  x^15+x^14+1  depth={L}  width=1\n")
    f.write("memory_initialization_radix=2;\n")
    f.write("memory_initialization_vector=\n")
    f.write(",\n".join(str(int(b)) for b in chips))
    f.write(";\n")

# 6c. params summary
with open(f"{OUT}/nodeA_tx_params.txt", "w") as f:
    f.write("\n".join(REPORT))
    f.write("\n\nFiles: rrc_taps.coe, prn_bram.coe, fig1..fig5 png\n")

say("Artifacts written:")
say("  rrc_taps.coe        (FIR Compiler)")
say("  prn_bram.coe        (BRAM init, 32767x1)")
say("  nodeA_tx_params.txt")
say("  fig1..fig5 .png")
print("\nDONE")
