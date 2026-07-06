#!/usr/bin/env python3
# Z2 PRS-vs-physical null-test -- LONGER / higher-res follow-up.
#
# Why: at N=512 (a_max=19) neither model has reached scaling (d ln A1/d ln eta ~ +0.3,
# not 0) and physical hits ~1 cell/wall right at the end, so its rising A1 is partly the
# under-resolution overshoot.  N=1024 with a bigger box (i) extends the resolved a-reach
# and (ii) ~doubles cells/wall at fixed a, cleaning the existing eta<18 range -- which
# separates "rising because pre-scaling" from "rising because under-resolved".
#
# STRETCHED sizing (user choice 2026-06-25): push a_max as far as N=1024 allows, tolerating
# ~0.7 cell/wall at the very end (flag that tail in analysis).
#   a = 1+eta (radiation, H0/mu=1).  phys cells/wall = sqrt(2) N /(mu (1+eta) L).
#   N=1024, lSide=64:  a_max = lSide/2 = 32 (2 horizons),  eta_max = 31,
#   cells/wall(a=32) = sqrt(2)*1024/(32*64) = 0.71,  PRS frozen = sqrt(2)*1024/64 = 22.6
#   (mu L = 64 >> 2pi -> PRS breaks).  dx = 0.0625, dt = 0.0125 (dt/dx = 0.20).
#
# ONLY 1 seed per model (2 runs): N=1024 is ~16x more expensive than N=512 (~1 hr/run).
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUTBASE = "/mt/user-batch/dpasari/scan_z2_N1024"
SEEDS = [1]                       # single seed per model

MU = 1.0
VEV = 1.0
N = 1024
LSIDE = 64.0
DT = 0.0125
TMAX = 31.0
H0 = 1.0

COMMON = f"""#Evolution
expansion = true
evolver = LF

#Lattice
N = {N}
dt = {DT}
lSide = {LSIDE}     # dx=0.0625; delta=sqrt(2)/mu=1.414 -> 22.6 cells/wall at form

#Times
t0 = 0
tOutputFreq  = 0.1
tOutputInfreq  = 5
tOutputRareFreq = 3
tMax = {TMAX}      # a = 1+eta; a_max=32: 2 horizons & phys ~0.71 cell/wall at end (stretched)

#Spectra options
PS_type = 1
PS_version = 1

#GWs
GWprojectorType = 2
withGWs = false

fixedBackground = true
omegaEoS = 0.3333
H0 = {H0}          # H0/mu = 1 -> a = 1+eta

#IC
kCutOff = 3
initial_amplitudes = 0
initial_momenta    = 0

# Z2 domain wall (single real scalar). v = vev, Lambda = (mu/vev)^2 = 1, kick = 1.
# Single real scalar => no phase => no winding => NO junctions (the null-test).
mu  = {MU}
vev = {VEV}
"""

PRS_BLOCK = """
# PRS package (fat wall + Hubble drag). Matched mu=1 to the physical twin.
prsWall        = true
prsDamping     = 1.0
frictionGamma  = 0.0
frictionStartA = 0.0
frictionEndA   = 0.0
prsStartA      = 0
"""

PHYS_BLOCK = """
# PHYSICAL (non-PRS). Same box/mu/IC as the PRS twin; only the EOM differs. A=scal/(2aH).
prsWall = false
"""

n = 0
for model in ("phys", "prs"):
    block = PHYS_BLOCK if model == "phys" else PRS_BLOCK
    for seed in SEEDS:
        outdir = f"{OUTBASE}/results_{model}_s{seed:02d}/"
        text = f"#Output\noutputfile = {outdir}\n\n" + COMMON + f"baseSeed = z2_s{seed:02d}\n" + block
        fn = os.path.join(HERE, f"DWZ2_z2N1024_{model}_s{seed:02d}.in")
        with open(fn, "w") as f:
            f.write(text)
        n += 1
        print("wrote", os.path.basename(fn))
print(f"\n{n} cards in {HERE}")
