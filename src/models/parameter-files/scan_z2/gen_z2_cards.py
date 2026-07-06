#!/usr/bin/env python3
# Z2 PRS-vs-physical null-test cards: {physical, PRS} x 3 seeds.
#
# MATCHED mu = vev = 1.0 for BOTH models -- identical box, IC, couplings; the ONLY
# difference is prsWall (fat-wall + Hubble drag).  Removes the mu=0.07-vs-1.0 confound
# that contaminates the Z4 phys-vs-PRS comparison.
#
# Sizing (analytic kink width delta = sqrt(2)/mu = 1.414; verified vs Z4 PRS card):
#   phys cells/wall = sqrt(2) N /(mu (1+eta) L)   (thins ~1/a)
#   PRS  cells/wall = sqrt(2) N /(mu L)           (frozen)
#   horizons/side   = mu L /(1+eta)
# Balanced box (>=2 horizons AND phys >=1 cell/wall both hit at a_max):
#   (1+eta_max)^2 = sqrt(2) N /(2) = 362  ->  eta_max = 18, lSide = 38  (N=512).
#   dx = 0.0742, dt = 0.015 (dt/dx = 0.20).  PRS: mu L = 38 >> 2pi (breaks).
# Window: formation ~eta6(PRS)/~eta10(phys) -> resolved+sub-horizon to eta18.
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUTBASE = "/mt/user-batch/dpasari/scan_z2"
SEEDS = [1, 2, 3]

MU = 1.0
VEV = 1.0
N = 512
LSIDE = 38.0
DT = 0.015
TMAX = 18.0
H0 = 1.0

COMMON = f"""#Evolution
expansion = true
evolver = LF

#Lattice
N = {N}
dt = {DT}
lSide = {LSIDE}     # dx=0.0742; delta=sqrt(2)/mu=1.414 -> 19 cells/wall at form; balanced box

#Times
t0 = 0
tOutputFreq  = 0.1
tOutputInfreq  = 5
tOutputRareFreq = 3
tMax = {TMAX}      # a = 1+eta; a_max=19: ~2 horizons & phys ~1 cell/wall at end

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
        fn = os.path.join(HERE, f"DWZ2_z2_{model}_s{seed:02d}.in")
        with open(fn, "w") as f:
            f.write(text)
        n += 1
        print("wrote", os.path.basename(fn))
print(f"\n{n} cards in {HERE}")
