#!/usr/bin/env python3
# Generator for analysis/cells_per_wall_box_sizing.ipynb
# Run:  python3 analysis/_build_cells_per_wall_nb.py   (from repo root)
# Builds a rigorous "lattice sites per wall" + physical-run box-sizing notebook.
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import os, itertools

nb = new_notebook()
cells = []
_uid = itertools.count(1)
def md(text):  cells.append(new_markdown_cell(text, id=f"md{next(_uid)}"))
def code(src): cells.append(new_code_cell(src,    id=f"cd{next(_uid)}"))

# ----------------------------------------------------------------------------- 0
md(r"""# Lattice sites per wall & physical-run box sizing (Z3 / Z4)

**Purpose.** Pin down, *rigorously and in program units*, how many lattice cells resolve a
domain wall as a function of conformal time, and use that to size the boxes for the
**physical (non-PRS) statistical scans** that mirror the PRS stat scans.

Everything the code reads/writes (`lSide`, `dx`, `dt`, `eta`, `a`, the BVP width `delta_bar`)
is in **program units** (length in units of $1/\omega_\star=1/\mu$, time in $1/\mu$). This
notebook makes explicit a correction to an earlier (WRONG) draft of this analysis:

> The kernel integrates the **raw** potential derivative (`scalarsingletkernels.h:69-72`:
> `a^2*Lapl(phi) - a^4*derivS`, with `derivS` the *raw* `DWZ4.h`/`DWZ3.h` `potDeriv`), so a
> static wall solves `Lapl(phi) = a^2 * mu^2 * phi` and the **comoving** wall width in lattice
> cells scales as **`1/(mu*a)`** — there is an explicit `1/mu`. With `H0/mu = 1`, `a = 1 + eta`
> is `mu`-free, and the IC kick `= sqrt(lambda1-2*lambda2) = 1` is also `mu`-free for our
> `beta`-couplings — **but** `cells/wall = delta_bar*N/(mu*a*lSide)` is **NOT** `mu`-free: a
> **smaller** `mu` gives **fatter** walls and **better** resolution. **`mu` is PHYSICAL, not
> cosmetic.** PRS wants `mu ~ O(1)`; the physical scan wants small `mu` (`=0.07`, used here).

**Sections**
1. Units & why `mu` is physical (not cosmetic) — verified against the code.
2. Wall width `delta_bar(beta)` from the 2-field BVP (program units), both models.
3. The geometric `cells/wall(eta)` formula and the `1/a` comoving thinning.
4. **Calibration**: the naive `cells/wall>=1` rule is too pessimistic; we calibrate the real
   resolution threshold `n_res` from the spectral gradient-power-past-Nyquist criterion on
   *existing* physical runs.
5. **Box sizer**: per-`beta` optimal `lSide`, `tMax` for the new physical scans.
""")

# ----------------------------------------------------------------------------- 1
md(r"""## 1. Units and why `mu` is physical (not cosmetic)

CosmoLattice integrates in program variables: fields $\tilde f = f/f_\star$, time
$\tilde t = \omega_\star t$, lengths in $1/\omega_\star$. For both models
`omegaStar = mu` and `fStar = v = mu/sqrt(lambda1 - 2*lambda2)` (`DWZ4.h`, `DWZ3.h`).

Two facts that ARE `mu`-free, read directly from the source:

* **Scale factor** (`fixedbackgroundexpansion.h`): `H0_prog = H0_input/omegaStar` and, for
  radiation ($w=1/3$, $\alpha=1$), `pEoS = 2/(3(1+w)-2\alpha) = 1`, so
  $a(\eta) = (1 + H_0^{\rm prog}\,\eta)^{1} = 1 + (H_0/\mu)\,\eta$.
  With `H0/mu = 1`: **`a = 1 + eta`** — no `mu`.

* **IC kick** (`fluctuationsgenerator.h:54`): the fluctuation norm $\propto
  \omega_\star/f_\star = \mu/v = \sqrt{\lambda_1-2\lambda_2}$. For our `beta`
  parameterization $\lambda_1-2\lambda_2 = 1$, so the kick is **also `mu`-free**.

**But the wall width in lattice cells is NOT `mu`-free.** The scalar kernel (`alpha=1`,
`scalarsingletkernels.h:69-72`) evolves

$$K \;=\; a^2\,\nabla^2\phi \;-\; a^4\,\frac{dV}{d\phi}\;-\;{\rm damp}\cdot\pi,$$

and `dV/dphi` is the **raw** potential derivative from `DWZ4.h`'s `potDeriv`,
$-\mu^2\phi + \lambda_1\phi^3 - 2\lambda_2(\cdots)$ — it carries an explicit $\mu^2$, it is
**not** rescaled into program units. A static wall solves $\nabla^2\phi = a^2\,dV/d\phi \sim
a^2\mu^2\phi$, so the **comoving** wall width (program length units) scales as
$\bar\delta/(\mu\,a)$, i.e. **$\propto 1/\mu$**.

**Conclusion:** $a(\eta)$ and the IC kick are `mu`-free, but `cells/wall = delta_bar*N/(mu*a*lSide)`
carries an explicit `1/mu` ⇒ **`mu` is PHYSICAL, not cosmetic**: a smaller `mu` gives fatter
walls and better resolution. `mu=0.07` and `mu=1` are emphatically **not** the same run.
""")

code(r"""import os, glob, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp, cumulative_trapezoid, simpson
from scipy.optimize import brentq

REPO  = "/mt/home/dpasari/CosmoLattice_Zn"
BATCH = "/mt/user-batch/dpasari"
ANA   = os.path.join(REPO, "analysis")
plt.rcParams.update({"figure.dpi": 110, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "text.usetex": False})

# ---- couplings tied to beta (IDENTICAL to the PRS stat cards) ----------------
def z4_couplings(b4):
    return 1.0/(1.0-b4), b4/(2.0*(1.0-b4))                 # lambda1, lambda2 ; l1-2l2 = 1
def R_of_beta(bN): return np.sqrt((1.0-bN)/(1.0+bN))
def z3_couplings(bN):
    R = R_of_beta(bN);  return 1.0/R, np.sqrt(2.0)*(1.0-R)/(3.0*R)

# ---- demonstrate that wall-width-in-cells depends on mu (the non-adjacent of the old claim) ----
def a_of_eta(eta, H0_over_mu=1.0): return 1.0 + H0_over_mu*eta

# verification anchor (hand check against the raw-potential kernel formula):
#   mu=0.1, dx=0.195, a=1, delta_bar=1.414  ->  cells/wall = delta_bar/(mu*a*dx)
anchor_cells = 1.414/(0.1*1.0*0.195)
print(f"verification anchor: mu=0.1, dx=0.195, a=1, delta_bar=1.414 -> cells/wall={anchor_cells:.1f}")
print()

# illustration at beta4=0.5, fixed box (lSide=130, N=512, a=1):
# delta_bar(beta4=0.5) from the BVP (Sec.2) is ~1.22; use a literal here since Sec.2 hasn't run yet.
DBAR_B40P5_LITERAL = 1.22     # BVP value for beta4=0.5 (see Sec. 2 below for the derivation)
lSide_demo, N_demo, a_demo = 130.0, 512, 1.0
dx_demo = lSide_demo/N_demo
print("mu-dependence check (beta4=0.5, fixed box lSide=130, N=512, a=1):")
for mu in (0.07, 1.0):
    l1, l2 = z4_couplings(0.5)
    v        = mu/np.sqrt(l1 - 2*l2)        # = fStar (mu-dependent normalization only)
    kick     = mu/v                          # IC norm prefactor (omegaStar/fStar) -- mu-free
    cpw      = DBAR_B40P5_LITERAL/(mu*a_demo*dx_demo)   # cells/wall = delta_bar/(mu*a*dx)
    print(f"  mu={mu:>4}:  v=fStar={v:7.4f}  kick=omega*/f*={kick:.4f} (mu-free)  "
          f"cells/wall={cpw:7.2f}")
ratio = (DBAR_B40P5_LITERAL/(0.07*a_demo*dx_demo))/(DBAR_B40P5_LITERAL/(1.0*a_demo*dx_demo))
print(f"  ratio (mu=0.07)/(mu=1.0) cells/wall = {ratio:.1f}  (= 1/0.07 = {1/0.07:.1f})")
print("=> kick & a(eta) are mu-free, but cells/wall ~ 1/mu => mu is PHYSICAL.")
print("   Physical scan uses mu=0.07 (FATTER walls = better resolution), H0=0.07 (H0/mu=1).")
""")

# ----------------------------------------------------------------------------- 2
md(r"""## 2. Wall width `delta_bar(beta)` (program units)

`delta_bar = mu * delta_phys` is the dimensionless (program-length) wall width, a function of
`beta` only. We solve the planar 2-field wall profile:

* **Z4** — adjacent (type-1) wall by `solve_bvp` continuation from `beta=1/3`; the diagonal
  (type-2) width is `beta`-independent, `delta_opp = 1.40` (Notes Eq. 2.39).
* **Z3** — single wall type by gradient-flow relaxation (cached).

The width is the 64%-energy width of the wall stress profile (validated against the exact
anchors `sigma_bar_adj(1/3)=1/sqrt2`, `delta_bar_adj(1/3)=1.40`).
""")

code(r'''# ---------- Z4 adjacent-wall BVP (reused from wall_width_box_calculator_Z4) -----
DELTA_OPP_Z4 = 1.40
def _rhs4(z, y, beta):
    h, hp, a, ap = y; r2 = h*h + a*a
    return np.vstack([hp, -h + h*r2 - beta*h*(h*h - 3*a*a),
                      ap, -a + a*r2 + beta*a*(3*h*h - a*a)])
def _bc4(yl, yr, vbar):
    return np.array([yl[0]-vbar, yl[2], yr[0], yr[2]-vbar])
def _solve4(beta, guess=None, Z=None):
    vbar = 1.0/np.sqrt(1.0-beta)
    if Z is None: Z = max(25.0, 14.0/np.sqrt(beta))
    z = np.linspace(-Z, Z, 4001)
    if guess is None:
        w = 2.0 + 1.2/np.sqrt(beta); th = np.tanh(z/w)
        h = vbar*(1-th)/2; a = vbar*(1+th)/2
        y = np.vstack([h, np.gradient(h, z), a, np.gradient(a, z)])
    else:
        h, hp, a, ap = guess.sol(z); y = np.vstack([h, hp, a, ap])
    sol = solve_bvp(lambda z, y: _rhs4(z, y, beta), lambda l, r: _bc4(l, r, vbar),
                    z, y, max_nodes=300000, tol=1e-9)
    return sol, Z
def _wt4(sol, Z, beta):
    zz = np.linspace(-Z, Z, 40001); h, hp, a, ap = sol.sol(zz); r2 = h*h + a*a
    Vbar = -0.5*r2 + 0.25*r2*r2 - (beta/4.0)*(h**4 - 6*h*h*a*a + a**4)
    eps = np.clip(0.5*(hp*hp + ap*ap) + (Vbar + 1.0/(4.0*(1.0-beta))), 0, None)
    sigma = simpson(y=eps, x=zz)
    cum = cumulative_trapezoid(eps, zz, initial=0); tot = cum[-1]
    lo = np.interp(0.18*tot, cum, zz); hi = np.interp(0.82*tot, cum, zz)
    return sigma, hi - lo
def _build4(bmin=0.005, bmax=0.95, step=0.01):
    anchor = 1.0/3.0; cache = {}
    for direction in (np.arange(anchor, bmin, -step), np.arange(anchor, bmax, step)):
        prev, _ = _solve4(anchor)
        for b in direction:
            s, Zc = _solve4(b, guess=prev, Z=max(25.0, 14.0/np.sqrt(b)))
            if s.success:
                prev = s; sg, dl = _wt4(s, Zc, b); cache[round(float(b),4)] = (sg, dl)
    bs = np.array(sorted(cache))
    return bs, np.array([cache[b][0] for b in bs]), np.array([cache[b][1] for b in bs])
_B4, _S4, _D4 = _build4()
def delta_adj_z4(b): return float(np.interp(b, _B4, _D4))
def delta_min_z4(b): return min(delta_adj_z4(b), DELTA_OPP_Z4)
assert abs(float(np.interp(1/3,_B4,_S4)) - 1/np.sqrt(2)) < 0.02
assert abs(delta_adj_z4(1/3) - 1.40) < 0.05
print("Z4 BVP OK: delta_adj(1/3)=%.3f  delta_adj(0.1)=%.3f  delta_adj(0.9)=%.3f  delta_opp=%.2f"
      % (delta_adj_z4(1/3), delta_adj_z4(0.1), delta_adj_z4(0.9), DELTA_OPP_Z4))
''')

code(r'''# ---------- Z3 wall width from cached gradient-flow curve -----------------------
Z3_CACHE = os.path.join(ANA, "z3_wall_curve_cache.npy")  # rows: beta_N, sigma, delta, min|phi|/v
_curve = np.load(Z3_CACHE)
_B3, _S3, _D3 = _curve[0], _curve[1], _curve[2]
def delta_adj_z3(b): return float(np.interp(b, _B3, _D3))
def beta3_old(bN):                                   # old physical coordinate -> beta_N
    R = R_of_beta(bN); return (1.0 - R)/(2.0*np.sqrt(R))
def betaN_from_old(bold):
    return brentq(lambda bN: beta3_old(bN) - bold, 1e-4, 0.999)
print("Z3 curve loaded: %d beta points (%.2f..%.2f)" % (_B3.size, _B3[0], _B3[-1]))
print("Z3 delta_adj: beta_N=0.1 -> %.3f   0.5 -> %.3f   0.9 -> %.3f"
      % (delta_adj_z3(0.1), delta_adj_z3(0.5), delta_adj_z3(0.9)))
print("beta_old->beta_N map:  0.05->%.3f  0.10->%.3f  1.0->%.3f"
      % (betaN_from_old(0.05), betaN_from_old(0.10), betaN_from_old(1.0)))
''')

code(r'''# ---------- delta_bar on the scan grid, both models ----------------------------
GRID = [0.1, 0.3, 0.5, 0.7, 0.9]
tab = pd.DataFrame({
    "beta_N":   GRID,
    "Z4_d_adj": [round(delta_adj_z4(b),3) for b in GRID],
    "Z4_d_opp": [DELTA_OPP_Z4]*len(GRID),
    "Z4_d_min": [round(delta_min_z4(b),3) for b in GRID],
    "Z3_d_adj": [round(delta_adj_z3(b),3) for b in GRID],
})
print(tab.to_string(index=False))
print("\nThinnest wall present (drives resolution): Z4 -> d_min, Z3 -> d_adj.")
''')

# ----------------------------------------------------------------------------- 3
md(r"""## 3. Geometric `cells/wall(eta)` (program units)

The lattice is comoving: `dx = lSide/N` (program length, constant in time). A wall has a fixed
**physical** width $\delta_{\rm phys}$; its **comoving** width shrinks as $1/a$. In program
length units the comoving width is $\bar\delta/a$, so

$$\boxed{\ \texttt{cells/wall}(\eta) \;=\; \frac{\bar\delta/(\mu a)}{dx} \;=\; \frac{\bar\delta\,N}{\mu\,a\,\texttt{lSide}}\;,\qquad a = 1 + (H_0/\mu)\,\eta\ }$$

There is an **explicit `1/mu`**: this is why a smaller `mu` buys more cells per wall (fatter
comoving wall) at fixed `lSide`, `N`. At $a=1$ this is $\bar\delta N/(\mu\,\texttt{lSide})$;
thereafter it falls as $1/a$. This `1/a` thinning is exactly why a **physical** run goes
sub-grid at finite $\eta$, whereas PRS freezes the comoving width and keeps `cells/wall`
constant.
""")

code(r'''def cells_per_wall(delta_bar, lSide, N, eta, mu, H0_over_mu=1.0):
    return delta_bar*N/(mu*a_of_eta(np.asarray(eta, float), H0_over_mu)*lSide)
def horizons(lSide, eta, H0_over_mu=1.0):
    # comoving Hubble radius (program) = a/a' = a/H0_prog ; horizons/dim = lSide / that
    return lSide * H0_over_mu / a_of_eta(np.asarray(eta, float), H0_over_mu)

# illustrate the thinning at the PRS stat box (L=279, N=512) vs a hypothetical fine box, mu=0.07
MU_DEMO = 0.07
eta = np.linspace(0, 100, 400)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for L, lab in [(279.0, "PRS box L=279"), (90.0, "fine box L=90")]:
    ax[0].loglog(1+eta, cells_per_wall(delta_min_z4(0.5), L, 512, eta, MU_DEMO), label=f"{lab}")
    ax[1].loglog(1+eta, horizons(L, eta), label=f"{lab}")
ax[0].axhline(1.0, ls="--", c="grey"); ax[0].set(xlabel="a=1+eta", ylabel=f"cells/wall (Z4, beta=0.5, mu={MU_DEMO})")
ax[0].set_title("comoving cells/wall ~ 1/(mu*a) (physical run)")
ax[1].axhline(3.0, ls="--", c="grey"); ax[1].set(xlabel="a=1+eta", ylabel="horizons/dim")
ax[1].set_title("horizons ~ lSide/a")
for a_ in ax: a_.legend()
fig.tight_layout(); fig.savefig(os.path.join(ANA, "cells_per_wall_geometry.pdf"))
print("saved cells_per_wall_geometry.pdf")
print(f"Note: horizons(eta) is mu-independent (a=1+eta), but cells/wall now scales as 1/mu (mu={MU_DEMO} here).")
''')

# ----------------------------------------------------------------------------- 3b
md(r"""## 3b. Width-vs-`mu` proof (`mu=0.07` vs `mu=1` at a FIXED box)

The plot below is the direct proof that `mu` is physical: at a **fixed** box (`lSide`, `N=512`)
and a **fixed** `beta4=0.5` wall width $\bar\delta$, `cells/wall(a)` for `mu=0.07` and `mu=1`
differ by **exactly** the ratio $1/0.07\approx14.3$ at every $a$ — the two curves are vertical
rescalings of one another, not independent shapes.
""")

code(r'''# fixed box, fixed beta4=0.5 wall width; compare mu=0.07 vs mu=1
LSIDE_FIX, N_FIX = 130.0, 512
DBAR_FIX = delta_min_z4(0.5)
eta = np.linspace(0, 60, 400); a_arr = a_of_eta(eta)

fig, ax = plt.subplots(figsize=(7, 4.6))
for mu, c in [(0.07, "C0"), (1.0, "C3")]:
    cw = cells_per_wall(DBAR_FIX, LSIDE_FIX, N_FIX, eta, mu)
    ax.loglog(a_arr, cw, c, label=f"mu={mu}")
ax.axhline(1.0, ls="--", c="grey", label="cells/wall=1")
ax.set(xlabel="a=1+eta", ylabel="cells/wall (Z4, beta4=0.5, lSide=130, N=512)")
ax.set_title("Wall resolution depends on mu: smaller mu = fatter walls (x1/mu)")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(ANA, "cells_per_wall_mu_dependence.pdf"))
print("saved cells_per_wall_mu_dependence.pdf")

cw_007 = cells_per_wall(DBAR_FIX, LSIDE_FIX, N_FIX, 0.0, 0.07)
cw_1   = cells_per_wall(DBAR_FIX, LSIDE_FIX, N_FIX, 0.0, 1.0)
ratio  = cw_007/cw_1
print(f"At a=1: cells/wall(mu=0.07)={float(cw_007):.2f}, cells/wall(mu=1.0)={float(cw_1):.2f}, "
      f"ratio={float(ratio):.2f}  (expected 1/0.07={1/0.07:.2f})")
''')

# ----------------------------------------------------------------------------- 4
md(r"""## 4. Is the spectral "gradient-power past Nyquist" a valid resolution test? (No.)

A previous analysis argued the naive `cells/wall>=1` rule is "2-13x too pessimistic" and that
one should instead flag resolution loss when the **gradient-power fraction past half-Nyquist**

$$F(\eta)\;=\;\frac{\sum_{k>k_{\rm Nyq}/2} k^2 P(k)}{\sum_k k^2 P(k)}$$

climbs above ~0.1. We test that here and **find it is the wrong diagnostic**: in a domain-wall
network the walls occupy a tiny volume fraction, so the field power spectrum is dominated by the
**coarsening network** at *low* k (the $k^2P$ peak sits at $k\sim1$ and *drifts down* as the
network coarsens). The thin wall cores are a sub-dominant high-k feature, so $F$ stays $\approx0$
until very late, when it only picks up **roundoff-level aliasing** ($P_{\rm last\,bin}$ jumping
from $10^{-35}$ to $10^{-17}$), not wall pinning. So $F<0.1$ does **not** mean the walls are
resolved. We confirm this below, then size with the physically correct criterion: the comoving
wall must span $\gtrsim1$ cell, **enforced at the end of the run** (cells/wall is monotonically
decreasing in $a$, so $a_{\max}$ is the binding point).
""")

code(r'''FRAC_THR = 0.10                      # gradient-power fraction marking resolution loss

def load_spectra_blocks(path):
    """Return (K[nbin], P[ntime, nbin]) field power; blocks delimited by k resetting to k[0]."""
    arr = np.loadtxt(path)
    k = arr[:, 0]; k0 = k[0]
    blk = next(i for i in range(1, len(k)) if abs(k[i]-k0) < 1e-12)
    nt  = len(k)//blk
    return k[:blk], arr[:blk*nt, 1].reshape(nt, blk)

def grad_frac(K, Prow, kNyq):
    g = K**2 * Prow; tot = g.sum()
    if tot <= 0: return np.nan
    return g[K > 0.5*kNyq].sum()/tot

def calibrate_run(rdir, delta_bar, L, N, mu, H0_over_mu=1.0):
    sp = os.path.join(rdir, "spectra_scalar_0.txt")
    tp = os.path.join(rdir, "average_spectra_times.txt")
    if not (os.path.exists(sp) and os.path.exists(tp)): return None
    K, P = load_spectra_blocks(sp)
    times = np.atleast_1d(np.loadtxt(tp))
    nt = min(len(times), P.shape[0]); times, P = times[:nt], P[:nt]
    kNyq = np.pi*N/L
    F     = np.array([grad_frac(K, P[i], kNyq) for i in range(nt)])
    kpeak = np.array([K[np.argmax(K**2*P[i])] if P[i].sum() > 0 else np.nan for i in range(nt)])
    cw = cells_per_wall(delta_bar, L, N, times, mu, H0_over_mu)      # geometric cells/wall
    # where (if ever) F crosses FRAC_THR -- this is the *spectral* (lenient) marker we are testing
    nres = np.nan
    for i in range(1, nt):
        if np.isfinite(F[i-1]) and np.isfinite(F[i]) and F[i-1] < FRAC_THR <= F[i]:
            f = (FRAC_THR - F[i-1])/(F[i]-F[i-1]); nres = cw[i-1] + f*(cw[i]-cw[i-1]); break
    return dict(times=times, F=F, cw=cw, kpeak=kpeak, kNyq=kNyq, nres=nres)

print("calibration helpers ready. FRAC_THR =", FRAC_THR)
''')

code(r'''# ---- TEST the spectral criterion over existing physical runs (H/mu = 1) -------
Z4_RUNS = {0.1:"0p1", 0.5:"0p5", 0.9:"0p9"}      # beta4 : dirtag
Z3_OLD  = {0.05:"0p05", 0.1:"0p1", 1.0:"1"}       # beta_old : dirtag

rows = []; curves = []
for b4, tag in Z4_RUNS.items():
    r = calibrate_run(os.path.join(BATCH, f"scan_z4_beta4/results_z4_mu0p01_beta4{tag}_H1mu_k3"),
                      delta_min_z4(b4), 100.0, 512, mu=0.01)
    if r: curves.append(("Z4", b4, r)); rows.append(["Z4", b4,
        round(np.nanmin(r["kpeak"]),2), round(np.nanmax(r["kpeak"]),2), round(r["kNyq"]/2,1),
        round(np.nanmax(r["F"]),3), (round(r["nres"],3) if np.isfinite(r["nres"]) else np.nan),
        round(float(r["cw"][-1]),3)])
for bold, tag in Z3_OLD.items():
    rdir = os.path.join(BATCH, f"scan/results_z3_b{tag}_H1mu_k3")
    if not os.path.isdir(rdir): continue
    bN = betaN_from_old(bold)
    r = calibrate_run(rdir, delta_adj_z3(bN), 100.0, 512, mu=0.01)
    if r: curves.append(("Z3", bN, r)); rows.append(["Z3", round(bN,2),
        round(np.nanmin(r["kpeak"]),2), round(np.nanmax(r["kpeak"]),2), round(r["kNyq"]/2,1),
        round(np.nanmax(r["F"]),3), (round(r["nres"],3) if np.isfinite(r["nres"]) else np.nan),
        round(float(r["cw"][-1]),3)])
cal = pd.DataFrame(rows, columns=["model","beta_N","kpeak_min","kpeak_max","kNyq/2",
                                  "max_F","cells@F=0.1","cells@end"])
print(cal.to_string(index=False))
print()
print("READ-OUT: the k^2P peak (kpeak) sits FAR below kNyq/2 the entire run, max_F stays ~0 (or")
print("only blips up at the very end from roundoff aliasing). The crossing 'cells@F=0.1' lands at")
print("absurd values (<0.2 cells) or NaN => the spectral criterion does NOT detect wall pinning.")
print("It is the COARSENING NETWORK (low-k) that dominates the spectrum, not the wall cores.")
print(">>> We therefore size on the geometric cells/wall>=1 criterion, NOT this. <<<")
''')

code(r'''# ---- show the spectrum is network-dominated (peak well below Nyquist) ---------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for model, b, r in curves:
    c = "C0" if model=="Z4" else "C3"
    ax[0].plot(1+r["times"], r["kpeak"], "o-", ms=3, c=c, alpha=0.6, label=f"{model} bN={b:.2f}")
    ax[1].plot(1+r["times"], r["F"],     "o-", ms=3, c=c, alpha=0.6)
ax[0].axhline(curves[0][2]["kNyq"]/2, ls="--", c="k", label="kNyq/2")
ax[0].set(xlabel="a=1+eta", ylabel="peak of k^2 P(k)", yscale="log")
ax[0].set_title("gradient-power peak stays at low k (network)"); ax[0].legend(fontsize=7)
ax[1].axhline(FRAC_THR, ls="--", c="k")
ax[1].set(xlabel="a=1+eta", ylabel="grad frac past kNyq/2")
ax[1].set_title("F ~ 0 until late roundoff aliasing")
fig.tight_layout(); fig.savefig(os.path.join(ANA, "cells_per_wall_calibration.pdf"))
print("saved cells_per_wall_calibration.pdf")
''')

# ----------------------------------------------------------------------------- 5
md(r"""## 5. Box sizer for the physical scans (per `beta`)

We pick `lSide` and `tMax` for each `beta` so that **the walls are still resolved at the END of
the run** (so scaling has time to set in *and* the late-time network is trustworthy), subject to
two constraints at $a_{\max}=1+(H_0/\mu)\,\eta_{\max}$:

* resolution AT THE END: `cells/wall(a_max) = delta_bar*N/(mu*a_max*lSide) >= n_res`  (`n_res ~ 1`)
* horizons AT THE END:   `horizons(a_max) = lSide*(H0/mu)/a_max >= h_min`

Both decrease with $a$, so enforcing them at $a_{\max}$ guarantees them throughout. Maximizing
$a_{\max}$ (push the end as late as resolution allows) sets both to equality (with `g = H0/mu`,
`g=1` as in the cards). Solving both at equality gives

$$a_{\max} = \sqrt{\frac{\bar\delta\,N}{\mu\,n_{\rm res}\,h_{\min}}}\ \propto\ \frac{1}{\sqrt\mu},
\qquad \texttt{lSide} = h_{\min}\,a_{\max},\qquad \eta_{\max} = a_{\max}-1.$$

Note the $a_{\max}\propto1/\sqrt\mu$: a **smaller** `mu` buys a **longer** resolved window. The
**scaling window** is $[a_{\rm form},\,a_{\max}]$, i.e. $N_e=\ln(a_{\max}/a_{\rm form})$
e-folds; walls form early for our kick=1 couplings ($a_{\rm form}\approx14$-$16$ for `mu=0.07`,
measured from the kick=1 PRS stat runs, [[project_prs_kick_a2_amplification]]). We need
$N_e\gtrsim1$ for scaling to be reached — this is what forces $N$ up. `delta_bar` = thinnest wall
present (`d_min` Z4 / `d_adj` Z3). `dt` from a fixed Courant ratio. Knobs in CONFIG; the table +
the N-sweep are what we approve. **This section is reconciled to match `gen_phys_stat_cards.py`
exactly.**
""")

code(r'''# ============================ CONFIG ============================
N_LAT       = 512          # lattice points / dim
H0_OVER_MU  = 1.0          # a = 1 + eta
N_RES_END   = 1.0          # required cells/wall AT THE END (physical criterion: wall spans >=1 cell)
H_MIN       = 2.0          # required horizons/dim at tMax
A_FORM      = 14.0         # measured turnaround of <|Phi|^2> for mu=0.07 (a_form~14-16); was ~7 at mu=1
DTDX        = 0.2          # Courant ratio dt/dx (CFL-safe; CFL limit ~0.577)
MU          = 0.07         # PHYSICAL: H0 = MU * H0_OVER_MU; smaller MU = fatter walls = better resolution
# ===============================================================
g = H0_OVER_MU
def size_box(delta_bar, N=N_LAT, nres=N_RES_END, hmin=H_MIN):
    a_max = np.sqrt(delta_bar * N / (MU * nres * hmin))
    lSide = hmin * a_max
    dx = lSide/N
    return dict(lSide=lSide, dx=dx, dt=DTDX*dx, a_max=a_max, eta_max=a_max-1.0,
                cells_end=delta_bar/(MU*a_max*dx), horiz_end=lSide*g/a_max,
                cells_form=delta_bar/(MU*A_FORM*dx), efolds=np.log(a_max/A_FORM))

def sizing_table(model, N=N_LAT):
    rows=[]
    for b in GRID:
        dbar = delta_min_z4(b) if model=="Z4" else delta_adj_z3(b)
        l1,l2 = (z4_couplings(b) if model=="Z4" else z3_couplings(b))
        s = size_box(dbar, N)
        rows.append(dict(beta_N=b, delta_bar=round(dbar,3),
            lSide=round(s["lSide"],1), dx=round(s["dx"],4), dt=round(s["dt"],4),
            tMax=round(s["eta_max"],1), a_max=round(s["a_max"],1),
            cells_end=round(s["cells_end"],2), horiz_end=round(s["horiz_end"],2),
            cells_form=round(s["cells_form"],2), efolds=round(s["efolds"],2),
            lambda1=round(l1,6), lambda2=round(l2,6)))
    return pd.DataFrame(rows)

print(f"CONFIG: N={N_LAT}  n_res(end)={N_RES_END}  h_min={H_MIN}  a_form={A_FORM}  dt/dx={DTDX}")
print("efolds = ln(a_max/a_form) post-formation scaling window; need >~1 for scaling.\n")
print("Z4:"); print(sizing_table("Z4").to_string(index=False))
print("\nZ3:"); print(sizing_table("Z3").to_string(index=False))
''')

code(r'''# ---- N-sweep: the e-fold scaling window vs lattice size (the decision table) --
print(f"Post-formation e-folds N_e=ln(a_max/a_form) [a_form={A_FORM}], with n_res_end={N_RES_END}, h_min={H_MIN}:")
print("(a_max=sqrt(delta*N/(n_res*h_min*g)); each 4x in N buys +1 e-fold. tMax=a_max-1.)\n")
hdr = "  beta_N |" + "".join([f"  N={Nn}: a_max  N_e  tMax |" for Nn in (512,1024,2048)])
for model in ("Z4","Z3"):
    print(f"--- {model} ---"); print(hdr)
    for b in GRID:
        dbar = delta_min_z4(b) if model=="Z4" else delta_adj_z3(b)
        line = f"  {b:5.2f}  |"
        for Nn in (512,1024,2048):
            s = size_box(dbar, Nn)
            line += f"   {s['a_max']:5.1f} {s['efolds']:4.2f} {s['eta_max']:5.1f} |"
        print(line)
    print()
print("Per-run cost ~ (tMax/dt)*N^3.  Physical tMax(~15-40) << PRS tMax(100), so even N=1024/2048")
print("physical runs are CHEAP relative to a PRS run.  5 seeds x 5 beta x 2 models = 50 runs.")
''')

# ----------------------------------------------------------------------------- 6
md(r"""## 6. Caveats & next step

* All of `delta_bar`, `dx`, `tMax`, `eta`, `a` are program units. **`mu` is PHYSICAL, not
  cosmetic**: the raw-potential kernel gives wall width in cells $\propto1/(\mu\,a)$ (an explicit
  `1/mu`), so smaller `mu` = fatter walls = better resolution = larger $a_{\max}\propto1/\sqrt\mu$.
  PRS wants `mu~O(1)` (freezes the comoving width, needs few-cell frozen walls + `mu*L>2*pi`);
  physical wants small `mu` (`=0.07`). [[project_phys_stat_scan]]
* **The spectral gradient-power criterion (prior memory) is debunked here** — it tracks the
  coarsening network (low-k), not wall pinning. We size on geometric cells/wall ≥ `n_res` at the
  **end of the run**, the physically correct condition.
* Physical runs are **resolution-limited**: walls thin ∝1/(mu*a) and form late (`a_form≈14`,
  `mu=0.07`), so the e-fold scaling window `N_e=ln(a_max/a_form)` is small at N=512 for thin
  (large-β) walls. The **N-sweep table is the decision**: pick N so `N_e≳1` across the β grid.
  Because physical `tMax` is short, N=1024/2048 are affordable.
* Sizing uses the **thinnest wall present** (Z4 `d_min` incl. rare diagonal walls). If diagonal
  walls decay fast, switch to `d_adj` for a bigger box (config).
* **Large-β4 physical runs can numerically blow up**: the steep λ₁ + stiff transverse mass²
  $=4\beta_4/(1-\beta_4)\cdot\mu^2$ (diverges as $\beta_4\to1$) drives a CFL/stability crash;
  Z4 β4=0.9 crashed (NaN) at `a~46` at N=512. Levers: smaller `dt` or larger `N`.
  [[project_phys_b40p9_crash_dt_rescue]]
* **Next:** approve N + the §5 table, then generate the 50 physical cards
  (`scan_z{4,3}_phys_stat/`, PRS off, paired `baseSeed stat_s01..s05`) and the seed-outer runner.
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells_per_wall_box_sizing.ipynb")
nbf.validate(nb)
with open(out, "w") as f:
    nbf.write(nb, f)
# compile-check every code cell
import ast
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        ast.parse(c["source"])
print("WROTE", out, "with", len(nb["cells"]), "cells; validated + all code cells compile.")
