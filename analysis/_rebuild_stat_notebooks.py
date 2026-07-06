#!/usr/bin/env python3
"""Rebuild the two PRS 10-seed stat notebooks in the clean val-notebook style.

Keeps the existing (correct) infrastructure + key-result cells verbatim, sets
USETEX=True (to match Z3_prs_val_analysis.ipynb), and inserts the val-style
diagnostic sections (health, potential E, kinetic/gradient E, scalar means,
scale factor & Hubble) as seed-mean+-sigma bands, ahead of the statistical
key-result sections.  Every figure has <= 2 subplots.
"""
import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell

REF_TITLE = None  # keep existing title cell as-is


def source_index(nb, marker, startswith=False):
    for i, c in enumerate(nb.cells):
        if c.cell_type != 'code':
            continue
        s = c.source
        if (s.startswith(marker) if startswith else (marker in s)):
            return i
    raise KeyError(marker)


def src(nb, marker, startswith=False):
    return nb.cells[source_index(nb, marker, startswith)].source


# ----------------------------------------------------------------------------
# New diagnostic-section code (seed mean +- sigma bands).  has_opp toggles Z4/Z3.
# ----------------------------------------------------------------------------

HEALTH_CODE = r"""def drift_curve(run):
    # |<Phi>| = sqrt(<h>^2 + <a>^2); field 1 interpolated onto field 0's eta
    s0, s1 = run['scalar_0'], run['scalar_1']
    n = min(len(s0['eta']), len(s1['eta']))
    eta = s0['eta'][:n]
    phi1 = np.interp(eta, s1['eta'], s1['phi'])
    return eta, np.sqrt(s0['phi'][:n]**2 + phi1**2)


fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
for b in BETAS:
    eta, mean, std, n = seed_mean_std(b, phi2_radial)
    if eta.size:
        axes[0].plot(eta, mean, color=COLOR[b], lw=2.4)
        axes[0].fill_between(eta, mean - std, mean + std, color=COLOR[b], alpha=0.18, lw=0)
    etad, meand, stdd, nd = seed_mean_std(b, drift_curve)
    if etad.size:
        axes[1].plot(etad, meand, color=COLOR[b], lw=2.4)
        axes[1].fill_between(etad, meand - stdd, meand + stdd, color=COLOR[b], alpha=0.18, lw=0)

axes[0].axhline(0.5,  color='k', ls=':', lw=1.2, label='vacuum (0.5)')
axes[0].axhline(0.25, color='r', ls=':', lw=1.2, alpha=0.8, label='breaking marker (0.25)')
axes[0].legend(fontsize=12, frameon=False)
axes[1].set_yscale('log')
paper_ax(axes[0], ylabel=r'$\langle|\Phi|^2\rangle$',
         title=r'radial health (vacuum $=0.5$)')
paper_ax(axes[1], ylabel=r'$|\langle\Phi\rangle|$',
         title=r'mean-field drift (finite volume)')
add_legend(axes[1], BETAS)
plt.tight_layout(); plt.show()"""

POT_CODE = r"""fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
for b in BETAS:
    e1, m1, s1, _ = seed_mean_std(b, lambda r: (r['energies']['eta'], r['energies']['Ev_1']))
    if e1.size:
        axes[0].plot(e1, m1, color=COLOR[b], lw=2.4)
        axes[0].fill_between(e1, m1 - s1, m1 + s1, color=COLOR[b], alpha=0.18, lw=0)
    e2, m2, s2, _ = seed_mean_std(b, lambda r: (r['energies']['eta'], r['energies']['Ev_2']))
    if e2.size:
        axes[1].plot(e2, m2, color=COLOR[b], lw=2.4)
        axes[1].fill_between(e2, m2 - s2, m2 + s2, color=COLOR[b], alpha=0.18, lw=0)
    e3, m3, s3, _ = seed_mean_std(b, lambda r: (r['energies']['eta'], r['energies']['Ev_3']))
    if e3.size:
        axes[1].plot(e3, m3, color=COLOR[b], ls=(0, (3, 2)), lw=1.6, alpha=0.7)

axes[0].axhline(0, color='k', ls=':', lw=1.2)
paper_ax(axes[0], ylabel=r'$E_{V_1}$', title=r'$E_{V_1}$ (mass term $+\,V_0$)')
paper_ax(axes[1], ylabel=r'$E_{V_2},\;E_{V_3}$',
         title=r'quartic terms (solid $E_{V_2}$, dash $E_{V_3}$)')
add_legend(axes[1], BETAS)
plt.tight_layout(); plt.show()"""

KG_CODE = r"""fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
for b in BETAS:
    ek, mk, sk, _ = seed_mean_std(
        b, lambda r: (r['energies']['eta'], r['energies']['Ek_1'] + r['energies']['Ek_2']))
    if ek.size:
        axes[0].plot(ek, mk, color=COLOR[b], lw=2.4)
        axes[0].fill_between(ek, np.clip(mk - sk, 1e-30, None), mk + sk,
                             color=COLOR[b], alpha=0.18, lw=0)
    eg, mg, sg, _ = seed_mean_std(
        b, lambda r: (r['energies']['eta'], r['energies']['Eg_1'] + r['energies']['Eg_2']))
    if eg.size:
        axes[1].plot(eg, mg, color=COLOR[b], lw=2.4)
        axes[1].fill_between(eg, np.clip(mg - sg, 1e-30, None), mg + sg,
                             color=COLOR[b], alpha=0.18, lw=0)

for ax in axes:
    ax.set_xscale('log'); ax.set_yscale('log')
paper_ax(axes[0], ylabel=r'$E_K$', title=r'kinetic (both components)')
paper_ax(axes[1], ylabel=r'$E_G$', title=r'gradient (both components)')
add_legend(axes[1], BETAS)
plt.tight_layout(); plt.show()"""

MEANS_CODE = r"""fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharex=True, sharey=True)
for b in BETAS:
    eh, mh, sh, _ = seed_mean_std(b, lambda r: (r['scalar_0']['eta'], r['scalar_0']['phi']))
    if eh.size:
        axes[0].plot(eh, mh, color=COLOR[b], lw=2.4)
        axes[0].fill_between(eh, mh - sh, mh + sh, color=COLOR[b], alpha=0.18, lw=0)
    ea, ma, sa, _ = seed_mean_std(b, lambda r: (r['scalar_1']['eta'], r['scalar_1']['phi']))
    if ea.size:
        axes[1].plot(ea, ma, color=COLOR[b], lw=2.4)
        axes[1].fill_between(ea, ma - sa, ma + sa, color=COLOR[b], alpha=0.18, lw=0)

for ax in axes:
    ax.axhline(0, color='k', ls=':', lw=1.0)
paper_ax(axes[0], ylabel=r'$\langle h\rangle$', title=r'scalar 0 ($h$), band $=$ seed scatter')
paper_ax(axes[1], ylabel=r'$\langle a\rangle$', title=r'scalar 1 ($a$), band $=$ seed scatter')
add_legend(axes[1], BETAS)
plt.tight_layout(); plt.show()"""

SF_CODE = r"""fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
for b in BETAS:
    ea, ma, sa, _ = seed_mean_std(b, lambda r: (r['scale_factor']['eta'], r['scale_factor']['a']))
    if ea.size:
        axes[0].plot(ea, ma, color=COLOR[b], lw=2.4)
    eh, mh, sh, _ = seed_mean_std(b, lambda r: (r['scale_factor']['eta'], r['scale_factor']['H']))
    if eh.size:
        axes[1].plot(eh, mh, color=COLOR[b], lw=2.4)

paper_ax(axes[0], ylabel=r'$a$', title=r'scale factor (fixed radiation background)')
paper_ax(axes[1], ylabel=r"$\mathcal{H}$", title=r"conformal Hubble $\mathcal{H}=a'/a$")
add_legend(axes[1], BETAS)
plt.tight_layout(); plt.show()"""

# Z3-only wall-area panel (linear | log-log), seed mean +- sigma
WALL_Z3_CODE = r"""fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
for b in BETAS:
    eta, mean, std, n = seed_mean_std(b, lambda r: area_param('scal1', r))
    if eta.size == 0:
        continue
    axes[0].plot(eta, mean, color=COLOR[b], lw=2.4)
    axes[0].fill_between(eta, mean - std, mean + std, color=COLOR[b], alpha=0.18, lw=0)
    axes[1].plot(eta, mean, color=COLOR[b], lw=2.4)
    axes[1].fill_between(eta, np.clip(mean - std, 1e-30, None), mean + std,
                         color=COLOR[b], alpha=0.18, lw=0)

for ax in axes:
    ax.axhline(1.0, color='gray', ls=':', lw=1.2, alpha=0.7)
axes[1].set_xscale('log'); axes[1].set_yscale('log'); horizon_guides(axes[1])
paper_ax(axes[0], ylabel=r'$\mathcal{A}_1$', title=r'wall area (linear, seed mean $\pm\sigma$)')
paper_ax(axes[1], ylabel=r'$\mathcal{A}_1$', title=r'wall area (log--log)')
add_legend(axes[1], BETAS)
plt.tight_layout(); plt.show()"""


# ----------------------------------------------------------------------------
# Markdown blocks
# ----------------------------------------------------------------------------

def md_health():
    return (
        "## 1. Health: radial diagnostic $\\langle|\\Phi|^2\\rangle$ and mean-field drift\n\n"
        "**Left:** $\\langle|\\Phi|^2\\rangle = (\\langle h^2\\rangle + \\langle a^2\\rangle)/2$; "
        "guide at $0.5$ = vacuum, dashed at $0.25$ = **breaking/melt marker** (the reliable "
        "criterion for both Z4 and Z3 -- *not* $E_{V_1}<0$).\n\n"
        "**Right:** $|\\langle\\Phi\\rangle| = \\sqrt{\\langle h\\rangle^2+\\langle a\\rangle^2}$ "
        "-- growth toward $\\mathcal{O}(1)$ signals finite-volume domain depletion.\n\n"
        "Each curve is the across-seed mean; the shaded band is $\\pm1\\sigma$ across the available "
        "seeds at that $\\beta$.")


def md_pot():
    return (
        "## 2. Potential energies $E_{V_1}$, $E_{V_2}$, $E_{V_3}$\n\n"
        "$E_{V_1}=-\\mu^2\\langle|\\Phi|^2\\rangle + V_0$ with a $\\beta$-dependent vacuum offset "
        "$V_0$, so $E_{V_1}$ can stay positive even in the broken vacuum (e.g. $\\beta=0.9$). Use "
        "the $\\langle|\\Phi|^2\\rangle>0.25$ marker for breaking, **not** the sign of $E_{V_1}$. "
        "Right panel: quartic terms (solid $E_{V_2}$, dashed $E_{V_3}$). Bands are seed $\\pm1\\sigma$.")


def md_kg():
    return (
        "## 3. Kinetic and gradient energies\n\n"
        "Both scalar components summed; log--log. Seed mean $\\pm1\\sigma$ band per $\\beta$.")


def md_means():
    return (
        "## 4. Scalar-field means $\\langle h\\rangle$, $\\langle a\\rangle$\n\n"
        "Per-component volume-averaged field. Here the band is **seed scatter** "
        "($\\pm1\\sigma$ across seeds), *not* the within-volume spatial rms: each seed's mean "
        "field drifts to a random sign under finite-volume depletion, so the seed mean sits near "
        "$0$ with a band that widens as domains are depleted.")


def md_sf():
    return (
        "## 5. Scale factor and Hubble\n\n"
        "Sanity panel: $a(\\eta)$ and the conformal Hubble $\\mathcal{H}=a'/a$ read from "
        "`average_scale_factor.txt`. This is the **fixed radiation background** ($H_0=\\mu$), "
        "identical for every $\\beta$ and every seed, so all curves overlap into one line each "
        "($a$ rises linearly, $\\mathcal{H}\\propto 1/a$). It is what makes $2aH=2$ exactly, "
        "which is why $\\mathcal{A}\\equiv\\mathrm{scal}/(2aH)$ below is just $\\mathrm{scal}/2$.")


def md_wall(has_opp):
    if has_opp:
        return (
            "## 6. Wall-area parameters $\\mathcal{A}_1$ (adjacent) and $\\mathcal{A}_2$ (non-adjacent)\n\n"
            "$\\mathcal{A}_1\\equiv\\mathrm{scal}_1/(2aH)$ (adjacent walls), "
            "$\\mathcal{A}_2\\equiv\\mathrm{scal}_2/(2aH)$ (non-adjacent walls); log--log, seed mean "
            "$\\pm1\\sigma$. Dotted verticals mark the 3- and 2.5-horizon points. A plateau at "
            "$\\mathcal{A}\\sim\\mathcal{O}(1)$ is the scaling signature.")
    return (
        "## 6. Wall-area parameter $\\mathcal{A}_1$\n\n"
        "$\\mathcal{A}_1\\equiv\\mathrm{scal}_1/(2aH)$; since $2aH=2$ exactly here, "
        "$\\mathcal{A}_1=\\mathrm{scal}_1/2$. Z3 has only one wall type (single topological "
        "charge step), so there is no $\\mathcal{A}_2$. Left: linear; right: log--log (eye-guide "
        "for plateau/power-law). Seed mean $\\pm1\\sigma$; guide at $\\mathcal{A}_1=1$.")


def md_slope():
    return (
        "## 7. Local logarithmic slope ${\\rm d}\\ln\\mathcal{A}_1/{\\rm d}\\ln\\eta$\n\n"
        "Computed **per seed first** (only $\\eta>1.5\\,\\eta_{\\rm break}$ to skip the transient), "
        "then aggregated across seeds. Scaling $\\Leftrightarrow$ slope $\\to0$ (horizontal guide).")


def md_window():
    return (
        "## 8. Slope-in-window vs $\\beta$ --- THE key result\n\n"
        "Per seed, fit a line to $\\ln\\mathcal{A}_1$ vs $\\ln\\eta$ over $\\eta\\in[20,\\eta_{\\rm hi}]$ "
        "($\\eta_{\\rm hi}$ = the 2.5-horizon $\\eta$, in practice the run's last $\\eta\\approx100$). "
        "Errorbar = across-seed mean $\\pm1\\sigma$. A slope consistent with $0$ within error means "
        "scaling.")


def md_plateau():
    return (
        "## 9. $\\mathcal{A}_1$ plateau value vs $\\beta$\n\n"
        "Seed mean $\\pm1\\sigma$ of $\\mathcal{A}_1$ at the fiducial late $\\eta=80$ (interpolated "
        "per seed), errorbar vs $\\beta$.")


def md_ratio():
    return (
        "## 10. non-adjacent-wall $\\mathcal{A}_2$ and the ratio $\\mathcal{A}_2/\\mathcal{A}_1$ (Z4 only)\n\n"
        "$\\bar\\sigma_{\\rm opp}=2\\bar\\sigma_{\\rm adj}$ exactly at $\\beta_4=1/3$: below this the "
        "non-adjacent ($\\pi$-jump) walls split into two adjacent walls (decaying ratio); above it they "
        "are bound (stable/rising ratio). Left: $\\mathcal{A}_2$ seed mean $\\pm1\\sigma$; right: the "
        "ratio across the $\\beta_4=1/3$ transition.")


def md_summary(has_opp):
    extra = (" $\\mathcal{A}_1$ plateau and in-window slope (mean$\\pm$std), and the mean "
             "$\\mathcal{A}_2$ at the end of the run.") if has_opp else \
            (" and the $\\mathcal{A}_1$ plateau and in-window slope (mean$\\pm$std).")
    n = "11" if has_opp else "10"
    return (f"## {n}. Summary table\n\nPer $\\beta$: seeds present, mean breaking-onset $\\eta$, mean "
            f"$\\langle|\\Phi|^2\\rangle$ at the end of the run,{extra}")


# ----------------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------------

def build(path, model):
    has_opp = (model == 'Z4')
    nb = nbformat.read(path, as_version=4)

    title = nb.cells[0]  # keep title markdown verbatim

    imports = src(nb, 'import os', startswith=True).replace(
        'USETEX = False', 'USETEX = True')

    config   = src(nb, 'def run_dir')
    loaders  = src(nb, 'def load_energies')
    disc     = src(nb, 'Discovery:')
    helpers  = src(nb, 'def paper_ax')
    seedagg  = src(nb, 'def seed_mean_std')

    logslope = src(nb, 'def logslope_curve')
    window   = src(nb, 'ETA_LO = 20.0')
    plateau  = src(nb, 'ETA_FID = 80.0')
    summary  = src(nb, 'def eb_per_seed')

    cells = [
        title,
        new_code_cell(imports),
        new_markdown_cell("## Configuration"),
        new_code_cell(config),
        new_markdown_cell("## Loaders\n\nPer-(`beta`,`seed`) loader returns the four data "
                          "products or `None` if any file is missing/empty/too-short "
                          "(`plot-if-ran`: missing seeds are silently skipped downstream)."),
        new_code_cell(loaders),
        new_markdown_cell("## Load all runs\n\nDiscovery: how many of the 10 seeds are "
                          "present per $\\beta$, and how far each ran."),
        new_code_cell(disc),
        new_markdown_cell("## Plot helpers"),
        new_code_cell(helpers),
        new_markdown_cell("## Seed-aggregation helper\n\n`seed_mean_std(beta, quantity_fn)`: "
                          "for each available seed compute `quantity_fn(run) -> (eta, y)`, "
                          "interpolate every seed onto a common $\\eta$ grid, and return the "
                          "across-seed mean and std (NaN-tolerant of partial coverage)."),
        new_code_cell(seedagg),

        new_markdown_cell(md_health()),
        new_code_cell(HEALTH_CODE),
        new_markdown_cell(md_pot()),
        new_code_cell(POT_CODE),
        new_markdown_cell(md_kg()),
        new_code_cell(KG_CODE),
        new_markdown_cell(md_means()),
        new_code_cell(MEANS_CODE),
        new_markdown_cell(md_sf()),
        new_code_cell(SF_CODE),
        new_markdown_cell(md_wall(has_opp)),
    ]

    if has_opp:
        cells.append(new_code_cell(src(nb, 'sharex=True')))  # existing A1|A2 panel
    else:
        cells.append(new_code_cell(WALL_Z3_CODE))

    cells += [
        new_markdown_cell(md_slope()),
        new_code_cell(logslope),
        new_markdown_cell(md_window()),
        new_code_cell(window),
        new_markdown_cell(md_plateau()),
        new_code_cell(plateau),
    ]

    if has_opp:
        cells += [
            new_markdown_cell(md_ratio()),
            new_code_cell(src(nb, 'def ratio_curve')),
        ]

    cells += [
        new_markdown_cell(md_summary(has_opp)),
        new_code_cell(summary),
    ]

    nb.cells = cells
    nbformat.validate(nb)
    nbformat.write(nb, path)
    print(f"wrote {path}: {len(cells)} cells")


if __name__ == '__main__':
    build('Z4_prs_stat_analysis.ipynb', 'Z4')
    build('Z3_prs_stat_analysis.ipynb', 'Z3')
