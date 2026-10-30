"""Generates every chart used in the thesis from the stored result files.

    python scripts/make_figures.py            # writes PNGs to docs/figures/

Figures that only need results/final/ always run. The dataset figure needs the TravelPlanner
data in data/ (or THESIS_DATASET_ROOT) and is skipped with a message if it is missing.
Pairing and filtering are the same as in src/analyze.py: a query counts only if both
conditions produced a record without an error.
"""
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from src.analyze import FINAL_BASELINE, FINAL_GATEWAY  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100, "savefig.dpi": 200,
})
BASE, GATE = "#8c8c8c", "#1f5fa6"
LEVELS = ["easy", "medium", "hard"]


def load(files):
    out = {}
    for f in files:
        for line in open(config.RESULTS_DIR / f, encoding="utf-8"):
            r = json.loads(line)
            r["_batch"] = f
            out[(f, r["idx"])] = r
    return list(out.values())


base_all, gate_all = load(FINAL_BASELINE), load(FINAL_GATEWAY)
B = {r["idx"]: r for r in base_all if not r["error"]}
G = {r["idx"]: r for r in gate_all if not r["error"]}
M = sorted(set(B) & set(G))
b = [B[i] for i in M]
g = [G[i] for i in M]
level = np.array([r["level"] for r in b])
tok = lambda r: r["input_tokens"] + r["output_tokens"]  # noqa: E731


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mean_ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return x.mean(), x.mean(), x.mean()
    m, s = x.mean(), stats.sem(x)
    lo, hi = stats.t.interval(0.95, len(x) - 1, loc=m, scale=s)
    return m, lo, hi


def groups():
    yield "All", np.ones(len(M), bool)
    for lv in LEVELS:
        yield lv.capitalize(), level == lv


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def grouped_bars(ax, vals, los, his, labels, ylabel, fmt, ymax=None):
    x = np.arange(len(labels)); w = 0.36
    for j, (lab, col) in enumerate([("Baseline", BASE), ("Gateway", GATE)]):
        v = np.array(vals[j]); err = [v - np.array(los[j]), np.array(his[j]) - v]
        ax.bar(x + (j - 0.5) * w, v, w, color=col, label=lab, yerr=err, capsize=3, error_kw={"lw": 0.8})
        for xi, vi in zip(x + (j - 0.5) * w, v):
            ax.text(xi, 0.01 * (ymax or 1), fmt(vi), ha="center", va="bottom", fontsize=7.5, color="white")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel(ylabel)
    if ymax: ax.set_ylim(0, ymax)
    ax.legend(frameon=False, loc="upper right", fontsize=8)


# ------------------------------------------------------------------ Figure 5.1 TCR
def fig_tcr():
    labs, vals, los, his = [], [[], []], [[], []], [[], []]
    for name, mask in groups():
        n = int(mask.sum()); labs.append(f"{name}\n(n = {n})")
        for j, rs in enumerate((b, g)):
            k = sum(int(r["tcr"]) for r, m in zip(rs, mask) if m)
            lo, hi = wilson(k, n); vals[j].append(100 * k / n); los[j].append(100 * lo); his[j].append(100 * hi)
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    grouped_bars(ax, vals, los, his, labs, "Task Completion Rate (%)", lambda v: f"{v:.0f}", ymax=110)
    save(fig, "fig_tcr.png")


# ------------------------------------------------------------------ Figure 5.2 CSR
def fig_csr():
    labs, vals, los, his = [], [[], []], [[], []], [[], []]
    for name, mask in groups():
        labs.append(f"{name}\n(n = {int(mask.sum())})")
        for j, rs in enumerate((b, g)):
            m, lo, hi = mean_ci([r["csr"] for r, mk in zip(rs, mask) if mk])
            vals[j].append(100 * m); los[j].append(100 * max(lo, 0)); his[j].append(100 * min(hi, 1))
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    grouped_bars(ax, vals, los, his, labs, "Constraint Satisfaction Rate (%)", lambda v: f"{v:.0f}", ymax=110)
    save(fig, "fig_csr.png")


# ------------------------------------------------------------------ box plots (tokens, latency)
def fig_box(key, ylabel, name, fmt_note):
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    rng = np.random.default_rng(0)
    pos0 = np.arange(4) * 2.4
    labs = []
    for gi, (nm, mask) in enumerate(groups()):
        labs.append(f"{nm}\n(n = {int(mask.sum())})")
        for j, (rs, col) in enumerate(((b, BASE), (g, GATE))):
            y = [key(r) for r, mk in zip(rs, mask) if mk]
            x = pos0[gi] + (j - 0.5) * 0.9
            bp = ax.boxplot(y, positions=[x], widths=0.7, patch_artist=True, showfliers=False,
                            medianprops={"color": "black", "lw": 1.2})
            bp["boxes"][0].set(facecolor=col, alpha=0.55, edgecolor=col)
            ax.scatter(x + rng.uniform(-0.2, 0.2, len(y)), y, s=7, color=col, alpha=0.8, zorder=3, lw=0)
            ax.scatter([x], [np.mean(y)], marker="D", s=22, color="white", edgecolor="black", zorder=4, lw=0.8)
    ax.set_xticks(pos0); ax.set_xticklabels(labs); ax.set_ylabel(ylabel)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[Patch(color=BASE, alpha=.7, label="Baseline"), Patch(color=GATE, alpha=.7, label="Gateway"),
                       Line2D([], [], marker="D", ls="", markerfacecolor="white", markeredgecolor="black", label="Mean")],
              frameon=False, fontsize=8, loc="lower center", ncol=3)
    ax.set_ylim(bottom=0)
    save(fig, name)


# ------------------------------------------------------------------ Figure 5.5 rows pruned
def fig_rows():
    frac = [r["rows_pruned"] / r["rows_seen"] for r in g if r["rows_seen"]]
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    ax.hist(100 * np.array(frac), bins=np.arange(0, 105, 5), color=GATE, alpha=0.8, edgecolor="white")
    ax.axvline(100 * np.mean(frac), color="black", ls="--", lw=1)
    ax.text(100 * np.mean(frac) + 14, ax.get_ylim()[1] * 0.85, f"mean {100*np.mean(frac):.1f}%", fontsize=8, ha="center")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Share of tool-result rows removed per query (%)"); ax.set_ylabel("Number of queries")
    save(fig, "fig_rows_pruned.png")


# ------------------------------------------------------------------ Figure 5.6 by batch
def fig_batches():
    order = list(zip(FINAL_BASELINE, FINAL_GATEWAY))
    labs, tb, tg, red, ns = [], [], [], [], []
    for k, (fb, fg) in enumerate(order, 1):
        Bb = {r["idx"]: r for r in base_all if r["_batch"] == fb and not r["error"]}
        Gg = {r["idx"]: r for r in gate_all if r["_batch"] == fg and not r["error"]}
        m = sorted(set(Bb) & set(Gg)); ns.append(len(m)); labs.append(f"Batch {k}\n(n = {len(m)})")
        tb.append(100 * np.mean([int(Bb[i]["tcr"]) for i in m])); tg.append(100 * np.mean([int(Gg[i]["tcr"]) for i in m]))
        mb, mg = np.mean([tok(Bb[i]) for i in m]), np.mean([tok(Gg[i]) for i in m]); red.append(100 * (mb - mg) / mb)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3.0))
    x = np.arange(len(labs)); w = 0.36
    a1.bar(x - w / 2, tb, w, color=BASE, label="Baseline"); a1.bar(x + w / 2, tg, w, color=GATE, label="Gateway")
    a1.set_xticks(x); a1.set_xticklabels(labs, fontsize=7.5); a1.set_ylabel("TCR (%)"); a1.set_ylim(0, 115)
    a1.legend(frameon=False, fontsize=8, loc="upper left", ncol=2); a1.set_title("(a) Task Completion Rate", fontsize=9)
    a2.bar(x, red, 0.55, color=GATE); a2.set_xticks(x); a2.set_xticklabels(labs, fontsize=7.5)
    for xi, v in zip(x, red): a2.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=8)
    a2.set_ylabel("Token reduction (%)"); a2.set_ylim(0, 40); a2.set_title("(b) Tokens per query", fontsize=9)
    fig.tight_layout(); save(fig, "fig_batches.png")


# ------------------------------------------------------------------ Figure 5.7 differences with 95% CI
def effects():
    n = len(M)
    kb, kg = sum(int(r["tcr"]) for r in b), sum(int(r["tcr"]) for r in g)
    p1, p2 = kb / n, kg / n; se = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    out = {"TCR": (100 * (p2 - p1), 100 * (p2 - p1 - 1.96 * se), 100 * (p2 - p1 + 1.96 * se))}
    def welch(key, scale=1.0, rel=None):
        x, y = np.array([key(r) for r in b], float), np.array([key(r) for r in g], float)
        res = stats.ttest_ind(y, x, equal_var=False); ci = res.confidence_interval(0.95)
        d = y.mean() - x.mean()
        return (d * scale, ci.low * scale, ci.high * scale) if rel is None else (100 * d / x.mean(), 100 * ci.low / x.mean(), 100 * ci.high / x.mean())
    out["CSR"] = welch(lambda r: r["csr"], 100)
    out["Tokens"] = welch(tok, rel=True)
    out["Latency"] = welch(lambda r: r["latency_s"])
    return out


def fig_forest():
    e = effects()
    spec = [("TCR", "Task Completion Rate\n(percentage points)"), ("CSR", "Constraint Satisfaction Rate\n(percentage points)"),
            ("Tokens", "Tokens per query\n(% change)"), ("Latency", "Model time per query\n(seconds)")]
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.5))
    for ax, (k, title) in zip(axes, spec):
        d, lo, hi = e[k]
        ax.axvline(0, color="black", lw=0.8)
        ax.plot([lo, hi], [0, 0], color=GATE, lw=2.2); ax.plot([d], [0], "o", color=GATE, ms=6)
        ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_title(title, fontsize=8)
        pad = (hi - lo) * 0.35; ax.set_xlim(min(lo, 0) - pad, max(hi, 0) + pad)
        ax.text(d, 0.22, f"{d:+.1f}", ha="center", fontsize=8); ax.set_ylim(-0.5, 0.6)
        ax.tick_params(axis="x", labelsize=7.5)
    fig.suptitle("Gateway minus Baseline, with 95% confidence intervals", fontsize=9, y=1.02)
    fig.tight_layout(); save(fig, "fig_effects.png")


# ------------------------------------------------------------------ Figure 6.1 summary
def fig_summary():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3.0))
    x = np.arange(3); w = 0.36
    for j, (rs, col, lab) in enumerate(((b, BASE, "Baseline"), (g, GATE, "Gateway"))):
        mt = [np.mean([tok(r) for r, lv in zip(rs, level) if lv == L]) for L in LEVELS]
        tc = [100 * np.mean([int(r["tcr"]) for r, lv in zip(rs, level) if lv == L]) for L in LEVELS]
        a1.bar(x + (j - 0.5) * w, mt, w, color=col, label=lab); a2.bar(x + (j - 0.5) * w, tc, w, color=col, label=lab)
    lbl = [f"{L.capitalize()}\n(n = {int((level == L).sum())})" for L in LEVELS]
    for a in (a1, a2): a.set_xticks(x); a.set_xticklabels(lbl)
    a1.set_ylabel("Mean tokens per query"); a1.set_title("(a) Cost: lower with the Gateway", fontsize=9)
    a2.set_ylabel("TCR (%)"); a2.set_title("(b) Completion: not improved", fontsize=9); a2.set_ylim(0, 110)
    a2.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout(); save(fig, "fig_summary.png")


# ------------------------------------------------------------------ Figure 1.1 context growth (Baseline)
def fig_growth():
    rows = [(r["react_steps"], r["input_tokens"]) for r in B.values()]
    steps = np.array([s for s, _ in rows]); inp = np.array([t for _, t in rows])
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(6.0, 3.1))
    ax.scatter(steps + rng.uniform(-0.12, 0.12, len(steps)), inp / 1000, s=14, color=BASE, alpha=0.8, lw=0)
    ms = sorted(set(steps)); ax.plot(ms, [inp[steps == s].mean() / 1000 for s in ms], color="black", lw=1.3, marker="o", ms=3.5)
    r = np.corrcoef(steps, inp)[0, 1]
    ax.set_xlabel("Reason-Act steps used by the query"); ax.set_ylabel("Input tokens per query (thousands)")
    ax.text(0.03, 0.93, f"Baseline condition, n = {len(rows)} completed queries\nPearson r = {r:.2f}; line = mean per step count",
            transform=ax.transAxes, fontsize=8, va="top")
    ax.set_xticks(ms)
    save(fig, "fig_context_growth.png")


# ------------------------------------------------------------------ Figure 3.2 dataset
def fig_dataset():
    try:
        from src import data_loader
        qs = data_loader.load_queries("validation")
    except Exception as exc:  # noqa: BLE001
        print("dataset figure skipped:", exc); return
    sample = [qs[i] for i in M]
    fig, axes = plt.subplots(2, 3, figsize=(6.7, 4.6))
    bud = np.array([q.budget for q in qs])
    ax = axes[0, 0]; ax.hist(bud, bins=np.arange(0, 25000, 1500), color=BASE, edgecolor="white")
    ax.axvline(np.median(bud), color="black", ls="--", lw=1); ax.set_title(f"(a) Budget, all 180\nskewness {stats.skew(bud):.2f}", fontsize=8.5)
    ax.set_xlabel("USD"); ax.set_ylabel("Queries"); ax.tick_params(labelsize=7)
    ax = axes[0, 1]; data = [[q.budget for q in qs if q.level == L] for L in LEVELS]
    ax.boxplot(data, tick_labels=[L.capitalize() for L in LEVELS], showfliers=True, flierprops={"ms": 3}, medianprops={"color": "black"}); ax.set_title("(b) Budget by level", fontsize=8.5)
    ax.set_ylabel("USD"); ax.tick_params(labelsize=7)
    ax = axes[0, 2]; pc = np.array([q.people_number for q in qs]); vals, cnt = np.unique(pc, return_counts=True)
    ax.bar(vals, cnt, color=BASE); ax.set_title(f"(c) Party size, all 180\nskewness {stats.skew(pc):.2f}", fontsize=8.5); ax.set_xlabel("Travellers"); ax.tick_params(labelsize=7)
    def cmp(ax, key, title, cats):
        x = np.arange(len(cats)); w = 0.38
        full = [sum(key(q) == c for q in qs) for c in cats]; samp = [sum(key(q) == c for q in sample) for c in cats]
        ax.bar(x - w / 2, full, w, color=BASE, label="Validation split (180)"); ax.bar(x + w / 2, samp, w, color=GATE, label="Matched sample (57)")
        ax.set_xticks(x); ax.set_xticklabels([str(c).capitalize() for c in cats]); ax.set_title(title, fontsize=8.5); ax.tick_params(labelsize=7)
    cmp(axes[1, 0], lambda q: q.days, "(d) Trip length (days)", [3, 5, 7])
    cmp(axes[1, 1], lambda q: q.visiting_city_number, "(e) Cities visited", [1, 2, 3])
    cmp(axes[1, 2], lambda q: q.level, "(f) Difficulty level", LEVELS)
    axes[1, 0].set_ylabel("Queries")
    h, l = axes[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 1]); save(fig, "fig_dataset.png")


if __name__ == "__main__":
    fig_growth(); fig_dataset()
    fig_tcr(); fig_csr()
    fig_box(tok, "Tokens per query", "fig_tokens.png", "")
    fig_box(lambda r: r["latency_s"], "Model time per query (s)", "fig_latency.png", "")
    fig_rows(); fig_batches(); fig_forest(); fig_summary()
