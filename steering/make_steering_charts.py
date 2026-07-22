import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
GREEN = "#008300"
RED_TINT = "#e34948"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "text.color": INK_PRIMARY,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

# ============================================================
# Chart 1: dose-response line chart (misalignment% and coherence% vs scale)
# ============================================================
scales = [0, 2, 4, 5, 6, 7, 8, 12, 16, 24]
misalign, coherence, mean_aligned = [], [], []
for s in scales:
    df = pd.read_csv(f"/mnt/localssd/em/results/steering/sweep/scale_{s}.csv").dropna(subset=["aligned", "coherent"])
    n = len(df)
    misalign.append(100 * len(df[(df["aligned"] <= 30) & (df["coherent"] > 50)]) / n)
    coherence.append(100 * len(df[df["coherent"] > 50]) / n)
    mean_aligned.append(df["aligned"].mean())

fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)

ax.plot(scales, misalign, color=BLUE, linewidth=2, marker="o", markersize=5, zorder=3, label="Misalignment rate (aligned≤30 & coherent>50)")
ax.plot(scales, coherence, color=GREEN, linewidth=2, marker="o", markersize=5, zorder=3, label="Coherence rate (coherent>50)")

ax.set_xlabel("Steering scale (added to layer-24 residual stream)", fontsize=10.5)
ax.set_ylabel("Rate (%)", fontsize=10.5)
ax.set_ylim(-3, 103)
ax.set_xticks(scales)
ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color(BASELINE)
ax.spines["bottom"].set_color(BASELINE)
ax.tick_params(axis="both", length=0)

legend = ax.legend(loc="upper right", frameon=False, fontsize=9.5)
for text in legend.get_texts():
    text.set_color(INK_SECONDARY)

ax.set_title(
    "Adding the misalignment direction to the base model: a peak, not a ramp\n"
    "misalignment rises then collapses with coherence as the scale increases",
    fontsize=11, color=INK_PRIMARY, loc="left", pad=14,
)

fig.tight_layout()
fig.savefig("/tmp/claude-1000/-mnt-localssd-em/668caf07-f9ba-4513-be34-d8a9e12037b9/scratchpad/charts/steering_dose_response.png", bbox_inches="tight")
plt.close(fig)

# ============================================================
# Chart 2: ablation comparison bar chart
# ============================================================
conditions = ["Un-ablated\n(control)", "Weight-space\nablation\n(LoRA B vector)", "Activation-space\nablation\n(runtime projection)"]
misalignment_vals = [9.2, 10.0, 26.5]
coherence_vals = [98.8, 98.8, 96.2]

fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=200)
x = np.arange(len(conditions))
w = 0.32

b1 = ax.bar(x - w / 2, misalignment_vals, width=w, color=BLUE, label="Misalignment rate", zorder=3)
b2 = ax.bar(x + w / 2, coherence_vals, width=w, color=GREEN, label="Coherence rate", zorder=3)

for bars in (b1, b2):
    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, color=INK_PRIMARY)

ax.set_ylim(0, 112)
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=9.5, color=INK_PRIMARY)
ax.set_ylabel("Rate (%)", fontsize=10.5)
ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(BASELINE)
ax.tick_params(axis="both", length=0)

legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.26), ncol=2, frameon=False, fontsize=10)
for text in legend.get_texts():
    text.set_color(INK_SECONDARY)

ax.set_title(
    "Ablating the mean-diff direction did not reduce misalignment\n"
    "— activation-level ablation made it worse",
    fontsize=11, color=INK_PRIMARY, loc="left", pad=40,
)

fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig("/tmp/claude-1000/-mnt-localssd-em/668caf07-f9ba-4513-be34-d8a9e12037b9/scratchpad/charts/ablation_comparison.png", bbox_inches="tight")
plt.close(fig)

print("done")
