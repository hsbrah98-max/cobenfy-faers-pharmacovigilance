#!/usr/bin/env python3
"""
Generate ultra-clean, single-glance 300-DPI figures for the APA 2027 Cobenfy
(xanomeline-trospium) polypharmacy study.

Reads 100% of data dynamically from data/analysis_summary.json:
- Figure 1: Forest Plot of Reporter-Adjusted aROR (Strong ACB-3 vs. Cobenfy Alone), with the
  negative control composite as the last row
- Figure 2: 1x2 Two-Panel Breakdown with matching colors:
    Panel A: Specific Concomitant Medications (Red = Strong Antimuscarinics; Blue = Non-Antimuscarinic APs)
    Panel B: Side-Effect Rates Across the 3 Mutually Exclusive Study Groups (Red vs. Blue vs. Gray Cobenfy Alone)
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.9

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "analysis_summary.json")
FIG_DIR = os.path.join(BASE_DIR, "figures")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    with open(DATA_PATH, "r") as f:
        summary = json.load(f)

    n_s1 = summary["n_ach_ah"]
    n_s2 = summary["n_d2_only"]
    n_mono = summary["n_pure_mono"]
    results = summary["results"]
    by_label = {r["label"]: r for r in results}

    # --- FIGURE 1: Clean Single-Model Forest Plot (Reporter-Adjusted aROR Only) ---
    forest_order = [
        ("Hospitalization (All-Cause)", "All-Cause Hospitalization"),
        ("Non-Psychiatric Medical Hospitalization", "Medical Hospitalization (Non-Psych)"),
        ("GI Hypomotility (Constipation / Ileus / Obstruction)", "GI Hypomotility (Constip. / Ileus / Obst.)"),
        ("Constipation (Single MedDRA PT)", "Constipation (Single Term)"),
        ("Gastroesophageal Reflux Disease (GERD)", "Acid Reflux (GERD)"),
        ("Urinary Retention (Nominal; Attenuates in HCP)", "Urinary Retention"),
    ]

    labels, s1_str, ref_str, aror_str = [], [], [], []
    a_ror, a_lo, a_hi, sig_flags = [], [], [], []

    by_label.update({r["label"]: r for r in summary["negative_controls"]})
    forest_order.append(("Negative Control Composite (Any of 5)", "Negative Control (Rash / URI / Arthralgia /\nBack Pain / Pruritus)"))

    for key, short_lbl in forest_order:
        r = by_label[key]
        ach = r["ach_vs_mono"]
        mv = r["multivariable_logit"]

        labels.append(short_lbl)
        s1_str.append(f"{ach['a']}/{ach['n1']} ({ach['pct1']:.1f}%)")
        ref_str.append(f"{ach['c']}/{ach['n0']} ({ach['pct0']:.1f}%)")
        is_sig = mv["s1_p"] < 0.05
        sig_flags.append(is_sig)
        p_s = "p < 0.0001*" if mv["s1_p"] < 0.0001 else (f"p = {mv['s1_p']:.4f}*" if is_sig else f"p = {mv['s1_p']:.2f} (NS)")
        aror_str.append(f"{mv['s1_aror']:.2f} ({mv['s1_ci_l']:.2f}–{mv['s1_ci_u']:.2f}), {p_s}")

        a_ror.append(mv["s1_aror"])
        a_lo.append(mv["s1_ci_l"])
        a_hi.append(mv["s1_ci_u"])

    fig, ax = plt.subplots(figsize=(13.8, 5.6), dpi=300)
    y_pos = np.arange(len(labels))[::-1]

    ax.axvline(1.0, color="#222222", linestyle="--", linewidth=1.2, zorder=1)
    for idx, y in enumerate(y_pos):
        if idx % 2 == 0:
            ax.axhspan(y - 0.42, y + 0.42, color="#F4F6F9", alpha=0.85, zorder=0)

    for i, y in enumerate(y_pos):
        col = "#B22222" if sig_flags[i] else "#777777"
        ax.errorbar(
            a_ror[i], y, xerr=[[a_ror[i] - a_lo[i]], [a_hi[i] - a_ror[i]]],
            fmt="s", color=col, ecolor=col, elinewidth=2.4, capsize=4.5, markersize=8.0, zorder=4,
        )

    ax.set_xscale("log")
    ax.set_xlim(0.4, 8.0)
    ax.set_xticks([0.5, 1, 2, 3, 5, 7])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Reporter-Adjusted Reporting Odds Ratio (aROR, 95% CI)", fontsize=10.5, fontweight="bold", labelpad=8)

    trans = ax.get_yaxis_transform()
    hdr_y = len(labels) - 0.32
    ax.text(1.04, hdr_y, f"ACB-3 (n={n_s1})", transform=trans, fontsize=9.5, fontweight="bold", ha="left")
    ax.text(1.42, hdr_y, f"Alone (n={n_mono:,})", transform=trans, fontsize=9.5, fontweight="bold", ha="left")
    ax.text(2.02, hdr_y, "Adjusted aROR (95% CI), p", transform=trans, fontsize=9.5, fontweight="bold", ha="left", color="#B22222")

    for i, y in enumerate(y_pos):
        col = "#B22222" if sig_flags[i] else "#555555"
        ax.text(1.04, y, s1_str[i], transform=trans, fontsize=9.2, va="center", ha="left")
        ax.text(1.42, y, ref_str[i], transform=trans, fontsize=9.2, va="center", ha="left")
        ax.text(2.02, y, aror_str[i], transform=trans, fontsize=9.2, va="center", ha="left", fontweight="bold", color=col)

    ax.set_title(
        f"Figure 1. Disproportionate Reporting With Concomitant Strong Antimuscarinics (ACB=3, n={n_s1}) vs. Cobenfy Alone (n={n_mono:,})",
        fontsize=11.5, fontweight="bold", pad=22, loc="left",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.subplots_adjust(left=0.26, right=0.48, top=0.86, bottom=0.16)

    fig1_path = os.path.join(FIG_DIR, "fig1_cobenfy_polypharmacy_forest_plot.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig1_path}")

    # --- FIGURE 2: 1x2 Two-Panel Breakdown with Matching 3-Group Colors ---
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(17.0, 6.4), dpi=300)

    # Panel A: Specific Concomitant Agents
    spec = summary["specific_counts"]
    sorted_drugs = sorted(spec.items(), key=lambda kv: kv[1], reverse=True)
    d_names = [k for k, _ in sorted_drugs][::-1]
    d_vals = [v for _, v in sorted_drugs][::-1]
    d_cols = ["#B22222" if ("ACB 3" in n or "Glycopyrrolate" in n or "Meclizine" in n) else "#2B5B84" for n in d_names]

    bars = ax1.barh(d_names, d_vals, color=d_cols, height=0.72, edgecolor="#222222", linewidth=0.5)
    for bar, val in zip(bars, d_vals):
        ax1.text(val + 1.5, bar.get_y() + bar.get_height() / 2, f"n={val}", va="center", fontsize=8.2, fontweight="bold")
    ax1.set_xlim(0, max(d_vals) + 18)
    ax1.set_xlabel("Number of FAERS Reports Co-Prescribed with Cobenfy", fontsize=9.8, fontweight="bold")
    ax1.set_title("A. Which Drugs Are Co-Prescribed? (Red = Strong ACB-3; Blue = Non-Anti-M AP)", fontsize=10.3, fontweight="bold", pad=10)
    ax1.tick_params(axis="y", labelsize=8.2)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B: Side-Effect Rates Across the 3 Mutually Exclusive Study Groups
    # Same color language as Panel A: Red = Strong ACB-3 (n=260), Blue = Non-Antimuscarinic AP (n=164), Gray = Cobenfy Alone (n=1,724)
    panel_b_specs = [
        ("Hospitalization (All-Cause)", "All-Cause\nHosp."),
        ("Non-Psychiatric Medical Hospitalization", "Medical\nHosp."),
        ("GI Hypomotility (Constipation / Ileus / Obstruction)", "GI Hypomotility\n(Composite)"),
        ("Constipation (Single MedDRA PT)", "Constipation\n(Single Term)"),
        ("Gastroesophageal Reflux Disease (GERD)", "Acid Reflux\n(GERD)"),
    ]

    cats, r_s1, r_s2, r_ref = [], [], [], []
    for full_k, short_k in panel_b_specs:
        row = by_label[full_k]
        cats.append(short_k)
        r_s1.append(row["ach_vs_mono"]["pct1"])
        r_s2.append(row["d2_vs_mono"]["pct1"])
        r_ref.append(row["ach_vs_mono"]["pct0"])

    x = np.arange(len(cats))
    w = 0.25
    b1 = ax2.bar(x - w, r_s1, w, label=f"Cobenfy + Strong ACB-3 Antimuscarinic (n = {n_s1})", color="#B22222", edgecolor="#222222", linewidth=0.6)
    b2 = ax2.bar(x, r_s2, w, label=f"Cobenfy + Non-Antimuscarinic AP Only (n = {n_s2})", color="#2B5B84", edgecolor="#222222", linewidth=0.6)
    b3 = ax2.bar(x + w, r_ref, w, label=f"Cobenfy Alone / Reference (n = {n_mono:,})", color="#888888", edgecolor="#222222", linewidth=0.6)

    for bars_grp, col in [(b1, "#B22222"), (b2, "#2B5B84"), (b3, "#555555")]:
        for b in bars_grp:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2, h + 0.35, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color=col)

    ax2.set_xticks(x)
    ax2.set_xticklabels(cats, fontsize=9.2, fontweight="bold")
    ax2.set_ylabel("Percentage of Patients in Group Reporting Event (%)", fontsize=9.8, fontweight="bold")
    ax2.set_ylim(0, 22.0)
    ax2.set_title("B. Reported Event Rates: Elevated Only in the Strong ACB-3 Group", fontsize=10.3, fontweight="bold", pad=10)
    ax2.legend(frameon=False, fontsize=9.0, loc="upper right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout(w_pad=2.8)
    fig2_path = os.path.join(FIG_DIR, "fig2_cobenfy_concomitant_breakdown.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {fig2_path}")


if __name__ == "__main__":
    main()
