#!/usr/bin/env python3
"""
Generate publication-quality 300-DPI figures dynamically from data/analysis_summary.json
following the 2024 READUS-PV pharmacovigilance reporting guidelines:
Figure 1: Multi-Domain Forest Plot of Multivariable Reporter-Adjusted Odds Ratios (aROR, 95% CI)
          and Crude RORs for Strong Antimuscarinics (ACB Score = 3, n = 260) vs. Unexposed
          Reference (n = 1,724), alongside HCP-Only (n = 222 vs. 885) sensitivity results.
Figure 2: Three-panel visualization derived 100% from analysis_summary.json:
          (A) Specific Concomitant Agents Co-Prescribed with Cobenfy
          (B) 3-Stratum Event Rates (Strong ACB-3 n=260 vs. Non-Antimuscarinic AP n=164 vs. Reference n=1,724)
          (C) 4-Model Sensitivity Robustness Check (Crude ROR, Multivariable aROR, HCP-Only, Non-Clozapine)
"""
import json
import os
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(BASE_DIR, "data", "analysis_summary.json")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    with open(SUMMARY_PATH, "r") as f:
        summary = json.load(f)

    results = summary["results"]
    n_s1 = summary["n_ach_ah"]
    n_s2 = summary["n_d2_only"]
    n_mono = summary["n_pure_mono"]
    n_hcp_s1 = summary["n_hcp_s1"]
    n_hcp_mono = summary["n_hcp_mono"]
    n_nocloz = summary["n_nocloz_s1"]

    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#333333"

    # --- FIGURE 1: Forest Plot (Multivariable aROR + Crude ROR + HCP-Only) ---
    fig, ax = plt.subplots(figsize=(16.0, 9.2), dpi=300)
    y_pos = np.arange(len(results))[::-1]

    for idx, y in enumerate(y_pos):
        if idx % 2 == 0:
            ax.axhspan(y - 0.45, y + 0.45, color="#F5F7FA", zorder=0)

    ax.axvline(1.0, color="#222222", linestyle="--", linewidth=1.3, zorder=2)

    for idx, row in enumerate(results):
        label = row["label"]
        m = row["multivariable_logit"]
        aror, ci_l, ci_u, p_val = m["s1_aror"], m["s1_ci_l"], m["s1_ci_u"], m["s1_p"]
        y = y_pos[idx]
        is_sig = p_val < 0.05

        if "Hospitalization" in label:
            color, marker = "#8B0000", "D"
        elif any(k in label for k in ("Hypomotility", "Constipation", "Reflux", "Urinary", "Vision", "Tachycardia")):
            color = "#B22222" if is_sig else "#6E7B8B"
            marker = "s" if is_sig else "o"
        elif any(k in label for k in ("Akathisia", "Sialorrhea", "Salivary", "Weight")):
            color = "#1F4E78" if is_sig else "#6E7B8B"
            marker = "s" if is_sig else "o"
        else:
            color = "#006400" if is_sig else "#6E7B8B"
            marker = "s" if is_sig else "o"

        ax.errorbar(
            aror, y,
            xerr=[[aror - ci_l], [ci_u - aror]],
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=2.1 if is_sig else 1.4,
            capsize=4,
            markersize=8.2 if is_sig else 6.5,
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_xlim(0.10, 150.0)
    ax.set_xticks([0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 100.0])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["label"] for r in results], fontsize=9.3, fontweight="bold")
    ax.set_xlabel(
        "Reporter-Adjusted Odds Ratio vs. Unexposed Reference (Multivariable aROR, 95% CI, log scale)\n"
        "< 1.0: Lower Reporting with Strong Antimuscarinics   |   > 1.0: Higher Reporting with Strong Antimuscarinics",
        fontsize=9.8, fontweight="bold", labelpad=10
    )

    trans = ax.get_yaxis_transform()
    col_x = [1.03, 1.33, 1.64, 1.98, 2.48]
    ax.text(col_x[0], len(results) - 0.30, f"Strong ACB-3\n(n = {n_s1})", transform=trans, fontsize=8.6, fontweight="bold", ha="left", va="bottom", clip_on=False)
    ax.text(col_x[1], len(results) - 0.30, f"Reference\n(n = {n_mono:,})", transform=trans, fontsize=8.6, fontweight="bold", ha="left", va="bottom", clip_on=False)
    ax.text(col_x[2], len(results) - 0.30, "Crude ROR\n(95% CI)", transform=trans, fontsize=8.6, fontweight="bold", ha="left", va="bottom", clip_on=False)
    ax.text(col_x[3], len(results) - 0.30, "Adjusted aROR\n(95% CI), p", transform=trans, fontsize=8.6, fontweight="bold", ha="left", va="bottom", clip_on=False)
    ax.text(col_x[4], len(results) - 0.30, f"HCP-Only ROR\n({n_hcp_s1} vs {n_hcp_mono})", transform=trans, fontsize=8.6, fontweight="bold", ha="left", va="bottom", clip_on=False)

    for idx, row in enumerate(results):
        label = row["label"]
        r = row["ach_vs_mono"]
        m = row["multivariable_logit"]
        r_hcp = row["hcp_ach_vs_mono"]
        y = y_pos[idx]
        p_val = m["s1_p"]
        fw = "bold" if p_val < 0.05 else "normal"
        if "Hospitalization" in label and p_val < 0.05:
            c_txt = "#8B0000"
        elif any(k in label for k in ("Hypomotility", "Constipation", "Reflux")) and p_val < 0.05:
            c_txt = "#B22222"
        elif any(k in label for k in ("Akathisia", "Salivary", "Weight")) and p_val < 0.05:
            c_txt = "#1F4E78"
        else:
            c_txt = "#333333"

        p_str = "<0.001*" if p_val < 0.001 else (f"{p_val:.3f}*" if p_val < 0.05 else f"{p_val:.2f}")
        aror_str = f"{m['s1_aror']:.2f} ({m['s1_ci_l']:.2f}-{m['s1_ci_u']:.2f}), {p_str}"
        hcp_sig = "*" if r_hcp["p_val"] < 0.05 else ""
        hcp_str = f"{r_hcp['ror']:.2f} ({r_hcp['ci_l']:.2f}-{r_hcp['ci_u']:.2f}){hcp_sig}"

        ax.text(col_x[0], y, f"{r['a']}/{r['n1']} ({r['pct1']:.1f}%)", transform=trans, fontsize=8.4, fontweight=fw, ha="left", va="center", clip_on=False)
        ax.text(col_x[1], y, f"{r['c']}/{r['n0']} ({r['pct0']:.1f}%)", transform=trans, fontsize=8.4, fontweight=fw, ha="left", va="center", clip_on=False)
        ax.text(col_x[2], y, f"{r['ror']:.2f} ({r['ci_l']:.2f}-{r['ci_u']:.2f})", transform=trans, fontsize=8.4, fontweight="normal", ha="left", va="center", color="#333333", clip_on=False)
        ax.text(col_x[3], y, aror_str, transform=trans, fontsize=8.4, fontweight=fw, ha="left", va="center", color=c_txt, clip_on=False)
        ax.text(col_x[4], y, hcp_str, transform=trans, fontsize=8.4, fontweight="bold" if r_hcp["p_val"] < 0.05 else "normal", ha="left", va="center", color="#222222", clip_on=False)

    ax.set_title(
        f"READUS-PV Disproportionality Analysis: Xanomeline-Trospium + Strong Antimuscarinics (ACB Score = 3, n = {n_s1})\n"
        f"Crude ROR, Reporter-Adjusted Multivariable aROR, and HCP-Only Sensitivity Across N = {summary['n_total']:,} FAERS Cases",
        fontsize=10.5, fontweight="bold", pad=26, loc="left"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.subplots_adjust(left=0.26, right=0.49, top=0.88, bottom=0.12)

    fig1_path = os.path.join(FIG_DIR, "fig1_cobenfy_polypharmacy_forest_plot.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig1_path}")

    # --- FIGURE 2: Three-panel visualization derived 100% from analysis_summary.json ---
    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20.0, 6.8), dpi=300, gridspec_kw={"width_ratios": [1.05, 1.32, 1.12]})

    # Panel A: Specific Concomitant Agents
    spec = summary["specific_counts"]
    sorted_drugs = sorted(spec.items(), key=lambda kv: kv[1], reverse=True)
    d_names = [k for k, _ in sorted_drugs][::-1]
    d_vals = [v for _, v in sorted_drugs][::-1]
    d_cols = ["#B22222" if ("ACB 3" in n or "Glycopyrrolate" in n or "Meclizine" in n) else "#2B5B84" for n in d_names]

    bars = ax1.barh(d_names, d_vals, color=d_cols, height=0.72, edgecolor="#222222", linewidth=0.5)
    for bar, val in zip(bars, d_vals):
        ax1.text(val + 1.5, bar.get_y() + bar.get_height() / 2, f"n={val}", va="center", fontsize=8.0, fontweight="bold")
    ax1.set_xlim(0, max(d_vals) + 18)
    ax1.set_xlabel("Unique FAERS Cases Co-Prescribed", fontsize=9.5, fontweight="bold")
    ax1.set_title("A. Concomitant Agents (Red = ACB-3; Blue = Non-Anti-M AP)", fontsize=9.8, fontweight="bold", pad=10)
    ax1.tick_params(axis="y", labelsize=8.0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B: 3-Stratum Event Rates dynamically from JSON
    by_label = {r["label"]: r for r in results}
    panel_b_specs = [
        ("Hospitalization (All-Cause)", "All Hosp."),
        ("Non-Psychiatric Medical Hospitalization", "Med. Hosp."),
        ("GI Hypomotility (Constipation / Ileus / Obstruction)", "GI Hypomot."),
        ("Constipation (Single MedDRA PT)", "Constipation"),
        ("Gastroesophageal Reflux Disease (GERD)", "GERD"),
        ("Urinary Retention (Nominal; Attenuates in HCP)", "Urinary Ret."),
    ]
    categories, ach_rates, d2_rates, mono_rates = [], [], [], []
    for full_lab, short_lab in panel_b_specs:
        row = by_label[full_lab]
        p_val = row["multivariable_logit"]["s1_p"]
        p_str = "p<0.001*" if p_val < 0.001 else (f"p={p_val:.3f}*" if p_val < 0.05 else f"p={p_val:.2f}")
        categories.append(f"{short_lab}\n({p_str})")
        ach_rates.append(row["ach_vs_mono"]["pct1"])
        d2_rates.append(row["d2_vs_mono"]["pct1"])
        mono_rates.append(row["ach_vs_mono"]["pct0"])

    x = np.arange(len(categories))
    width = 0.25
    b1 = ax2.bar(x - width, ach_rates, width, label=f"Strong ACB-3 Antimuscarinics (n = {n_s1})", color="#B22222", edgecolor="#222222", linewidth=0.6)
    b2 = ax2.bar(x, d2_rates, width, label=f"Non-Antimuscarinic APs Only (n = {n_s2})", color="#D95F02", edgecolor="#222222", linewidth=0.6)
    b3 = ax2.bar(x + width, mono_rates, width, label=f"Unexposed Reference (n = {n_mono:,})", color="#2B5B84", edgecolor="#222222", linewidth=0.6)

    for bars_grp, col in [(b1, "#B22222"), (b2, "#D95F02"), (b3, "#2B5B84")]:
        for b in bars_grp:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2, h + 0.3, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color=col)

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=7.8, fontweight="bold")
    ax2.set_ylabel("Reporting Frequency (% of Stratum)", fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, max(ach_rates) + 4.0)
    ax2.set_title("B. 3-Stratum Event Rates (Strong ACB-3 vs. D2 AP vs. Reference)", fontsize=9.8, fontweight="bold", pad=10)
    ax2.legend(frameon=False, fontsize=8.2, loc="upper right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Panel C: 4-Model Sensitivity Robustness Check across Core Endpoints
    robust_specs = [
        ("Hospitalization (All-Cause)", "All Hosp."),
        ("GI Hypomotility (Constipation / Ileus / Obstruction)", "GI Hypomot."),
        ("Constipation (Single MedDRA PT)", "Constipation"),
        ("Gastroesophageal Reflux Disease (GERD)", "GERD"),
        ("Urinary Retention (Nominal; Attenuates in HCP)", "Urinary Ret."),
    ]
    rx = np.arange(len(robust_specs))
    rw = 0.19
    m_crude = [by_label[k]["ach_vs_mono"]["ror"] for k, _ in robust_specs]
    m_aror = [by_label[k]["multivariable_logit"]["s1_aror"] for k, _ in robust_specs]
    m_hcp = [by_label[k]["hcp_ach_vs_mono"]["ror"] for k, _ in robust_specs]
    m_nocloz = [by_label[k]["nocloz_ach_vs_mono"]["ror"] for k, _ in robust_specs]

    ax3.axhline(1.0, color="#222222", linestyle="--", linewidth=1.1)
    rc1 = ax3.bar(rx - 1.5 * rw, m_crude, rw, label="Crude ROR (n=260)", color="#888888", edgecolor="#222222", linewidth=0.5)
    rc2 = ax3.bar(rx - 0.5 * rw, m_aror, rw, label="Reporter-Adjusted aROR", color="#B22222", edgecolor="#222222", linewidth=0.5)
    rc3 = ax3.bar(rx + 0.5 * rw, m_hcp, rw, label=f"HCP-Only (n={n_hcp_s1})", color="#1F4E78", edgecolor="#222222", linewidth=0.5)
    rc4 = ax3.bar(rx + 1.5 * rw, m_nocloz, rw, label=f"No Clozapine (n={n_nocloz})", color="#D95F02", edgecolor="#222222", linewidth=0.5)

    for bars_grp, col in [(rc1, "#555555"), (rc2, "#B22222"), (rc3, "#1F4E78"), (rc4, "#D95F02")]:
        for b in bars_grp:
            h = b.get_height()
            ax3.text(b.get_x() + b.get_width() / 2, h + 0.08, f"{h:.2f}", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color=col, rotation=90)

    ax3.set_xticks(rx)
    ax3.set_xticklabels([s for _, s in robust_specs], fontsize=8.4, fontweight="bold")
    ax3.set_ylabel("Reporting Odds Ratio (ROR / aROR)", fontsize=9.5, fontweight="bold")
    ax3.set_ylim(0, 4.7)
    ax3.set_title("C. Sensitivity Robustness Across 4 Models", fontsize=9.8, fontweight="bold", pad=10)
    ax3.legend(frameon=False, fontsize=8.0, loc="upper left")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    plt.tight_layout(w_pad=2.4)
    fig2_path = os.path.join(FIG_DIR, "fig2_cobenfy_concomitant_breakdown.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {fig2_path}")


if __name__ == "__main__":
    main()
