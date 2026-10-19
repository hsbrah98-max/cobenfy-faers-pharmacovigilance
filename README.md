# Cobenfy (Xanomeline-Trospium) Polypharmacy FAERS Study

**Author:** Harman Brah, MD

Complete, self-contained dataset, code, figures, and manuscript package evaluating muscarinic vs. dopaminergic adverse event disproportionality across 2,148 deduplicated U.S. FDA FAERS reports for xanomeline-trospium.

## Folder Contents

| File | Description |
| :--- | :--- |
| [audit_checks.py](audit_checks.py) | Re-derives every statistical claim directly from raw openFDA data; output saved to `data/audit_output.txt` |
| [APA_2027_Cobenfy_Polypharmacy_Study.md](APA_2027_Cobenfy_Polypharmacy_Study.md) | Full Background, Introduction, Methods, Results (Tables 1 to 3), Discussion, and References |
| [fetch_and_analyze.py](fetch_and_analyze.py) | Downloads/loads all 2,148 deduplicated FDA FAERS records, runs 3-stratum and 4 sensitivity analyses, and exports JSON + CSV tables |
| [generate_figures.py](generate_figures.py) | Generates 300-DPI PNG and vector PDF versions of Figure 1 (Forest Plot) and Figure 2 (Concomitant Breakdown & 3-Stratum Dissociation) |
| [data/cobenfy_faers_raw.json](data/cobenfy_faers_raw.json) | Complete raw U.S. FDA OpenFDA dataset (`N = 2,148` unique `safetyreportid` records through 2026 Q2) |
| [data/analysis_summary.json](data/analysis_summary.json) | Machine-readable statistical output across all strata and sensitivity analyses |
| [data/table1_concomitant_drugs.csv](data/table1_concomitant_drugs.csv) | CSV export of concomitant antipsychotics, antihistamines, and strong anticholinergics |
| [data/table2_three_stratum_disproportionality.csv](data/table2_three_stratum_disproportionality.csv) | CSV export of 3-stratum disproportionality results (Stratum 1, Stratum 2, All Polypharmacy vs. Pure Monotherapy) |
| [data/table3_sensitivity_controls.csv](data/table3_sensitivity_controls.csv) | CSV export of the 4 illness-severity and polypharmacy sensitivity controls |
| [figures/fig1_cobenfy_polypharmacy_forest_plot.png](figures/fig1_cobenfy_polypharmacy_forest_plot.png) | 300-DPI Multi-Domain Forest Plot (also saved as `.pdf`) |
| [figures/fig2_cobenfy_concomitant_breakdown.png](figures/fig2_cobenfy_concomitant_breakdown.png) | 300-DPI Two-Panel Concomitant Drug Landscape & 3-Stratum Dissociation (also saved as `.pdf`) |

## How to Re-Run from Terminal

```bash
python3 fetch_and_analyze.py
python3 generate_figures.py
python3 audit_checks.py
```
