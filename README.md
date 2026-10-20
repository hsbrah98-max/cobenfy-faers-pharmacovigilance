# Cobenfy (Xanomeline-Trospium) Polypharmacy FAERS Study

**Author:** Harman Brah, MD

Complete, self-contained dataset, code, figures, and manuscript package evaluating disproportionate adverse event reporting with concomitant strong antimuscarinics vs. non-antimuscarinic antipsychotics across 2,148 deduplicated U.S. FDA FAERS reports for xanomeline-trospium.

## Folder Contents

| File | Description |
| :--- | :--- |
| [audit_checks.py](audit_checks.py) | Re-derives every statistical claim directly from raw openFDA data; output saved to `data/audit_output.txt` |
| [APA_2027_Cobenfy_Polypharmacy_Study.md](APA_2027_Cobenfy_Polypharmacy_Study.md) | APA 2027 abstract, figures, core results table (Stratum 1 and Stratum 2 vs. Reference), exploratory LAI vs. oral route, negative control outcomes, READUS-PV scope, and references |
| [SUPPLEMENT_AND_DEMOGRAPHICS.md](SUPPLEMENT_AND_DEMOGRAPHICS.md) | 15-endpoint sensitivity matrix, negative controls, LAI exploratory table, sex/age subgroups |
| [fetch_and_analyze.py](fetch_and_analyze.py) | Downloads/loads all 2,148 deduplicated FDA FAERS records, runs the 3-stratum analysis with 4 sensitivity models, prespecified negative controls, and an exploratory LAI vs. oral comparison (LAI from reporter-entered product names only), and exports JSON + CSV tables |
| [generate_figures.py](generate_figures.py) | Generates 300-DPI PNGs of Figure 1 (forest plot with negative control row) and Figure 2 (concomitant agents and 3-stratum event rates) |
| [data/cobenfy_faers_raw.json](data/cobenfy_faers_raw.json) | Complete raw U.S. FDA OpenFDA dataset (`N = 2,148` unique `safetyreportid` records through 2026 Q2) |
| [data/analysis_summary.json](data/analysis_summary.json) | Machine-readable statistical output across all strata and sensitivity analyses |
| [data/table1_concomitant_drugs.csv](data/table1_concomitant_drugs.csv) | CSV export of concomitant antipsychotics, antihistamines, and strong anticholinergics |
| [data/table2_three_stratum_disproportionality.csv](data/table2_three_stratum_disproportionality.csv) | CSV export of Stratum 1, Stratum 2, and Stratum 1 vs. 2 results (crude and adjusted) for all endpoints and negative controls |
| [data/table3_sensitivity_controls.csv](data/table3_sensitivity_controls.csv) | CSV export of HCP-only, non-clozapine, active-comparator, and subgroup (route, sex, age) analyses |
| [figures/fig1_cobenfy_polypharmacy_forest_plot.png](figures/fig1_cobenfy_polypharmacy_forest_plot.png) | 300-DPI forest plot of reporter-adjusted aROR, with negative control composite |
| [figures/fig2_cobenfy_concomitant_breakdown.png](figures/fig2_cobenfy_concomitant_breakdown.png) | 300-DPI two-panel concomitant drug landscape and 3-stratum event rates |

## How to Re-Run from Terminal

```bash
python3 fetch_and_analyze.py
python3 generate_figures.py
python3 audit_checks.py > data/audit_output.txt
```
