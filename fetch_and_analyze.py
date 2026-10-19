#!/usr/bin/env python3
"""
READUS-PV Compliant Disproportionality & Clinical Subgroup Analysis of
Xanomeline-Trospium (Cobenfy) in the U.S. FDA Adverse Event Reporting System
(FAERS, N = 2,148 unique safety reports through Q2 2026).

Evaluates three clinician-relevant questions with full sensitivity checks:
  1. Concomitant Antimuscarinic Burden (ACB Score = 3, n = 260) vs.
     Non-Antimuscarinic D2/5-HT2A Antipsychotics Only (n = 164) vs.
     Unexposed Reference (n = 1,724) across 4 models:
       - Model 1: Crude Reporting Odds Ratio (ROR, 95% CI)
       - Model 2: Multivariable Logistic Regression Adjusted ROR (aROR, 95% CI)
                  controlling for Healthcare Professional (HCP) vs. Consumer status
       - Model 3: HCP-Only Restricted Cohort (n = 222 vs. n = 885)
       - Model 4: Non-Clozapine ACB-3 Cohort (n = 162 vs. n = 1,724)
  2. Antipsychotic Administration Route: Long-Acting Injectable (LAI) Antipsychotics
     (n = 119) vs. Oral Antipsychotics Only (n = 284) co-prescribed with Cobenfy
     (overall and HCP-only sensitivity, n = 101 vs. n = 233).
  3. Demographic Risk Stratification:
     - Sex Differences (n = 1,646: 678 Female vs. 968 Male; HCP-only n = 330 vs. 505)
     - Age & Older Male Vulnerability (n = 1,076 adults: 143 aged >=60 vs. 933 aged 18-59;
       57 Older Men aged >=60 vs. 986 Other Adults; plus HCP-only sensitivity).
"""
import json
import math
import os
import time
import urllib.parse
import urllib.request
from collections import Counter
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
RAW_PATH = os.path.join(DATA_DIR, "cobenfy_faers_raw.json")
SUMMARY_PATH = os.path.join(DATA_DIR, "analysis_summary.json")

QUERY = 'patient.drug.medicinalproduct:("COBENFY" OR "XANOMELINE")'


def fetch_all_records():
    if os.path.exists(RAW_PATH):
        with open(RAW_PATH, "r") as f:
            cached = json.load(f)
        if len(cached.get("records", [])) >= 2140:
            print(f"Loaded {len(cached['records'])} cached records (last_updated={cached.get('last_updated')})")
            return cached["records"], cached.get("last_updated")

    q_enc = urllib.parse.quote(QUERY)
    first_url = f"https://api.fda.gov/drug/event.json?search={q_enc}&limit=1"
    with urllib.request.urlopen(first_url, timeout=15) as r:
        meta = json.loads(r.read().decode())["meta"]
    total = meta["results"]["total"]
    last_updated = meta["last_updated"]
    records = []
    for skip in range(0, total, 100):
        url = f"https://api.fda.gov/drug/event.json?search={q_enc}&limit=100&skip={skip}"
        with urllib.request.urlopen(url, timeout=20) as r:
            records.extend(json.loads(r.read().decode())["results"])
        time.sleep(0.15)
    with open(RAW_PATH, "w") as f:
        json.dump({"last_updated": last_updated, "total": total, "records": records}, f)
    return records, last_updated


def fisher_exact_two_sided(a, b, c, d):
    n = a + b + c + d
    r1, r2, c1, c2 = a + b, c + d, a + c, b + d
    if min(r1, r2, c1, c2) == 0:
        return 1.0

    def log_hg(x):
        return (
            math.lgamma(r1 + 1) + math.lgamma(r2 + 1) + math.lgamma(c1 + 1) + math.lgamma(c2 + 1)
            - math.lgamma(n + 1) - math.lgamma(x + 1) - math.lgamma(r1 - x + 1)
            - math.lgamma(c1 - x + 1) - math.lgamma(r2 - c1 + x + 1)
        )

    p_obs = math.exp(log_hg(a))
    p_val = 0.0
    for x in range(max(0, r1 + c1 - n), min(r1, c1) + 1):
        px = math.exp(log_hg(x))
        if px <= p_obs * (1 + 1e-7):
            p_val += px
    return min(1.0, p_val)


def norm_two_sided_p(z):
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def fit_logistic_aror(cases, fn):
    """Fit multivariable logistic regression: Y ~ 1 + S1 + S2 + HCP."""
    X = np.column_stack([
        np.ones(len(cases)),
        [1.0 if c["stratum"] == "S1" else 0.0 for c in cases],
        [1.0 if c["stratum"] == "S2" else 0.0 for c in cases],
        [1.0 if c["hcp"] else 0.0 for c in cases],
    ])
    y = np.array([1.0 if fn(c) else 0.0 for c in cases])
    beta = np.zeros(4)
    for _ in range(30):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))
        w = p * (1.0 - p) + 1e-9
        grad = X.T @ (y - p)
        hess = (X.T * w) @ X
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-7:
            break
    cov = np.linalg.inv(hess)
    se = np.sqrt(np.diag(cov))
    aror_s1 = math.exp(beta[1])
    ci_l_s1, ci_u_s1 = math.exp(beta[1] - 1.96 * se[1]), math.exp(beta[1] + 1.96 * se[1])
    p_s1 = norm_two_sided_p(beta[1] / se[1])
    aror_s2 = math.exp(beta[2])
    ci_l_s2, ci_u_s2 = math.exp(beta[2] - 1.96 * se[2]), math.exp(beta[2] + 1.96 * se[2])
    p_s2 = norm_two_sided_p(beta[2] / se[2])
    return {
        "s1_aror": round(aror_s1, 2), "s1_ci_l": round(ci_l_s1, 2), "s1_ci_u": round(ci_u_s1, 2), "s1_p": round(p_s1, 4),
        "s2_aror": round(aror_s2, 2), "s2_ci_l": round(ci_l_s2, 2), "s2_ci_u": round(ci_u_s2, 2), "s2_p": round(p_s2, 4),
    }


def bh_adjust(p_vals):
    m = len(p_vals)
    order = sorted(range(m), key=lambda i: p_vals[i])
    q_vals = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        prev = min(prev, p_vals[idx] * m / rank)
        q_vals[idx] = round(prev, 4)
    return q_vals


def compute_2x2(grp1, grp0, fn):
    a = sum(1 for x in grp1 if fn(x))
    b = len(grp1) - a
    c = sum(1 for x in grp0 if fn(x))
    d = len(grp0) - c
    p_val = fisher_exact_two_sided(a, b, c, d)
    if min(a, b, c, d) == 0:
        return {
            "a": a, "n1": len(grp1), "pct1": round(a / len(grp1) * 100, 2) if grp1 else 0.0,
            "c": c, "n0": len(grp0), "pct0": round(c / len(grp0) * 100, 2) if grp0 else 0.0,
            "ror": 0.0 if a == 0 else None, "ci_l": 0.0 if a == 0 else None, "ci_u": 0.0 if a == 0 else None,
            "p_val": round(p_val, 4),
        }
    ror = (a * d) / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    ci_l, ci_u = math.exp(math.log(ror) - 1.96 * se), math.exp(math.log(ror) + 1.96 * se)
    return {
        "a": a, "n1": len(grp1), "pct1": round(a / len(grp1) * 100, 2),
        "c": c, "n0": len(grp0), "pct0": round(c / len(grp0) * 100, 2),
        "ror": round(ror, 2), "ci_l": round(ci_l, 2), "ci_u": round(ci_u, 2),
        "p_val": round(p_val, 4),
    }


def main():
    records, last_updated = fetch_all_records()
    unique_map = {r["safetyreportid"]: r for r in records}

    strong_antimuscarinic_aps = {
        "CLOZAPINE", "CLOZARIL", "VERSACLOZ",
        "OLANZAPINE", "ZYPREXA", "LYBALVI",
        "QUETIAPINE", "SEROQUEL",
        "CHLORPROMAZINE", "THORAZINE",
        "LOXAPINE", "LOXEPIN", "ADASUVE",
        "THIORIDAZINE",
    }
    non_antimuscarinic_aps = {
        "RISPERIDONE", "RISPERDAL", "PERSERIS", "UZEDY",
        "PALIPERIDONE", "INVEGA",
        "ARIPIPRAZOLE", "ABILIFY", "ARISTADA",
        "HALOPERIDOL", "HALDOL",
        "LUMATEPERONE", "CAPLYTA",
        "CARIPRAZINE", "VRAYLAR",
        "LURASIDONE", "LATUDA",
        "BREXPIPRAZOLE", "REXULTI",
        "ZIPRASIDONE", "GEODON",
        "ILOPERIDONE", "FANAPT",
        "PIMAVANSERIN", "NUPLAZID",
        "ASENAPINE", "SAPHRIS",
        "FLUPHENAZINE", "PROLIXIN",
        "PERPHENAZINE", "TRILAFON",
    }
    first_gen_antihistamines = {
        "HYDROXYZINE", "VISTARIL", "ATARAX",
        "DIPHENHYDRAMINE", "BENADRYL",
        "MECLIZINE", "ANTIVERT",
        "PROMETHAZINE", "PHENERGAN",
        "CHLORPHENIRAMINE", "DOXYLAMINE",
        "CYPROHEPTADINE", "DIMENHYDRINATE",
    }
    systemic_anticholinergics = {
        "BENZTROPINE", "COGENTIN",
        "TRIHEXYPHENIDYL", "ARTANE",
        "GLYCOPYRROLATE", "ROBINUL",
        "ATROPINE", "SCOPOLAMINE",
        "OXYBUTYNIN", "DITROPAN", "SOLIFENACIN", "VESICARE", "TOLTERODINE", "DETROL",
        "DICYCLOMINE", "BENTYL", "HYOSCYAMINE",
        "PAROXETINE", "PAXIL", "AMITRIPTYLINE", "NORTRIPTYLINE", "DOXEPIN",
        "AMOXAPINE", "IMIPRAMINE", "CLOMIPRAMINE",
    }
    lai_keywords = {
        "SUSTENNA", "TRINZA", "HAFYERA", "MAINTENA", "ASIMTUFII", "ARISTADA",
        "LAUROXIL", "UZEDY", "PERSERIS", "CONSTA", "PALMITATE", "DECANOATE", "RELPREVV",
    }
    all_ap_keywords = strong_antimuscarinic_aps | non_antimuscarinic_aps

    psych_relapse_pts = {
        "PSYCHOTIC DISORDER", "SCHIZOPHRENIA", "CONDITION AGGRAVATED", "PSYCHOTIC SYMPTOM",
        "HALLUCINATION", "HALLUCINATION, AUDITORY", "DELUSION", "PARANOIA", "AGITATION",
    }
    gi_hypomotility_pts = {
        "CONSTIPATION", "INTESTINAL OBSTRUCTION", "ILEUS", "ILEUS PARALYTIC",
        "GASTROINTESTINAL HYPOMOTILITY", "FAECALOMA",
    }

    g_ach_strong = []
    g_d2_only = []
    g_all_poly = []
    g_pure_mono = []

    specific_counts = Counter()
    class_counts = Counter()

    track_drugs = {
        "Clozapine (ACB 3 M1-M5 AP)": {"CLOZAPINE", "CLOZARIL", "VERSACLOZ"},
        "Olanzapine / Lybalvi (ACB 3 M1-M5 AP)": {"OLANZAPINE", "ZYPREXA", "LYBALVI"},
        "Risperidone (D2/5-HT2A AP)": {"RISPERIDONE", "RISPERDAL", "PERSERIS", "UZEDY"},
        "Aripiprazole (D2 Partial Agonist AP)": {"ARIPIPRAZOLE", "ABILIFY", "ARISTADA"},
        "Paliperidone (D2/5-HT2A AP)": {"PALIPERIDONE", "INVEGA"},
        "Haloperidol (D2 Antagonist AP)": {"HALOPERIDOL", "HALDOL"},
        "Quetiapine (ACB 3 Antimuscarinic AP)": {"QUETIAPINE", "SEROQUEL"},
        "Benztropine (ACB 3 Central M1/M4 Blocker)": {"BENZTROPINE", "COGENTIN"},
        "Hydroxyzine (ACB 3 1st-Gen H1/M1 Blocker)": {"HYDROXYZINE", "VISTARIL", "ATARAX"},
        "Lumateperone (D2/5-HT2A AP)": {"LUMATEPERONE", "CAPLYTA"},
        "Cariprazine (D2/D3 Partial Agonist AP)": {"CARIPRAZINE", "VRAYLAR"},
        "Brexpiprazole / Ziprasidone / Iloperidone (D2/5-HT2A AP)": {
            "BREXPIPRAZOLE", "REXULTI", "ZIPRASIDONE", "GEODON", "ILOPERIDONE", "FANAPT",
        },
        "Fluphenazine / Perphenazine (Typical D2 AP)": {"FLUPHENAZINE", "PROLIXIN", "PERPHENAZINE", "TRILAFON"},
        "Lurasidone (D2/5-HT2A AP)": {"LURASIDONE", "LATUDA"},
        "Atropine / Glycopyrrolate / Oxybutynin / Scopolamine": {
            "ATROPINE", "GLYCOPYRROLATE", "ROBINUL", "OXYBUTYNIN", "SCOPOLAMINE",
            "DICYCLOMINE", "BENTYL", "HYOSCYAMINE", "SOLIFENACIN", "TOLTERODINE",
        },
        "Chlorpromazine / Loxapine (ACB 3 Antimuscarinic AP)": {
            "CHLORPROMAZINE", "THORAZINE", "LOXAPINE", "LOXEPIN", "ADASUVE", "THIORIDAZINE",
        },
        "Pimavanserin / Asenapine (5-HT2A / Non-Antimuscarinic AP)": {
            "PIMAVANSERIN", "NUPLAZID", "ASENAPINE", "SAPHRIS",
        },
        "Diphenhydramine (ACB 3 1st-Gen H1/M1 Blocker)": {"DIPHENHYDRAMINE", "BENADRYL"},
        "Paroxetine / Amitriptyline / Doxepin / Amoxapine (ACB 3)": {
            "PAROXETINE", "PAXIL", "AMITRIPTYLINE", "NORTRIPTYLINE", "DOXEPIN", "AMOXAPINE",
        },
        "Meclizine / Promethazine / Chlorpheniramine / Doxylamine": {
            "MECLIZINE", "ANTIVERT", "PROMETHAZINE", "PHENERGAN", "CHLORPHENIRAMINE", "DOXYLAMINE", "CYPROHEPTADINE",
        },
    }

    all_items = []
    for sid, rec in unique_map.items():
        p = rec.get("patient", {})
        dnames = set()
        for d in p.get("drug", []):
            mp = (d.get("medicinalproduct") or "").upper()
            if mp:
                dnames.add(mp)
            for g in d.get("openfda", {}).get("generic_name", []):
                dnames.add(g.upper())
            for b in d.get("openfda", {}).get("brand_name", []):
                dnames.add(b.upper())

        has_m_ap = any(any(k in dn for k in strong_antimuscarinic_aps) for dn in dnames)
        has_o_ap = any(any(k in dn for k in non_antimuscarinic_aps) for dn in dnames)
        has_ah = any(any(k in dn for k in first_gen_antihistamines) for dn in dnames)
        has_ach = any(any(k in dn for k in systemic_anticholinergics) for dn in dnames)
        has_lai = any(any(k in dn for k in lai_keywords) for dn in dnames)
        has_any_ap = any(any(k in dn for k in all_ap_keywords) for dn in dnames)

        if has_m_ap:
            class_counts["Strong Antimuscarinic Antipsychotics (ACB 3)"] += 1
        if has_o_ap:
            class_counts["Non-Antimuscarinic D2/5-HT2A Antipsychotics"] += 1
        if has_ah:
            class_counts["1st-Gen Antimuscarinic Antihistamines (ACB 3)"] += 1
        if has_ach:
            class_counts["Systemic Antimuscarinics & ACB-3 Antidepressants"] += 1
        if has_lai:
            class_counts["Long-Acting Injectable (LAI) Antipsychotics"] += 1

        for label, kw_set in track_drugs.items():
            if any(any(k in dn for k in kw_set) for dn in dnames):
                specific_counts[label] += 1

        pts = {(rx.get("reactionmeddrapt") or "").upper() for rx in p.get("reaction", []) if rx.get("reactionmeddrapt")}
        hosp = str(rec.get("seriousnesshospitalization", "")) == "1"
        ps = rec.get("primarysource") or {}
        hcp = str(ps.get("qualification", "")) in ("1", "2", "3")
        cloz = any("CLOZ" in dn for dn in dnames)
        age = float(p["patientonsetage"]) if p.get("patientonsetage") and str(p.get("patientonsetageunit")) == "801" else None
        sex = str(p.get("patientsex", ""))

        if has_m_ap or has_ah or has_ach:
            stratum = "S1"
        elif has_o_ap:
            stratum = "S2"
        else:
            stratum = "MONO"

        item = {
            "sid": sid, "pts": pts, "hosp": hosp, "hcp": hcp,
            "clozapine": cloz, "stratum": stratum,
            "age": age, "sex": sex,
            "lai": has_lai, "oral_ap_only": has_any_ap and not has_lai,
            "psych_relapse": bool(pts & psych_relapse_pts),
            "nonpsych_hosp": hosp and not bool(pts & psych_relapse_pts),
            "gi_hypo": bool(pts & gi_hypomotility_pts),
        }
        all_items.append(item)

        if stratum == "S1":
            g_ach_strong.append(item)
            g_all_poly.append(item)
        elif stratum == "S2":
            g_d2_only.append(item)
            g_all_poly.append(item)
        else:
            g_pure_mono.append(item)

    endpoints = [
        ("Hospitalization (All-Cause)", lambda x: x["hosp"]),
        ("Non-Psychiatric Medical Hospitalization", lambda x: x["nonpsych_hosp"]),
        ("GI Hypomotility (Constipation / Ileus / Obstruction)", lambda x: x["gi_hypo"]),
        ("Constipation (Single MedDRA PT)", lambda x: "CONSTIPATION" in x["pts"]),
        ("Gastroesophageal Reflux Disease (GERD)", lambda x: "GASTROOESOPHAGEAL REFLUX DISEASE" in x["pts"]),
        ("Urinary Retention (Nominal; Attenuates in HCP)", lambda x: "URINARY RETENTION" in x["pts"]),
        ("Blurred Vision", lambda x: "VISION BLURRED" in x["pts"]),
        ("Tachycardia / Heart Rate Increased", lambda x: bool(x["pts"] & {"TACHYCARDIA", "HEART RATE INCREASED"})),
        ("Akathisia (Concomitant D2 Burden)", lambda x: "AKATHISIA" in x["pts"]),
        ("Sialorrhea + Drooling (Combined MedDRA PTs)", lambda x: bool(x["pts"] & {"SALIVARY HYPERSECRETION", "DROOLING"})),
        ("Salivary Hypersecretion (Clinician PT Only)", lambda x: "SALIVARY HYPERSECRETION" in x["pts"]),
        ("Weight Increased", lambda x: "WEIGHT INCREASED" in x["pts"]),
        ("Nausea", lambda x: "NAUSEA" in x["pts"]),
        ("Vomiting (Nominal; Attenuates in HCP)", lambda x: "VOMITING" in x["pts"]),
        ("Dyspepsia", lambda x: "DYSPEPSIA" in x["pts"]),
    ]

    hcp_s1 = [x for x in g_ach_strong if x["hcp"]]
    hcp_s2 = [x for x in g_d2_only if x["hcp"]]
    hcp_mono = [x for x in g_pure_mono if x["hcp"]]
    nocloz_s1 = [x for x in g_ach_strong if not x["clozapine"]]

    results = []
    for label, fn in endpoints:
        r_ach = compute_2x2(g_ach_strong, g_pure_mono, fn)
        r_d2 = compute_2x2(g_d2_only, g_pure_mono, fn)
        r_all = compute_2x2(g_all_poly, g_pure_mono, fn)
        r_h2h = compute_2x2(g_ach_strong, g_d2_only, fn)
        r_hcp_ach = compute_2x2(hcp_s1, hcp_mono, fn)
        r_nocloz = compute_2x2(nocloz_s1, g_pure_mono, fn)
        logit = fit_logistic_aror(all_items, fn)
        results.append({
            "label": label,
            "ach_vs_mono": r_ach,
            "d2_vs_mono": r_d2,
            "all_poly_vs_mono": r_all,
            "s1_vs_s2_h2h": r_h2h,
            "hcp_ach_vs_mono": r_hcp_ach,
            "nocloz_ach_vs_mono": r_nocloz,
            "multivariable_logit": logit,
        })

    q_s1 = bh_adjust([r["ach_vs_mono"]["p_val"] for r in results])
    q_aror = bh_adjust([r["multivariable_logit"]["s1_p"] for r in results])
    for r, q1, qa in zip(results, q_s1, q_aror):
        r["ach_vs_mono"]["bh_q"] = q1
        r["multivariable_logit"]["s1_bh_q"] = qa

    # Clinical Subgroup Analyses (LAI vs Oral AP, Female vs Male, Age >=60 & Older Men >=60)
    lai_all = [x for x in all_items if x["lai"]]
    oral_ap_all = [x for x in all_items if x["oral_ap_only"]]
    lai_hcp = [x for x in lai_all if x["hcp"]]
    oral_ap_hcp = [x for x in oral_ap_all if x["hcp"]]

    fem_all = [x for x in all_items if x["sex"] == "2"]
    mal_all = [x for x in all_items if x["sex"] == "1"]
    fem_hcp = [x for x in fem_all if x["hcp"]]
    mal_hcp = [x for x in mal_all if x["hcp"]]

    old_all = [x for x in all_items if x["age"] is not None and x["age"] >= 60]
    yng_all = [x for x in all_items if x["age"] is not None and 18 <= x["age"] < 60]
    old_hcp = [x for x in old_all if x["hcp"]]
    yng_hcp = [x for x in yng_all if x["hcp"]]

    old_men_all = [x for x in all_items if x["age"] is not None and x["age"] >= 60 and x["sex"] == "1"]
    oth_adult_all = [x for x in all_items if x["age"] is not None and x["age"] >= 18 and x["sex"] in ("1", "2") and not (x["age"] >= 60 and x["sex"] == "1")]
    old_men_hcp = [x for x in old_men_all if x["hcp"]]
    oth_adult_hcp = [x for x in oth_adult_all if x["hcp"]]

    clinical_subgroups = {
        "lai_vs_oral_ap_gi_hypo": {
            "overall": compute_2x2(lai_all, oral_ap_all, lambda x: x["gi_hypo"]),
            "hcp_only": compute_2x2(lai_hcp, oral_ap_hcp, lambda x: x["gi_hypo"]),
        },
        "oral_ap_vs_lai_gi_hypo": {
            "overall": compute_2x2(oral_ap_all, lai_all, lambda x: x["gi_hypo"]),
            "hcp_only": compute_2x2(oral_ap_hcp, lai_hcp, lambda x: x["gi_hypo"]),
        },
        "lai_vs_oral_ap_gerd": {
            "overall": compute_2x2(lai_all, oral_ap_all, lambda x: "GASTROOESOPHAGEAL REFLUX DISEASE" in x["pts"]),
            "hcp_only": compute_2x2(lai_hcp, oral_ap_hcp, lambda x: "GASTROOESOPHAGEAL REFLUX DISEASE" in x["pts"]),
        },
        "lai_vs_oral_ap_hosp": {
            "overall": compute_2x2(lai_all, oral_ap_all, lambda x: x["hosp"]),
            "hcp_only": compute_2x2(lai_hcp, oral_ap_hcp, lambda x: x["hosp"]),
        },
        "female_vs_male_nausea": {
            "overall": compute_2x2(fem_all, mal_all, lambda x: "NAUSEA" in x["pts"]),
            "hcp_only": compute_2x2(fem_hcp, mal_hcp, lambda x: "NAUSEA" in x["pts"]),
        },
        "female_vs_male_vomiting": {
            "overall": compute_2x2(fem_all, mal_all, lambda x: "VOMITING" in x["pts"]),
            "hcp_only": compute_2x2(fem_hcp, mal_hcp, lambda x: "VOMITING" in x["pts"]),
        },
        "age_60_vs_18_59_uret": {
            "overall": compute_2x2(old_all, yng_all, lambda x: "URINARY RETENTION" in x["pts"]),
            "hcp_only": compute_2x2(old_hcp, yng_hcp, lambda x: "URINARY RETENTION" in x["pts"]),
        },
        "older_men_60_vs_others_uret": {
            "overall": compute_2x2(old_men_all, oth_adult_all, lambda x: "URINARY RETENTION" in x["pts"]),
            "hcp_only": compute_2x2(old_men_hcp, oth_adult_hcp, lambda x: "URINARY RETENTION" in x["pts"]),
        },
    }

    summary = {
        "last_updated": last_updated,
        "n_total": len(unique_map),
        "n_all_poly": len(g_all_poly),
        "n_ach_ah": len(g_ach_strong),
        "n_d2_only": len(g_d2_only),
        "n_pure_mono": len(g_pure_mono),
        "n_hcp_s1": len(hcp_s1),
        "n_hcp_s2": len(hcp_s2),
        "n_hcp_mono": len(hcp_mono),
        "n_nocloz_s1": len(nocloz_s1),
        "n_lai": len(lai_all),
        "n_oral_ap": len(oral_ap_all),
        "n_female": len(fem_all),
        "n_male": len(mal_all),
        "n_age_60": len(old_all),
        "n_age_18_59": len(yng_all),
        "n_older_men_60": len(old_men_all),
        "n_other_adults": len(oth_adult_all),
        "class_counts": dict(class_counts),
        "specific_counts": dict(specific_counts.most_common(30)),
        "results": results,
        "clinical_subgroups": clinical_subgroups,
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved combined READUS-PV & clinical subgroup summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
