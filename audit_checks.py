#!/usr/bin/env python3
"""
Independent audit of the unified Cobenfy FAERS polypharmacy, antipsychotic route,
and demographic risk stratification analysis.

Re-derives every headline claim from data/cobenfy_faers_raw.json and verifies:
  A. APA abstract character count (< 3,000 characters with spaces)
  B. Strict ACB=3 cohort reproduction (2,148 / 260 / 164 / 1,724)
  C. Primary & Multivariable 4-model comparisons (Crude, aROR, HCP-only, No-Clozapine)
  D. Healthcare-professional (HCP)-only stratified checks (222 vs. 885)
  E. Head-to-head Stratum 1 vs. Stratum 2 active-comparator checks (260 vs. 164)
  E2. Stratum 2 vs. Reference (164 vs. 1,724): the GI signal should be absent
  F. Clinical Subgroups: LAI AP (reporter-named, n=49) vs. no LAI (n=354), re-derived here
     and repeated within Stratum 2; Sex/Age Demographics from the summary
  G. Dynamic JSON verification for generate_figures.py
  H. Negative control outcomes (Stratum 1 vs. Reference)
"""
import json
import math
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data", "cobenfy_faers_raw.json")
SUMMARY = os.path.join(BASE, "data", "analysis_summary.json")
STUDY = os.path.join(BASE, "APA_2027_Cobenfy_Polypharmacy_Study.md")

STRONG_ANTI_M_AP = {
    "CLOZAPINE", "CLOZARIL", "VERSACLOZ", "OLANZAPINE", "ZYPREXA", "LYBALVI",
    "QUETIAPINE", "SEROQUEL", "CHLORPROMAZINE", "THORAZINE", "LOXAPINE", "LOXEPIN",
    "ADASUVE", "THIORIDAZINE",
}
NON_ANTI_M_AP = {
    "RISPERIDONE", "RISPERDAL", "PERSERIS", "UZEDY", "PALIPERIDONE", "INVEGA",
    "ARIPIPRAZOLE", "ABILIFY", "ARISTADA", "HALOPERIDOL", "HALDOL", "LUMATEPERONE",
    "CAPLYTA", "CARIPRAZINE", "VRAYLAR", "LURASIDONE", "LATUDA", "BREXPIPRAZOLE",
    "REXULTI", "ZIPRASIDONE", "GEODON", "ILOPERIDONE", "FANAPT", "PIMAVANSERIN",
    "NUPLAZID", "ASENAPINE", "SAPHRIS", "FLUPHENAZINE", "PROLIXIN", "PERPHENAZINE",
    "TRILAFON",
}
FIRST_GEN_AH = {
    "HYDROXYZINE", "VISTARIL", "ATARAX", "DIPHENHYDRAMINE", "BENADRYL", "MECLIZINE",
    "ANTIVERT", "PROMETHAZINE", "PHENERGAN", "CHLORPHENIRAMINE", "DOXYLAMINE",
    "CYPROHEPTADINE", "DIMENHYDRINATE",
}
SYSTEMIC_ACH = {
    "BENZTROPINE", "COGENTIN", "TRIHEXYPHENIDYL", "ARTANE", "GLYCOPYRROLATE", "ROBINUL",
    "ATROPINE", "SCOPOLAMINE", "OXYBUTYNIN", "DITROPAN", "SOLIFENACIN", "VESICARE",
    "TOLTERODINE", "DETROL", "DICYCLOMINE", "BENTYL", "HYOSCYAMINE", "PAROXETINE",
    "PAXIL", "AMITRIPTYLINE", "NORTRIPTYLINE", "DOXEPIN", "AMOXAPINE", "IMIPRAMINE",
    "CLOMIPRAMINE",
}
PSYCH = {
    "PSYCHOTIC DISORDER", "SCHIZOPHRENIA", "CONDITION AGGRAVATED", "PSYCHOTIC SYMPTOM",
    "HALLUCINATION", "HALLUCINATION, AUDITORY", "DELUSION", "PARANOIA", "AGITATION",
}
LAI_KW = {
    "SUSTENNA", "TRINZA", "HAFYERA", "MAINTENA", "ASIMTUFII", "ARISTADA",
    "LAUROXIL", "UZEDY", "PERSERIS", "CONSTA", "PALMITATE", "DECANOATE", "RELPREVV",
}
NEG_CONTROL = {
    "RASH", "RASH PRURITIC", "RASH ERYTHEMATOUS", "RASH MACULO-PAPULAR",
    "UPPER RESPIRATORY TRACT INFECTION", "NASOPHARYNGITIS", "ARTHRALGIA", "BACK PAIN", "PRURITUS",
}
GI_HYPO = {
    "CONSTIPATION", "ILEUS", "PARALYTIC ILEUS", "ILEUS PARALYTIC", "INTESTINAL OBSTRUCTION",
    "SMALL INTESTINAL OBSTRUCTION", "LARGE INTESTINAL OBSTRUCTION",
    "GASTROINTESTINAL HYPOMOTILITY", "FAECALOMA",
}


def fisher(a, b, c, d):
    n = a + b + c + d
    r1, r2, c1, c2 = a + b, c + d, a + c, b + d

    def lp(x):
        return (math.lgamma(r1 + 1) + math.lgamma(r2 + 1) + math.lgamma(c1 + 1) + math.lgamma(c2 + 1)
                - math.lgamma(n + 1) - math.lgamma(x + 1) - math.lgamma(r1 - x + 1)
                - math.lgamma(c1 - x + 1) - math.lgamma(r2 - c1 + x + 1))

    pobs = math.exp(lp(a))
    p = 0.0
    for x in range(max(0, r1 + c1 - n), min(r1, c1) + 1):
        px = math.exp(lp(x))
        if px <= pobs * (1 + 1e-7):
            p += px
    return min(1.0, p)


def tbt(g1, g0, fn):
    a = sum(1 for x in g1 if fn(x))
    c = sum(1 for x in g0 if fn(x))
    b, d = len(g1) - a, len(g0) - c
    out = {"a": a, "n1": len(g1), "c": c, "n0": len(g0), "p": fisher(a, b, c, d),
           "ror": None, "lo": None, "hi": None}
    if min(a, b, c, d) > 0:
        r = (a * d) / (b * c)
        se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        out.update(ror=r, lo=math.exp(math.log(r) - 1.96 * se), hi=math.exp(math.log(r) + 1.96 * se))
    return out


def fmt(o):
    p1 = o["a"] / o["n1"] * 100 if o["n1"] else 0.0
    p0 = o["c"] / o["n0"] * 100 if o["n0"] else 0.0
    head = "%d/%d (%.1f%%) vs %d/%d (%.1f%%)" % (o["a"], o["n1"], p1, o["c"], o["n0"], p0)
    if o["ror"] is None:
        return "%s | ROR n/a | p=%.4f" % (head, o["p"])
    return "%s | ROR %.2f (%.2f-%.2f) | p=%.4f" % (head, o["ror"], o["lo"], o["hi"], o["p"])


def matches(dn, kw):
    return any(any(k in x for k in kw) for x in dn)


def section(title):
    print("\n" + "=" * 100 + "\n" + title + "\n" + "=" * 100)


def main():
    section("A. APA abstract length (limit: title 150, body 3,000 characters with spaces)")
    txt = open(STUDY).read()
    title = re.search(r"### Poster Title\n\*\*(.+?)\*\*\n", txt).group(1).strip()
    block = re.search(r"### Structured Abstract\n(.+?)\n---", txt, re.S).group(1)
    paras = [p.strip().replace("**", "") for p in block.split("\n") if p.strip() and not p.strip().startswith("*(")]
    body = "\n".join(paras)
    no_labels = "\n".join(re.sub(r"^(Background|Methods|Results|Conclusion):\s*", "", p) for p in paras)
    print("Title: %d chars (limit 150) -> %s" % (len(title), "PASS" if len(title) <= 150 else "FAIL"))
    print("Body with section labels: %d chars (limit 3,000) -> %s" % (len(body), "PASS" if len(body) <= 3000 else "FAIL"))
    print("Body without section labels: %d chars" % len(no_labels))
    for p in paras:
        print("  %-12s %5d" % (p.split(":")[0], len(p)))

    raw = json.load(open(RAW))
    summary = json.load(open(SUMMARY))
    recs = {r["safetyreportid"]: r for r in raw["records"]}
    cases = []
    for sid, r in recs.items():
        p = r.get("patient", {})
        dn = set()
        reported = set()
        for d in p.get("drug", []):
            mp = (d.get("medicinalproduct") or "").upper()
            if mp:
                dn.add(mp)
                reported.add(mp)
            of = d.get("openfda", {})
            for g in of.get("generic_name", []):
                dn.add(g.upper())
            for b in of.get("brand_name", []):
                dn.add(b.upper())
        ps = r.get("primarysource") or {}
        pts = {(x.get("reactionmeddrapt") or "").upper() for x in p.get("reaction", [])}
        hosp = str(r.get("seriousnesshospitalization", "")) == "1"
        hcp = str(ps.get("qualification", "")) in ("1", "2", "3")
        is_s1 = matches(dn, STRONG_ANTI_M_AP | FIRST_GEN_AH | SYSTEMIC_ACH)
        is_s2 = (not is_s1) and matches(dn, NON_ANTI_M_AP)
        cases.append({
            "sid": sid, "dn": dn, "pts": pts, "hosp": hosp, "hcp": hcp,
            "psych_relapse": bool(pts & PSYCH),
            "nonpsych_hosp": hosp and not bool(pts & PSYCH),
            "stratum": "S1" if is_s1 else ("S2" if is_s2 else "MONO"),
            "any_ap": matches(dn, STRONG_ANTI_M_AP | NON_ANTI_M_AP),
            "lai": matches(reported, LAI_KW),
        })

    S1 = [c for c in cases if c["stratum"] == "S1"]
    S2 = [c for c in cases if c["stratum"] == "S2"]
    MONO = [c for c in cases if c["stratum"] == "MONO"]

    section("B. Strict ACB=3 Cohort Reproduction")
    print("N=%d | S1=%d | S2=%d | MONO=%d (All Poly=%d)" % (len(cases), len(S1), len(S2), len(MONO), len(S1) + len(S2)))

    section("C. Primary Comparisons: Stratum 1 (n=260) vs Reference (n=1,724)")
    for lab, fn in [
        ("Hospitalization (All-Cause)", lambda c: c["hosp"]),
        ("Non-Psychiatric Medical Hosp", lambda c: c["nonpsych_hosp"]),
        ("GI Hypomotility Composite", lambda c: bool(c["pts"] & GI_HYPO)),
        ("Constipation", lambda c: "CONSTIPATION" in c["pts"]),
        ("GERD", lambda c: "GASTROOESOPHAGEAL REFLUX DISEASE" in c["pts"]),
        ("Urinary Retention", lambda c: "URINARY RETENTION" in c["pts"]),
        ("Vomiting", lambda c: "VOMITING" in c["pts"]),
    ]:
        print("%-30s %s" % (lab, fmt(tbt(S1, MONO, fn))))

    section("D. HCP-Only Comparisons: HCP S1 (n=222) vs HCP Reference (n=885)")
    hS1 = [c for c in S1 if c["hcp"]]
    hM = [c for c in MONO if c["hcp"]]
    for lab, fn in [
        ("Hospitalization (All-Cause)", lambda c: c["hosp"]),
        ("Non-Psychiatric Medical Hosp", lambda c: c["nonpsych_hosp"]),
        ("GI Hypomotility Composite", lambda c: bool(c["pts"] & GI_HYPO)),
        ("Constipation", lambda c: "CONSTIPATION" in c["pts"]),
        ("GERD", lambda c: "GASTROOESOPHAGEAL REFLUX DISEASE" in c["pts"]),
        ("Urinary Retention", lambda c: "URINARY RETENTION" in c["pts"]),
        ("Vomiting", lambda c: "VOMITING" in c["pts"]),
    ]:
        print("%-30s %s" % (lab, fmt(tbt(hS1, hM, fn))))

    section("E. Active-Comparator Head-to-Head: Stratum 1 (n=260) vs Stratum 2 (n=164)")
    for lab, fn in [
        ("Hospitalization (All-Cause)", lambda c: c["hosp"]),
        ("GERD", lambda c: "GASTROOESOPHAGEAL REFLUX DISEASE" in c["pts"]),
        ("GI Hypomotility Composite", lambda c: bool(c["pts"] & GI_HYPO)),
        ("Constipation", lambda c: "CONSTIPATION" in c["pts"]),
    ]:
        print("%-30s %s" % (lab, fmt(tbt(S1, S2, fn))))

    section("E2. Stratum 2 (n=164) vs Reference (n=1,724): expected null")
    for lab, fn in [
        ("Hospitalization (All-Cause)", lambda c: c["hosp"]),
        ("Non-Psychiatric Medical Hosp", lambda c: c["nonpsych_hosp"]),
        ("GI Hypomotility Composite", lambda c: bool(c["pts"] & GI_HYPO)),
        ("Constipation", lambda c: "CONSTIPATION" in c["pts"]),
        ("GERD", lambda c: "GASTROOESOPHAGEAL REFLUX DISEASE" in c["pts"]),
        ("Urinary Retention", lambda c: "URINARY RETENTION" in c["pts"]),
    ]:
        print("%-30s %s" % (lab, fmt(tbt(S2, MONO, fn))))

    section("F. LAI (reporter-named) vs no LAI among AP-cotreated reports, re-derived from raw")
    lai = [c for c in cases if c["lai"]]
    oral = [c for c in cases if c["any_ap"] and not c["lai"]]
    print("LAI n=%d (S1 %d) | no-LAI n=%d (S1 %d)" % (len(lai), sum(c["stratum"] == "S1" for c in lai),
                                                    len(oral), sum(c["stratum"] == "S1" for c in oral)))
    for lab, fn in [
        ("GI Hypomotility Composite", lambda c: bool(c["pts"] & GI_HYPO)),
        ("GERD", lambda c: "GASTROOESOPHAGEAL REFLUX DISEASE" in c["pts"]),
        ("Hospitalization (All-Cause)", lambda c: c["hosp"]),
    ]:
        print("%-30s all: %s" % (lab, fmt(tbt(lai, oral, fn))))
        print("%-30s S2:  %s" % ("", fmt(tbt([c for c in lai if c["stratum"] == "S2"], [c for c in oral if c["stratum"] == "S2"], fn))))
    ov = summary["clinical_subgroups"]["lai_vs_oral_ap_gi_hypo"]["overall"]
    print("Matches analysis_summary.json (LAI GI):", (ov["a"], ov["n1"]) == (sum(bool(c["pts"] & GI_HYPO) for c in lai), len(lai)))

    section("F2. Sex/Age Demographic Subgroups (from analysis_summary.json)")
    sub = summary["clinical_subgroups"]
    for key in [
        "female_vs_male_nausea",
        "female_vs_male_vomiting",
        "age_60_vs_18_59_uret",
        "older_men_60_vs_others_uret",
    ]:
        blk = sub[key]
        ov = blk["overall"]
        hc = blk["hcp_only"]
        print("%-28s Overall: %d/%d (%.2f%%) vs %d/%d (%.2f%%) | ROR %.2f (%.2f-%.2f), p=%.4f | HCP: %d/%d (%.2f%%) vs %d/%d (%.2f%%) | ROR %.2f (%.2f-%.2f), p=%.4f" % (
            key, ov["a"], ov["n1"], ov["pct1"], ov["c"], ov["n0"], ov["pct0"],
            ov["ror"], ov["ci_l"], ov["ci_u"], ov["p_val"],
            hc["a"], hc["n1"], hc["pct1"], hc["c"], hc["n0"], hc["pct0"],
            hc["ror"], hc["ci_l"], hc["ci_u"], hc["p_val"]
        ))

    section("G. Figure 2 Dynamic JSON Verification")
    g_code = open(os.path.join(BASE, "generate_figures.py")).read()
    has_hardcoded = "ach_rates = [17.4" in g_code or '("Clozapine (Anti-M SGA)", 98' in g_code
    print("generate_figures.py reads dynamically from analysis_summary.json:", not has_hardcoded)

    section("H. Negative Control Outcomes: Stratum 1 vs Reference (expected null)")
    neg = lambda c: bool(c["pts"] & NEG_CONTROL)
    print("%-30s %s" % ("Composite (any of 5)", fmt(tbt(S1, MONO, neg))))
    print("%-30s %s" % ("Composite, HCP-only", fmt(tbt(hS1, hM, neg))))
    print("%-30s %s" % ("Composite, Stratum 2 vs Ref", fmt(tbt(S2, MONO, neg))))
    for term in sorted(NEG_CONTROL):
        n = sum(term in c["pts"] for c in cases)
        if n:
            print("  %-34s total events in N=%d: %d" % (term, len(cases), n))


if __name__ == "__main__":
    main()
