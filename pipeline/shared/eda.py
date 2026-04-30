"""
National OASIS EDA — Shared cohort exploration.
Loads the master episode scores CSV, cleans columns, builds outcome variable,
and generates demographic/outcome figures for both tasks.
Input:  data/National_OASIS_master_episode_scores.csv  (15.7M × 680)
Output: data/National_OASIS_surface_cleaned.parquet    (15.7M × 640)
        figures/state_patients.png
        figures/race_composition.png
        figures/gender_donut.png
        figures/class_imbalance.png
        figures/ed_overlap.png
        figures/comorbidity_score_charlson_boxplot.png
        figures/comorbidity_score_elixhauser_boxplot.png
        figures/hfrs_boxplot.png
        figures/state_hosp_lollipop.png
        figures/race_distribution.png
        figures/ethnicity_distribution.png
        figures/age_by_outcome.png
        figures/missingness_overview.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading master episode scores CSV...")
df = pd.read_csv("/home/fpd16/independentStudy/data/National_OASIS_master_episode_scores.csv", low_memory=False)
print(f"Shape: {df.shape}")

# Convert to Parquet for faster future loads
df.to_parquet("/home/fpd16/independentStudy/data/National_OASIS_master_episode_scores.parquet",
              engine="fastparquet", index=False)
print("Parquet saved.")

# ── Drop >99% null columns ────────────────────────────────────────────────────
missing_pct = (df.isnull().sum() / len(df) * 100).round(1)
drop_99 = missing_pct[missing_pct > 99].index.tolist()
df.drop(columns=drop_99, inplace=True)
print(f"Dropped {len(drop_99)} columns >99% null. Shape: {df.shape}")

# ── Drop identifier / admin / payment columns ─────────────────────────────────
drop_ids = [c for c in ["Assessment_ID", "HHA_Assessment_Internal_ID",
                         "Facility_Internal_ID", "Branch_Identifier_Number",
                         "Patient_Birth_Date", "Patient_ZIP_Code", "ZIP5",
                         "ZIPCODE", "ZCTA", "POINT_ZIP"] if c in df.columns]
df.drop(columns=drop_ids, inplace=True)

payment_cols = [c for c in df.columns if any(x in c.lower() for x in
    ["payment", "pymnt", "hipps", "medicare_fee", "medicare_hmo", "medicare_hha"])]
df.drop(columns=payment_cols, inplace=True)
print(f"After dropping IDs and payment cols. Shape: {df.shape}")

# ── Build outcome variable ────────────────────────────────────────────────────
emergent_cols = [c for c in df.columns if c.startswith("Emergent_Care_Reason")]
hospital_cols = [c for c in df.columns if c.startswith("Hospital_Reason")]
had_ed = df[emergent_cols].any(axis=1)
had_hospital = df[hospital_cols].any(axis=1)

df["outcome"] = "Stable"
df.loc[had_ed & ~had_hospital, "outcome"] = "ED Only"
df.loc[had_hospital, "outcome"] = "Hospitalized"

print(f"\nOutcome distribution:")
print(df["outcome"].value_counts())

# ── Save cleaned surface ──────────────────────────────────────────────────────
df.to_parquet("/home/fpd16/independentStudy/data/National_OASIS_surface_cleaned.parquet",
              engine="fastparquet", index=False)
print(f"\nSaved National_OASIS_surface_cleaned.parquet. Final shape: {df.shape}")

# ── Figures ───────────────────────────────────────────────────────────────────
plt.rcParams["figure.dpi"] = 100

# 1. Top 15 states by unique patients
state_patients = (df.groupby("Patient_State")["Beneficiary_ID"]
                   .nunique().sort_values(ascending=True).tail(15))
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(range(len(state_patients)), state_patients.values, color="steelblue", edgecolor="white")
ax.set_yticks(range(len(state_patients)))
ax.set_yticklabels(state_patients.index, fontsize=12)
ax.set_xlabel("Unique Patients", fontsize=12)
ax.set_title("Top 15 States by Unique Patient Count", fontsize=14, fontweight="bold")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
for bar in bars:
    w = bar.get_width()
    ax.text(w + w * 0.01, bar.get_y() + bar.get_height() / 2, f"{int(w):,}", va="center", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("/home/fpd16/independentStudy/figures/state_patients.png", dpi=150, bbox_inches="tight")
plt.close()

# 2. Race composition
race_cols = ["White", "Black_or_African_American", "Hispanic_or_Latino",
             "Asian", "American_Indian_or_Alaska_Native", "Native_Hawiian_or_Pacific_Islander"]
race_pcts = (df[race_cols].mean() * 100).sort_values(ascending=True)
race_labels = [c.replace("_", " ") for c in race_pcts.index]
fig, ax = plt.subplots(figsize=(10, 5))
colors = sns.color_palette("viridis", len(race_pcts))
bars = ax.barh(race_labels, race_pcts.values, color=colors)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=11, fontweight="bold")
ax.set_xlabel("% of Patients", fontsize=12)
ax.set_title("Race/Ethnicity Composition", fontsize=14, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("/home/fpd16/independentStudy/figures/race_composition.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. Gender donut
gender_counts = df["Gender"].value_counts()
fig, ax = plt.subplots(figsize=(7, 7))
colors_g = ["#e74c3c", "#3498db"] if gender_counts.index[0] == "Female" else ["#3498db", "#e74c3c"]
wedges, texts, autotexts = ax.pie(
    gender_counts.values, labels=gender_counts.index,
    colors=colors_g,
    autopct=lambda p: f"{p:.1f}%\n({int(p * sum(gender_counts.values) / 100):,})",
    startangle=90, pctdistance=0.75, textprops={"fontsize": 12},
)
for a in autotexts:
    a.set_fontweight("bold")
ax.add_artist(plt.Circle((0, 0), 0.50, fc="white"))
ax.set_title("Gender Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("/home/fpd16/independentStudy/figures/gender_donut.png", dpi=150, bbox_inches="tight")
plt.close()

# 4. Outcome class distribution (log scale)
outcome_counts = df["outcome"].value_counts().reindex(["Stable", "ED Only", "Hospitalized"])
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(range(3), outcome_counts.values, color=["#2ecc71", "#f39c12", "#e74c3c"],
              edgecolor="white", width=0.6)
ax.set_xticks(range(3))
ax.set_xticklabels(outcome_counts.index, fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Outcome Class Distribution", fontsize=14, fontweight="bold")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
for bar, count, pct in zip(bars, outcome_counts.values, outcome_counts.values / len(df) * 100):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.3,
            f"{count:,}\n({pct:.1f}%)", ha="center", fontsize=11, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("/home/fpd16/independentStudy/figures/class_imbalance.png", dpi=150, bbox_inches="tight")
plt.close()

# 5. ED overlap breakdown
ed_only = (had_ed & ~had_hospital).sum()
hosp_only = (had_hospital & ~had_ed).sum()
both = (had_ed & had_hospital).sum()
total_ed = had_ed.sum()
categories = ["ED → Hospitalized\n(ED visit + admitted)",
              "Hospital Only\n(No ED visit)",
              "ED Only\n(Discharged from ED)"]
values = [both, hosp_only, ed_only]
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(categories, values, color=["#c0392b", "#e74c3c", "#f39c12"],
               edgecolor="white", height=0.6)
for bar in bars:
    w = bar.get_width()
    ax.text(w + w * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{int(w):,}  ({w / sum(values) * 100:.1f}%)", va="center", fontsize=11, fontweight="bold")
ax.set_xlabel("Count", fontsize=12)
ax.set_title(f"ED & Hospitalization Overlap ({both / total_ed * 100:.0f}% of ED visits lead to admission)",
             fontsize=13, fontweight="bold")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("/home/fpd16/independentStudy/figures/ed_overlap.png", dpi=150, bbox_inches="tight")
plt.close()

# 6. Comorbidity score box plots by outcome
score_cols = ["comorbidity_score_charlson", "comorbidity_score_elixhauser", "hfrs"]
titles = ["Charlson Comorbidity Index", "Elixhauser Comorbidity Index", "Hospital Frailty Risk Score"]
order = ["Stable", "ED Only", "Hospitalized"]
colors_o = ["#2ecc71", "#f39c12", "#e74c3c"]
df_sample = df[["outcome"] + score_cols].sample(n=500_000, random_state=42)

for col, title in zip(score_cols, titles):
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot([df_sample[df_sample["outcome"] == o][col].values for o in order],
                    labels=order, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors_o):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    means = [df_sample[df_sample["outcome"] == o][col].mean() for o in order]
    ax.scatter(range(1, 4), means, color="black", marker="D", s=60, zorder=5, label="Mean")
    for i, m in enumerate(means):
        ax.text(i + 1.15, m, f"{m:.2f}", va="center", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"/home/fpd16/independentStudy/figures/{col}_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()

# 7. Hospitalization rate by state (lollipop)
state_outcome = pd.crosstab(df["Patient_State"], df["outcome"])
state_outcome["hosp_rate"] = state_outcome["Hospitalized"] / state_outcome.sum(axis=1) * 100
state_outcome["total"] = state_outcome.sum(axis=1)
state_outcome = state_outcome[state_outcome["total"] > 50_000]
state_hosp = state_outcome["hosp_rate"].sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 8))
ax.hlines(y=range(len(state_hosp)), xmin=0, xmax=state_hosp.values, color="indianred", alpha=0.7, linewidth=2)
ax.scatter(state_hosp.values, range(len(state_hosp)), color="indianred", s=80, zorder=3)
ax.set_yticks(range(len(state_hosp)))
ax.set_yticklabels(state_hosp.index, fontsize=11)
ax.set_xlabel("Hospitalization Rate (%)", fontsize=12)
ax.set_title("Hospitalization Rate by State\n(States with >50K rows only)", fontsize=14, fontweight="bold")
for i, val in enumerate(state_hosp.values):
    ax.text(val + 0.1, i, f"{val:.1f}%", va="center", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("/home/fpd16/independentStudy/figures/state_hosp_lollipop.png", dpi=150, bbox_inches="tight")
plt.close()

# 8. Race distribution
race_only = ["White", "Black_or_African_American", "Asian",
             "American_Indian_or_Alaska_Native", "Native_Hawiian_or_Pacific_Islander"]
race_pcts2 = (df[race_only].mean() * 100).sort_values(ascending=True)
race_labels2 = [c.replace("_", " ") for c in race_pcts2.index]
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(race_labels2, race_pcts2.values, color="steelblue", edgecolor="white", height=0.5)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=11, fontweight="bold")
ax.set_xlabel("% of Patients", fontsize=11)
ax.set_title("Patient Race Distribution", fontsize=13, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("/home/fpd16/independentStudy/figures/race_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# 9. Ethnicity
eth_pct = df["Hispanic_or_Latino"].mean() * 100
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.barh(["Hispanic/Latino", "Non-Hispanic/Latino"], [eth_pct, 100 - eth_pct],
               color=["#e67e22", "steelblue"], edgecolor="white", height=0.5)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=11, fontweight="bold")
ax.set_xlabel("% of Patients", fontsize=11)
ax.set_title("Patient Ethnicity Distribution", fontsize=13, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("/home/fpd16/independentStudy/figures/ethnicity_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# 10. Age by outcome
fig, ax = plt.subplots(figsize=(8, 5))
df_sample_age = df[["outcome", "Age"]].sample(n=500_000, random_state=42)
for outcome, color in zip(["Stable", "ED Only", "Hospitalized"], ["#2ecc71", "#f39c12", "#e74c3c"]):
    subset = df_sample_age[df_sample_age["outcome"] == outcome]["Age"].dropna()
    ax.hist(subset, bins=50, alpha=0.5, color=color,
            label=f"{outcome} (mean={subset.mean():.1f})", density=True)
ax.set_xlabel("Age", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Age Distribution by Outcome", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("/home/fpd16/independentStudy/figures/age_by_outcome.png", dpi=150, bbox_inches="tight")
plt.close()

# 11. Missingness overview
missing_pct = (df.isnull().sum() / len(df) * 100).round(1)
missing_buckets = pd.Series({
    "Complete (0%)": (missing_pct == 0).sum(),
    "0-10%":         ((missing_pct > 0) & (missing_pct <= 10)).sum(),
    "10-50%":        ((missing_pct > 10) & (missing_pct <= 50)).sum(),
    "50-95%":        ((missing_pct > 50) & (missing_pct <= 95)).sum(),
    ">95%":          (missing_pct > 95).sum(),
})
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(missing_buckets.index, missing_buckets.values, color="steelblue", edgecolor="white", width=0.6)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 3, f"{int(h)}", ha="center", fontsize=12, fontweight="bold")
ax.set_ylabel("Number of Columns", fontsize=12)
ax.set_title(f"Data Completeness ({len(df.columns)} total columns)", fontsize=13, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("/home/fpd16/independentStudy/figures/missingness_overview.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nAll figures saved.")
