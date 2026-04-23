
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import json
import re

# FILE PATHS

MODEL_A_PATH = "autoqa_output_gpt4o_240(1).csv"
MODEL_B_PATH = "autoqa_output_gpt51_240(2).csv"
OUTCOMES_PATH = "outcomes_240(2).csv"

OUTPUT_DIR = Path("autoqa_analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def parse_percent_from_text(text):
    
    if pd.isna(text):
        return np.nan
    text = str(text)
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
    if matches:
        return float(matches[0])
    return np.nan

def cohen_kappa_binary(y1, y2):
    y1 = np.asarray(y1, dtype=int)
    y2 = np.asarray(y2, dtype=int)
    n = len(y1)
    if n == 0:
        return np.nan

    p0 = np.mean(y1 == y2)

    p1_yes = np.mean(y1 == 1)
    p1_no = np.mean(y1 == 0)
    p2_yes = np.mean(y2 == 1)
    p2_no = np.mean(y2 == 0)

    pe = p1_yes * p2_yes + p1_no * p2_no

    if np.isclose(1 - pe, 0):
        return np.nan
    return (p0 - pe) / (1 - pe)

def prevalence_index(y1, y2):
    y1 = np.asarray(y1, dtype=int)
    y2 = np.asarray(y2, dtype=int)
    n = len(y1)
    if n == 0:
        return np.nan
    a = np.sum((y1 == 1) & (y2 == 1))
    d = np.sum((y1 == 0) & (y2 == 0))
    return abs(a - d) / n

def bias_index(y1, y2):
    y1 = np.asarray(y1, dtype=int)
    y2 = np.asarray(y2, dtype=int)
    n = len(y1)
    if n == 0:
        return np.nan
    b = np.sum((y1 == 1) & (y2 == 0))
    c = np.sum((y1 == 0) & (y2 == 1))
    return abs(b - c) / n

def point_biserial_manual(x, y):
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]
    if len(x) == 0 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def cv_auc_single_feature(x, y):
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    valid = x.notna() & y.notna()
    X = pd.DataFrame({"x": x[valid]})
    y = y[valid].astype(int)

    if len(X) < 30 or y.nunique() < 2:
        return np.nan, np.nan

    class_counts = y.value_counts()
    if class_counts.min() < 2:
        return np.nan, np.nan

    n_splits = min(5, int(class_counts.min()))
    if n_splits < 2:
        return np.nan, np.nan

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())

def cv_auc_multifeature(X, y):
    X = X.copy()
    y = pd.Series(y).astype(float)
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid].astype(int)

    if len(X) < 30 or y.nunique() < 2:
        return np.nan, np.nan

    class_counts = y.value_counts()
    if class_counts.min() < 2:
        return np.nan, np.nan

    n_splits = min(5, int(class_counts.min()))
    if n_splits < 2:
        return np.nan, np.nan

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", LogisticRegression(max_iter=5000))
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())

def readiness_bucket(kappa):
    if pd.isna(kappa):
        return "unknown"
    if kappa >= 0.70:
        return "ready_candidate"
    elif kappa >= 0.45:
        return "borderline_candidate"
    else:
        return "not_ready_candidate"


df_a = pd.read_csv(MODEL_A_PATH)
df_b = pd.read_csv(MODEL_B_PATH)
df_out = pd.read_csv(OUTCOMES_PATH)


required_id = "record_id"
if required_id not in df_a.columns or required_id not in df_b.columns or required_id not in df_out.columns:
    raise ValueError("record_id column is required in all files.")


binary_items = [f"item_{i}_answer" for i in range(1, 17)]
quant_comment_items = [f"item_{i}_comment" for i in range(17, 23)]
text_support_suffixes = ["question", "why", "examples", "instead"]

for col in binary_items:
    if col not in df_a.columns or col not in df_b.columns:
        raise ValueError(f"Missing binary item column: {col}")

merged = (
    df_a.merge(df_b, on="record_id", suffixes=("_a", "_b"))
        .merge(df_out, on="record_id", how="left")
)


reliability_rows = []

for i, item_col in enumerate(binary_items, start=1):
    col_a = f"{item_col}_a"
    col_b = f"{item_col}_b"

    y1 = pd.to_numeric(merged[col_a], errors="coerce")
    y2 = pd.to_numeric(merged[col_b], errors="coerce")
    valid = y1.notna() & y2.notna()

    y1v = y1[valid].astype(int)
    y2v = y2[valid].astype(int)

    agreement = float(np.mean(y1v == y2v))
    kappa = cohen_kappa_binary(y1v, y2v)
    prev_idx = prevalence_index(y1v, y2v)
    bias_idx = bias_index(y1v, y2v)

    reliability_rows.append({
        "item_num": i,
        "item_col": item_col,
        "question": merged[f"item_{i}_question_a"].dropna().iloc[0],
        "n": int(len(y1v)),
        "model_a_positive_rate": float(np.mean(y1v)),
        "model_b_positive_rate": float(np.mean(y2v)),
        "positive_rate_abs_diff": float(abs(np.mean(y1v) - np.mean(y2v))),
        "raw_agreement": agreement,
        "cohens_kappa": kappa,
        "prevalence_index": prev_idx,
        "bias_index": bias_idx,
        "both_1": int(np.sum((y1v == 1) & (y2v == 1))),
        "both_0": int(np.sum((y1v == 0) & (y2v == 0))),
        "a1_b0": int(np.sum((y1v == 1) & (y2v == 0))),
        "a0_b1": int(np.sum((y1v == 0) & (y2v == 1))),
        "discordant_total": int(np.sum(y1v != y2v)),
        "readiness_bucket": readiness_bucket(kappa)
    })

reliability_df = pd.DataFrame(reliability_rows).sort_values(
    ["cohens_kappa", "raw_agreement"], ascending=[False, False]
)
reliability_df.to_csv(OUTPUT_DIR / "01_item_reliability_summary.csv", index=False)


ready_df = reliability_df[reliability_df["readiness_bucket"] == "ready_candidate"].copy()
border_df = reliability_df[reliability_df["readiness_bucket"] == "borderline_candidate"].copy()
not_ready_df = reliability_df[reliability_df["readiness_bucket"] == "not_ready_candidate"].copy()

if len(ready_df) > 0:
    ready_pick = ready_df.iloc[0]
else:
    ready_pick = reliability_df.iloc[0]

if len(border_df) > 0:
    border_pick = border_df.sort_values(["cohens_kappa", "positive_rate_abs_diff"], ascending=[False, True]).iloc[0]
else:
    border_pick = reliability_df.iloc[len(reliability_df) // 2]

if len(not_ready_df) > 0:
    not_ready_pick = not_ready_df.sort_values(["cohens_kappa", "raw_agreement"], ascending=[True, True]).iloc[0]
else:
    not_ready_pick = reliability_df.iloc[-1]

deep_dive_choice_df = pd.DataFrame([
    {"category": "ready_for_operational_use_candidate", **ready_pick.to_dict()},
    {"category": "borderline_candidate", **border_pick.to_dict()},
    {"category": "not_ready_candidate", **not_ready_pick.to_dict()},
])
deep_dive_choice_df.to_csv(OUTPUT_DIR / "02_recommended_deep_dive_items.csv", index=False)


examples_rows = []

for item_num in [int(ready_pick["item_num"]), int(border_pick["item_num"]), int(not_ready_pick["item_num"])]:
    answer_a = f"item_{item_num}_answer_a"
    answer_b = f"item_{item_num}_answer_b"
    keep_cols = ["record_id", answer_a, answer_b]

    for suffix in text_support_suffixes:
        ca = f"item_{item_num}_{suffix}_a"
        cb = f"item_{item_num}_{suffix}_b"
        if ca in merged.columns:
            keep_cols.append(ca)
        if cb in merged.columns:
            keep_cols.append(cb)

    temp = merged[keep_cols].copy()
    temp["case_type"] = np.where(temp[answer_a] == temp[answer_b], "agreement", "disagreement")
    temp["item_num"] = item_num

    picked = pd.concat([
        temp[temp["case_type"] == "disagreement"].head(10),
        temp[temp["case_type"] == "agreement"].head(5),
    ], ignore_index=True)

    examples_rows.append(picked)

examples_df = pd.concat(examples_rows, ignore_index=True)
examples_df.to_csv(OUTPUT_DIR / "03_deep_dive_examples.csv", index=False)


outcomes = ["next_lesson_attended", "m1_retained"]
predictive_rows = []

for outcome in outcomes:
    y = pd.to_numeric(merged[outcome], errors="coerce")

    for i, item_col in enumerate(binary_items, start=1):
        x = pd.to_numeric(merged[f"{item_col}_b"], errors="coerce")
        valid = x.notna() & y.notna()
        xv = x[valid].astype(int)
        yv = y[valid].astype(int)

        rate_when_1 = float(yv[xv == 1].mean()) if (xv == 1).any() else np.nan
        rate_when_0 = float(yv[xv == 0].mean()) if (xv == 0).any() else np.nan
        abs_rate_diff = float(rate_when_1 - rate_when_0) if pd.notna(rate_when_1) and pd.notna(rate_when_0) else np.nan
        lift_ratio = float(rate_when_1 / rate_when_0) if pd.notna(rate_when_1) and pd.notna(rate_when_0) and rate_when_0 != 0 else np.nan
        corr = point_biserial_manual(xv, yv)
        auc_mean, auc_std = cv_auc_single_feature(xv, yv)

        predictive_rows.append({
            "outcome": outcome,
            "item_num": i,
            "item_col": item_col,
            "question": merged[f"item_{i}_question_b"].dropna().iloc[0],
            "n": int(len(xv)),
            "positive_rate": float(xv.mean()),
            "outcome_rate_when_item_1": rate_when_1,
            "outcome_rate_when_item_0": rate_when_0,
            "absolute_rate_diff": abs_rate_diff,
            "lift_ratio": lift_ratio,
            "pointbiserial_r": corr,
            "single_feature_cv_auc_mean": auc_mean,
            "single_feature_cv_auc_std": auc_std
        })

predictive_item_df = pd.DataFrame(predictive_rows).sort_values(
    ["outcome", "absolute_rate_diff"], ascending=[True, False]
)
predictive_item_df.to_csv(OUTPUT_DIR / "04_predictive_usefulness_item_level.csv", index=False)


overall_rows = []

for outcome in outcomes:
    y = pd.to_numeric(merged[outcome], errors="coerce")
    for label in ["a", "b"]:
        x = pd.to_numeric(merged[f"binary_items_ones_pct_{label}"], errors="coerce")
        corr = point_biserial_manual(x, y)
        auc_mean, auc_std = cv_auc_single_feature(x, y)
        overall_rows.append({
            "outcome": outcome,
            "model_version": f"model_{label.upper()}",
            "n": int((x.notna() & y.notna()).sum()),
            "pointbiserial_r": corr,
            "single_feature_cv_auc_mean": auc_mean,
            "single_feature_cv_auc_std": auc_std
        })

overall_df = pd.DataFrame(overall_rows)
overall_df.to_csv(OUTPUT_DIR / "05_predictive_usefulness_overall_score.csv", index=False)


multi_rows = []

X = merged[[f"item_{i}_answer_b" for i in range(1, 17)]].apply(pd.to_numeric, errors="coerce")
for outcome in outcomes:
    y = pd.to_numeric(merged[outcome], errors="coerce")
    auc_mean, auc_std = cv_auc_multifeature(X, y)
    multi_rows.append({
        "outcome": outcome,
        "feature_set": "all_16_binary_items_model_B",
        "n": int(y.notna().sum()),
        "multivariable_cv_auc_mean": auc_mean,
        "multivariable_cv_auc_std": auc_std
    })

multi_df = pd.DataFrame(multi_rows)
multi_df.to_csv(OUTPUT_DIR / "06_predictive_usefulness_multivariable.csv", index=False)


quant_rows = []

for i in range(17, 23):
    comment_col = f"item_{i}_comment_b"
    parsed = merged[comment_col].apply(parse_percent_from_text)

    for outcome in outcomes:
        y = pd.to_numeric(merged[outcome], errors="coerce")
        corr = point_biserial_manual(parsed, y)
        auc_mean, auc_std = cv_auc_single_feature(parsed, y)

        quant_rows.append({
            "comment_item_num": i,
            "comment_col": f"item_{i}_comment",
            "question": merged[f"item_{i}_question_b"].dropna().iloc[0],
            "outcome": outcome,
            "n": int((parsed.notna() & y.notna()).sum()),
            "parsed_percent_mean": float(parsed.mean()) if parsed.notna().any() else np.nan,
            "pointbiserial_r": corr,
            "single_feature_cv_auc_mean": auc_mean,
            "single_feature_cv_auc_std": auc_std
        })

quant_df = pd.DataFrame(quant_rows)
quant_df.to_csv(OUTPUT_DIR / "07_quant_comment_predictive_usefulness.csv", index=False)


top_attended = predictive_item_df[predictive_item_df["outcome"] == "next_lesson_attended"].head(3)
top_retained = predictive_item_df[predictive_item_df["outcome"] == "m1_retained"].head(3)

summary_lines = []
summary_lines.append("# AutoQA Analysis Summary")
summary_lines.append("")
summary_lines.append("## Files used")
summary_lines.append(f"- Model A: `{MODEL_A_PATH}`")
summary_lines.append(f"- Model B: `{MODEL_B_PATH}`")
summary_lines.append(f"- Outcomes: `{OUTCOMES_PATH}`")
summary_lines.append("")
summary_lines.append("## Reliability")
summary_lines.append(f"- Best ready candidate: item {int(ready_pick['item_num'])}")
summary_lines.append(f"  - kappa = {ready_pick['cohens_kappa']:.3f}")
summary_lines.append(f"  - agreement = {ready_pick['raw_agreement']:.3f}")
summary_lines.append(f"- Borderline candidate: item {int(border_pick['item_num'])}")
summary_lines.append(f"  - kappa = {border_pick['cohens_kappa']:.3f}")
summary_lines.append(f"  - agreement = {border_pick['raw_agreement']:.3f}")
summary_lines.append(f"- Not ready candidate: item {int(not_ready_pick['item_num'])}")
summary_lines.append(f"  - kappa = {not_ready_pick['cohens_kappa']:.3f}")
summary_lines.append(f"  - agreement = {not_ready_pick['raw_agreement']:.3f}")
summary_lines.append("")
summary_lines.append("## Top promising items for next_lesson_attended")
for _, row in top_attended.iterrows():
    summary_lines.append(
        f"- item {int(row['item_num'])}: diff={row['absolute_rate_diff']:.3f}, "
        f"r={row['pointbiserial_r']:.3f}, auc={row['single_feature_cv_auc_mean']:.3f}"
    )
summary_lines.append("")
summary_lines.append("## Top promising items for m1_retained")
for _, row in top_retained.iterrows():
    summary_lines.append(
        f"- item {int(row['item_num'])}: diff={row['absolute_rate_diff']:.3f}, "
        f"r={row['pointbiserial_r']:.3f}, auc={row['single_feature_cv_auc_mean']:.3f}"
    )
summary_lines.append("")
summary_lines.append("## Output files")
summary_lines.extend([
    "- `01_item_reliability_summary.csv`",
    "- `02_recommended_deep_dive_items.csv`",
    "- `03_deep_dive_examples.csv`",
    "- `04_predictive_usefulness_item_level.csv`",
    "- `05_predictive_usefulness_overall_score.csv`",
    "- `06_predictive_usefulness_multivariable.csv`",
    "- `07_quant_comment_predictive_usefulness.csv`",
])

(OUTPUT_DIR / "00_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

print("Done. Outputs saved to:", OUTPUT_DIR.resolve())
