import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from google import genai
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="FairGuard AI",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ FairGuard AI")
st.subheader("Real-time Bias Detector & Fairness Guardian")
st.markdown(
    "**Google Solution Challenge 2026** — powered by **Google Gemini** "
    "for AI ethics explanations and mitigation recommendations."
)


# ======================
# GEMINI CONFIG
# ======================
def get_secret(name: str, default: str = "") -> str:
    """Read from Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GEMINI_MODEL = get_secret("GEMINI_MODEL", "gemini-2.5-flash")

client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as exc:
        st.sidebar.error(f"Gemini setup failed: {exc}")
else:
    st.sidebar.warning(
        "Gemini is not configured. Add GEMINI_API_KEY in Streamlit Secrets "
        "or as an environment variable."
    )


# ======================
# DATA
# ======================
@st.cache_data
def generate_synthetic_data(n: int = 2000) -> pd.DataFrame:
    """Generate demo loan data with intentional gender bias for auditing."""
    rng = np.random.default_rng(42)

    age = rng.normal(35, 12, n).clip(18, 70).astype(int)
    income = rng.normal(55_000, 20_000, n).clip(15_000, 150_000).astype(int)
    credit_score = rng.normal(700, 80, n).clip(300, 850).astype(int)
    gender = rng.choice(["Male", "Female"], n, p=[0.5, 0.5])

    base_approval_probability = (
        (income > 45_000).astype(float) * 0.45
        + (credit_score > 650).astype(float) * 0.40
        + (age > 24).astype(float) * 0.10
    ).clip(0.05, 0.95)

    loan_approved = (rng.random(n) < base_approval_probability).astype(int)

    # Intentional bias: reduce positive outcomes for one protected group.
    female_mask = gender == "Female"
    approval_flip = rng.random(female_mask.sum()) < 0.28
    loan_approved[female_mask] = np.where(
        approval_flip,
        0,
        loan_approved[female_mask],
    )

    return pd.DataFrame(
        {
            "age": age,
            "income": income,
            "credit_score": credit_score,
            "gender": gender,
            "loan_approved": loan_approved,
        }
    )


# ======================
# VALIDATION + METRICS
# ======================
def validate_dataset(df: pd.DataFrame, protected_attr: str, target_col: str) -> Tuple[bool, str]:
    if df.empty:
        return False, "Dataset is empty."

    if protected_attr == target_col:
        return False, "Protected attribute and target column must be different."

    if df[protected_attr].nunique(dropna=True) < 2:
        return False, "Protected attribute must contain at least 2 groups."

    unique_targets = sorted(df[target_col].dropna().unique().tolist())
    if len(unique_targets) != 2:
        return False, "Target column must be binary, for example 0/1 or Yes/No."

    return True, ""


def normalize_binary_target(series: pd.Series) -> pd.Series:
    """Convert common binary labels to 0/1 integers."""
    if pd.api.types.is_numeric_dtype(series):
        unique_values = sorted(series.dropna().unique().tolist())
        mapping = {unique_values[0]: 0, unique_values[-1]: 1}
        return series.map(mapping).astype(int)

    normalized = series.astype(str).str.strip().str.lower()
    positive = {"1", "yes", "true", "approved", "accept", "accepted", "selected", "hire", "hired"}
    negative = {"0", "no", "false", "rejected", "reject", "declined", "not selected", "not hired"}
    mapped = normalized.map(lambda value: 1 if value in positive else 0 if value in negative else np.nan)

    if mapped.isna().any():
        unique_values = sorted(normalized.dropna().unique().tolist())
        mapping = {unique_values[0]: 0, unique_values[-1]: 1}
        mapped = normalized.map(mapping)

    return mapped.astype(int)


def calculate_fairness_metrics(
    df: pd.DataFrame,
    protected_attr: str,
    target_col: str = "actual",
    pred_col: str = "predicted",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []

    for group in sorted(df[protected_attr].dropna().unique()):
        subset = df[df[protected_attr] == group]
        selection_rate = float(subset[pred_col].mean())
        accuracy = float(accuracy_score(subset[target_col], subset[pred_col]))

        rows.append(
            {
                "Group": str(group),
                "Selection Rate": round(selection_rate, 3),
                "Accuracy": round(accuracy, 3),
                "Samples": int(len(subset)),
            }
        )

    metrics_df = pd.DataFrame(rows)

    if metrics_df.empty:
        summary = {
            "disparate_impact": 0,
            "verdict": "Unable to calculate",
            "disadvantaged_group": "Unknown",
            "advantaged_group": "Unknown",
        }
        return metrics_df, summary

    min_row = metrics_df.loc[metrics_df["Selection Rate"].idxmin()]
    max_row = metrics_df.loc[metrics_df["Selection Rate"].idxmax()]
    max_rate = float(max_row["Selection Rate"])
    min_rate = float(min_row["Selection Rate"])
    disparate_impact = round(min_rate / max_rate, 3) if max_rate > 0 else 0

    summary = {
        "disparate_impact": disparate_impact,
        "verdict": "Fair" if disparate_impact >= 0.8 else "Biased",
        "disadvantaged_group": str(min_row["Group"]),
        "advantaged_group": str(max_row["Group"]),
    }

    return metrics_df, summary


def get_gemini_explanation(
    metrics_df: pd.DataFrame,
    protected_attr: str,
    summary: Dict[str, Any],
) -> str:
    if client is None:
        return (
            "Gemini API is not configured. Add GEMINI_API_KEY to Streamlit Secrets "
            "to enable Google AI ethics explanations."
        )

    prompt = f"""
You are FairGuard, an AI ethics auditor for a Google Developer Student Clubs / Google Solution Challenge hackathon demo.

Audit context:
- Protected attribute: {protected_attr}
- Disparate Impact Score: {summary["disparate_impact"]}
- Verdict: {summary["verdict"]}
- Disadvantaged group: {summary["disadvantaged_group"]}
- Advantaged group: {summary["advantaged_group"]}
- Group metrics table: {metrics_df.to_dict(orient="records")}

Write a judge-friendly explanation in exactly 4 short bullets:
1. Verdict
2. Why this matters for real people
3. Likely root cause
4. Concrete technical fix

Keep it clear, ethical, non-legal, and under 110 words.
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text or "Gemini returned an empty response."
    except Exception as exc:
        return f"Gemini error: {exc}"


# ======================
# SIDEBAR
# ======================
st.sidebar.header("Demo Controls")
demo_size = st.sidebar.slider("Synthetic dataset size", min_value=200, max_value=5000, value=2000, step=100)
st.sidebar.caption(f"Gemini model: `{GEMINI_MODEL}`")


# ======================
# MAIN APP
# ======================
left, right = st.columns([1, 1])

with left:
    if st.button("Generate Synthetic Biased Dataset", type="primary", use_container_width=True):
        st.session_state.data = generate_synthetic_data(demo_size)
        st.success("Demo dataset generated.")

with right:
    uploaded_file = st.file_uploader("Or upload your own CSV", type=["csv"])

if uploaded_file is not None:
    try:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.success("Custom dataset loaded.")
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")

if "data" not in st.session_state:
    st.info("Start by generating the demo dataset or uploading a CSV.")
    st.stop()

df = st.session_state.data.copy()

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    protected = st.selectbox(
        "Protected Attribute",
        options=list(df.columns),
        index=list(df.columns).index("gender") if "gender" in df.columns else 0,
    )

with col2:
    default_target_index = list(df.columns).index("loan_approved") if "loan_approved" in df.columns else len(df.columns) - 1
    target = st.selectbox(
        "Target / Decision Column",
        options=list(df.columns),
        index=default_target_index,
    )

is_valid, validation_error = validate_dataset(df, protected, target)
if not is_valid:
    st.error(validation_error)
    st.stop()

if st.button("Train Model & Run Bias Analysis", type="primary"):
    with st.spinner("Training model and calculating fairness metrics..."):
        clean_df = df.dropna(subset=[protected, target]).copy()
        clean_df[target] = normalize_binary_target(clean_df[target])

        X = clean_df.drop(columns=[target])
        y = clean_df[target]

        X_encoded = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y if y.nunique() == 2 else None,
        )

        classifier = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced",
        )
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        test_df = clean_df.loc[X_test.index].copy()
        test_df["actual"] = y_test.values
        test_df["predicted"] = predictions

        metrics_df, summary = calculate_fairness_metrics(
            test_df,
            protected_attr=protected,
            target_col="actual",
            pred_col="predicted",
        )

    st.subheader("Overall Model Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy:.1%}")
    c2.metric("Disparate Impact", summary["disparate_impact"])
    c3.metric("Verdict", "✅ Fair" if summary["verdict"] == "Fair" else "❌ Bias Detected")

    st.subheader("Fairness Report")
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Selection Rate by Group")
    fig = px.bar(
        metrics_df,
        x="Group",
        y="Selection Rate",
        text="Selection Rate",
        title=f"Approval / Selection Rate by {protected}",
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Google Gemini Ethics Explanation")
    with st.spinner("Gemini is generating the judge-friendly explanation..."):
        explanation = get_gemini_explanation(metrics_df, protected, summary)
    st.info(explanation)

    st.subheader("Recommended Mitigation")
    st.markdown(
        f"""
- Re-train and evaluate the model after reducing dependency on `{protected}` and proxy features.
- Compare selection rates before and after mitigation.
- Keep a human-review step for high-impact decisions.
- Log fairness metrics continuously after deployment.
"""
    )
