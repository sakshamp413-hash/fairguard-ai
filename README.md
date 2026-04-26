# 🛡️ FairGuard AI

FairGuard AI is a hackathon-ready Streamlit app that audits machine-learning decisions for bias and explains fairness risks using **Google Gemini**.

## Problem

AI systems used in loans, hiring, admissions, and public services can unintentionally disadvantage protected groups. This supports **UN SDG 10: Reduced Inequalities** by helping teams detect and explain unfair outcomes before real people are affected.

## Solution

FairGuard AI lets a user:

1. Generate a synthetic biased loan dataset or upload a CSV.
2. Train a simple ML model.
3. Calculate group-level fairness metrics.
4. Detect disparate impact using the 80% rule.
5. Use **Google Gemini** to generate a judge-friendly ethics explanation and mitigation plan.

## Google Technology Used

| Google technology | Purpose |
| --- | --- |
| Gemini API / Google Gen AI SDK | Generates bias explanations and mitigation recommendations |
| Google AI Studio | API key creation and model testing |
| Optional Google Cloud / Cloud Run | Deployment-ready path for production |

## Local Setup

```bash
git clone https://github.com/sakshamp413-hash/fairguard-ai.git
cd fairguard-ai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

For macOS/Linux:

```bash
source .venv/bin/activate
streamlit run app.py
```

## Gemini API Key Setup

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your_api_key_here"
GEMINI_MODEL = "gemini-2.5-flash"
```

Do **not** commit your real API key to GitHub.

## CSV Format

Your uploaded CSV should have:

- one binary target column, such as `loan_approved`, `selected`, or `hired`
- one protected attribute column, such as `gender`, `race`, `location`, or `age_group`

Example:

```csv
age,income,credit_score,gender,loan_approved
25,40000,650,Female,0
34,70000,720,Male,1
```

## Run

```bash
streamlit run app.py
```

## Hackathon Demo Flow

1. Open the app.
2. Click **Generate Synthetic Biased Dataset**.
3. Select `gender` as the protected attribute.
4. Select `loan_approved` as the decision column.
5. Click **Train Model & Run Bias Analysis**.
6. Show the fairness metrics, chart, and Gemini explanation.

## Future Scope

- Add Vertex AI deployment.
- Add BigQuery dataset support.
- Add model cards for responsible AI reporting.
- Add fairness comparison before and after mitigation.
