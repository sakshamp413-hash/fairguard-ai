import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px

st.set_page_config(page_title="FairGuard AI", page_icon="🛡️", layout="wide")
st.title("🛡️ FairGuard AI")
st.subheader("Real-time Bias Detector & Fairness Guardian")
st.markdown("**Hackathon Working Prototype** — Detects bias instantly")

# ====================== SYNTHETIC DATA (with built-in bias) ======================
def generate_synthetic_data(n=2000):
    np.random.seed(42)
    age = np.random.normal(35, 12, n).clip(18, 70).astype(int)
    income = np.random.normal(55000, 20000, n).clip(15000, 150000).astype(int)
    credit_score = np.random.normal(700, 80, n).clip(300, 850).astype(int)
    gender = np.random.choice(['Male', 'Female'], n, p=[0.5, 0.5])
    
    # Base approval logic (good features → approved)
    base_approval = ((income > 45000).astype(int) * 0.6) + ((credit_score > 650).astype(int) * 0.4)
    loan_approved = (np.random.rand(n) < base_approval).astype(int)
    
    # === INTENTIONAL GENDER BIAS ===
    female_idx = (gender == 'Female')
    loan_approved[female_idx] = (loan_approved[female_idx] * np.random.choice([0.65, 1.0], female_idx.sum(), p=[0.35, 0.65])).astype(int)
    
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'gender': gender,
        'loan_approved': loan_approved
    })
    return df

# ====================== FAIRNESS METRICS ======================
def calculate_fairness_metrics(df, protected_attr, target_col='loan_approved', pred_col='predicted'):
    groups = df[protected_attr].unique()
    metrics = {}
    
    for group in groups:
        subset = df[df[protected_attr] == group]
        selection_rate = subset[pred_col].mean()
        accuracy = accuracy_score(subset[target_col], subset[pred_col])
        metrics[group] = {
            'Selection Rate': round(selection_rate, 3),
            'Accuracy': round(accuracy, 3),
            'Samples': len(subset)
        }
    
    # Disparate Impact (standard fairness metric)
    if len(groups) == 2:
        rates = [metrics[g]['Selection Rate'] for g in groups]
        di = min(rates) / max(rates) if max(rates) > 0 else 0
        metrics['Disparate Impact'] = round(di, 3)
        metrics['Fair?'] = '✅ Fair' if di >= 0.8 else '❌ Biased'
    
    return metrics

# ====================== MAIN APP ======================
# Option 1: Synthetic data
if st.button("🚀 Generate Synthetic Biased Dataset (Loan Approval)", type="primary"):
    df = generate_synthetic_data()
    st.session_state.data = df
    st.success("✅ Dataset with **real gender bias** generated!")

# Option 2: Upload your own CSV
uploaded = st.file_uploader("Or upload your own CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    st.session_state.data = df
    st.success("✅ Custom dataset loaded!")

if 'data' in st.session_state:
    df = st.session_state.data
    
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(8), use_container_width=True)
    
    # Column selection
    col1, col2 = st.columns(2)
    with col1:
        protected = st.selectbox("Protected Attribute (e.g. gender, race)", df.columns, key="protected")
    with col2:
        target = st.selectbox("Target Column (AI Decision)", df.columns, index=df.columns.get_loc('loan_approved') if 'loan_approved' in df.columns else len(df.columns)-1, key="target")
    
    if st.button("🔥 Train Model & Run Bias Analysis", type="primary"):
        with st.spinner("Training Random Forest + Running fairness audit..."):
            # Prepare data
            X = df.drop(columns=[target])
            y = df[target]
            X = pd.get_dummies(X, drop_first=True)  # handle categorical
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Overall accuracy
            acc = accuracy_score(y_test, y_pred)
            
            # Create test dataframe with original protected column
            test_df = df.loc[X_test.index].copy()
            test_df['predicted'] = y_pred
            test_df['actual'] = y_test.values
            
            # Calculate bias
            metrics = calculate_fairness_metrics(test_df, protected_attr=protected, target_col='actual', pred_col='predicted')
            
            # ====================== RESULTS ======================
            st.subheader("📈 Overall Model Performance")
            st.metric("Accuracy", f"{acc:.1%}")
            
            st.subheader("⚖️ Bias & Fairness Report")
            metric_df = pd.DataFrame(metrics).T
            st.dataframe(metric_df, use_container_width=True)
            
            # Visualization
            st.subheader("📊 Selection Rate by Group (where bias appears)")
            viz_data = {k: v['Selection Rate'] for k, v in metrics.items() 
                       if isinstance(v, dict) and 'Selection Rate' in v}
            fig = px.bar(x=list(viz_data.keys()), y=list(viz_data.values()),
                         labels={'x': protected, 'y': 'Selection Rate'},
                         title="Approval Rate by Group", color=list(viz_data.keys()))
            st.plotly_chart(fig, use_container_width=True)
            
            if 'Disparate Impact' in metrics:
                di = metrics['Disparate Impact']
                st.metric("Disparate Impact Ratio", di, 
                         "✅ Passes 80% rule" if di >= 0.8 else "❌ Bias Detected")
            
            st.success("✅ **Working prototype complete!** You just built a real bias detection system.")
            st.caption("In a real hackathon I would now add: SHAP explanations, one-click mitigation, PDF report, and 'Bias Bounty' mode.")

st.caption("FairGuard AI • Built live for the hackathon • Copy-paste → Run → Win")