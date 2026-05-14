import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')
 
# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)
 
# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding: 1rem 2rem; }
    .stMetric { background: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #2E75B6; }
    .fraud-badge { background:#ffebee; color:#c62828; padding:6px 14px; border-radius:20px; font-weight:700; font-size:15px; }
    .legit-badge { background:#e8f5e9; color:#2e7d32; padding:6px 14px; border-radius:20px; font-weight:700; font-size:15px; }
    .section-header { font-size:20px; font-weight:700; color:#1F4E79; margin:1rem 0 0.5rem 0; }
    div[data-testid="stFileUploader"] { border: 2px dashed #2E75B6; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)
 
 
# ── Load model files ──────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model     = pickle.load(open('fraud_model.pkl',     'rb'))
        scaler    = pickle.load(open('fraud_scaler.pkl',    'rb'))
        threshold = pickle.load(open('fraud_threshold.pkl', 'rb'))
        features  = pickle.load(open('fraud_features.pkl',  'rb'))
        return model, scaler, threshold, features, None
    except FileNotFoundError as e:
        return None, None, None, None, str(e)
 
model, scaler, threshold, features, load_error = load_model()
 
# ── Header ────────────────────────────────────────────────
st.title("🔍 Credit Card Fraud Detection")
st.markdown("Detect fraudulent transactions instantly using Machine Learning (XGBoost).")
st.markdown("---")
 
# ── Show error if model files not found ──────────────────
if load_error:
    st.error(f"""
    **Model files not found!**
 
    Make sure these 4 files are in the same folder as app.py:
    - `fraud_model.pkl`
    - `fraud_scaler.pkl`
    - `fraud_threshold.pkl`
    - `fraud_features.pkl`
 
    Run your Colab notebook first to generate these files, then download them.
 
    Error: `{load_error}`
    """)
    st.stop()
 
st.success(f"Model loaded successfully! Threshold = {threshold} | Features = {len(features)}")
 
# ═════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["📂 Upload CSV & Predict", "✍️ Manual Entry (Demo)"])
 
 
# ═════════════════════════════════════════════════════════
# TAB 1 — Upload CSV (Main way to use the app)
# ═════════════════════════════════════════════════════════
with tab1:

    st.markdown(
        '<p class="section-header">📂 Upload Your Transaction CSV</p>',
        unsafe_allow_html=True
    )

    st.info("""
    CSV must contain:
    • V1 to V28
    • Amount

    Optional columns:
    • id
    • Time
    • Class
    """)

    uploaded = st.file_uploader(
        "Choose CSV File",
        type=['csv'],
        key="csv_upload"
    )

    if uploaded is not None:

        # ── Read CSV ──────────────────────────────────────
        try:
            df_raw = pd.read_csv(uploaded)

        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()

        if df_raw.empty:
            st.error("Uploaded CSV is empty.")
            st.stop()

        st.success(
            f"Loaded {len(df_raw):,} rows and {df_raw.shape[1]} columns"
        )

        # ── Preview ───────────────────────────────────────
        with st.expander("👁️ Preview First 10 Rows"):

            st.dataframe(
                df_raw.head(10),
                use_container_width=True
            )

        # ── Copy for processing ───────────────────────────
        df_proc = df_raw.copy()

        # Remove unnecessary columns
        if 'id' in df_proc.columns:
            df_proc.drop(columns=['id'], inplace=True)

        if 'Time' in df_proc.columns:
            df_proc.drop(columns=['Time'], inplace=True)

        # Save actual labels if present
        actual_labels = None

        if 'Class' in df_proc.columns:

            actual_labels = (
                df_proc['Class']
                .copy()
                .reset_index(drop=True)
            )

            df_proc.drop(columns=['Class'], inplace=True)

        # ── Validate columns ──────────────────────────────
        missing_cols = [
            c for c in features
            if c not in df_proc.columns
            and c != 'Amount_scaled'
        ]

        amount_missing = (
            'Amount' not in df_proc.columns
            and 'Amount_scaled' not in df_proc.columns
        )

        if missing_cols or amount_missing:

            st.error(
                f"Missing columns: "
                f"{missing_cols + (['Amount'] if amount_missing else [])}"
            )

            st.stop()

        # ── Scale Amount ──────────────────────────────────
        if 'Amount' in df_proc.columns:

            df_proc['Amount_scaled'] = scaler.transform(
                df_proc[['Amount']]
            )

            df_proc.drop(columns=['Amount'], inplace=True)

        # ── Reorder features ──────────────────────────────
        try:
            df_proc = df_proc[features]

        except KeyError as e:

            st.error(f"Feature mismatch: {e}")
            st.stop()

        # ── Predict ───────────────────────────────────────
        probas = model.predict_proba(df_proc)[:, 1]

        preds = (
            probas >= threshold
        ).astype(int)

        # ── Build result dataframe ────────────────────────
        result_df = df_raw.copy().reset_index(drop=True)

        result_df['Fraud_Probability_%'] = (
            probas * 100
        ).round(2)

        result_df['Prediction'] = [
            '🚨 FRAUD'
            if p else '✅ LEGITIMATE'
            for p in preds
        ]

        if actual_labels is not None:
            result_df['Actual_Class'] = actual_labels

        # ── Summary metrics ───────────────────────────────
        st.markdown("---")

        st.markdown(
            '<p class="section-header">📊 Results Summary</p>',
            unsafe_allow_html=True
        )

        total = len(preds)

        n_fraud = int(preds.sum())

        n_legit = int((preds == 0).sum())

        fraud_pct = (
            n_fraud / total * 100
            if total > 0 else 0
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Transactions",
            f"{total:,}"
        )

        c2.metric(
            "🚨 Fraud Detected",
            f"{n_fraud:,}",
            delta=f"{fraud_pct:.2f}%"
        )

        c3.metric(
            "✅ Legitimate",
            f"{n_legit:,}"
        )

        c4.metric(
            "Threshold",
            f"{threshold*100:.0f}%"
        )

        # ── Model metrics ─────────────────────────────────
        if actual_labels is not None:

            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                roc_auc_score
            )

            acc = accuracy_score(actual_labels, preds)

            f1 = f1_score(
                actual_labels,
                preds,
                zero_division=0
            )

            auc = (
                roc_auc_score(actual_labels, probas)
                if actual_labels.nunique() > 1
                else 0
            )

            st.markdown("### 🤖 Model Performance")

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Accuracy",
                f"{acc*100:.2f}%"
            )

            m2.metric(
                "F1 Score",
                f"{f1:.4f}"
            )

            m3.metric(
                "ROC-AUC",
                f"{auc:.4f}"
            )

        # ── Top suspicious transactions ───────────────────
        st.markdown("---")

        st.markdown("### 🚨 Top 10 Most Suspicious Transactions")

        top10 = (
            result_df
            .nlargest(10, 'Fraud_Probability_%')
            [['Fraud_Probability_%', 'Amount', 'Prediction']]
            .reset_index(drop=True)
        )

        top10.index += 1

        st.dataframe(
            top10,
            use_container_width=True
        )

        # ── Prediction preview ────────────────────────────
        st.markdown("---")

        st.markdown("### 📋 Prediction Preview (First 100 Rows)")

        display_cols = [
            'Fraud_Probability_%',
            'Prediction'
        ]

        if 'Amount' in result_df.columns:
            display_cols = ['Amount'] + display_cols

        if actual_labels is not None:
            display_cols.append('Actual_Class')

        def color_fraud(val):

            if (
                isinstance(val, str)
                and 'FRAUD' in val
            ):

                return (
                    'background-color:#ffebee;'
                    'color:#c62828;'
                    'font-weight:bold'
                )

            if (
                isinstance(val, str)
                and 'LEGIT' in val
            ):

                return (
                    'background-color:#e8f5e9;'
                    'color:#2e7d32'
                )

            return ''

        st.dataframe(
            result_df[display_cols]
            .head(100)
            .style.map(
                color_fraud,
                subset=['Prediction']
            ),
            use_container_width=True,
            height=420
        )

        st.info(
            "Showing first 100 rows only for performance."
        )

        # ── Download button ───────────────────────────────
        st.markdown("---")

        csv_download = result_df.to_csv(index=False)

        st.download_button(
            label="⬇️ Download Full Results CSV",
            data=csv_download,
            file_name="fraud_predictions.csv",
            mime="text/csv",
            type="primary"
        )
                # ── Charts & Visual Analysis ─────────────────────
        st.markdown("---")

        st.markdown("## 📈 Visual Analysis")

        col_a, col_b = st.columns(2)

        # ── Histogram ────────────────────────────────────
        with col_a:

            st.markdown("### Fraud Probability Distribution")

            fig, ax = plt.subplots(figsize=(6, 4))

            if (preds == 0).sum() > 0:

                ax.hist(
                    probas[preds == 0] * 100,
                    bins=40,
                    alpha=0.7,
                    label='Legitimate',
                    color='green'
                )

            if (preds == 1).sum() > 0:

                ax.hist(
                    probas[preds == 1] * 100,
                    bins=40,
                    alpha=0.7,
                    label='Fraud',
                    color='red'
                )

            ax.axvline(
                x=threshold * 100,
                color='blue',
                linestyle='--',
                label=f'Threshold ({threshold*100:.0f}%)'
            )

            ax.set_xlabel("Fraud Probability (%)")
            ax.set_ylabel("Count")

            ax.legend()

            st.pyplot(fig)

            plt.close()

        # ── Pie Chart ────────────────────────────────────
        with col_b:

            st.markdown("### Transaction Breakdown")

            fig2, ax2 = plt.subplots(figsize=(5, 4))

            sizes = [n_legit, n_fraud]

            labels = [
                f'Legitimate\n{n_legit:,}',
                f'Fraud\n{n_fraud:,}'
            ]

            colors = ['green', 'red']

            ax2.pie(
                sizes,
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors
            )

            ax2.axis('equal')

            st.pyplot(fig2)

            plt.close()

        # ── Additional Charts ────────────────────────────
        col_c, col_d = st.columns(2)

        # ── Bar chart ────────────────────────────────────
        with col_c:

            st.markdown("### Fraud vs Legitimate Count")

            fig3, ax3 = plt.subplots(figsize=(5, 4))

            categories = ['Legitimate', 'Fraud']

            values = [n_legit, n_fraud]

            colors = ['green', 'red']

            ax3.bar(
                categories,
                values,
                color=colors
            )

            ax3.set_ylabel("Transactions")

            for i, v in enumerate(values):

                ax3.text(
                    i,
                    v,
                    str(v),
                    ha='center',
                    va='bottom'
                )

            st.pyplot(fig3)

            plt.close()

        # ── Amount statistics ────────────────────────────
        with col_d:

            st.markdown("### Amount Statistics")

            if 'Amount' in result_df.columns:

                amt_stats = result_df.copy()

                amt_stats['Type'] = [
                    'Fraud'
                    if p else 'Legitimate'
                    for p in preds
                ]

                stats = (
                    amt_stats
                    .groupby('Type')['Amount']
                    .agg(['mean', 'min', 'max', 'count'])
                    .round(2)
                )

                stats.columns = [
                    'Average',
                    'Minimum',
                    'Maximum',
                    'Count'
                ]

                st.dataframe(
                    stats,
                    use_container_width=True
                )

            else:

                st.info("Amount column not available.")

# ═════════════════════════════════════════════════════════
# TAB 2 — Manual Entry (Demo/Testing)
# ═════════════════════════════════════════════════════════
with tab2:
 
    st.markdown('<p class="section-header">✍️ Manual Entry — For Demo & Testing Only</p>', unsafe_allow_html=True)
    st.warning("""
    **Note:** In real usage, upload a CSV file (Tab 1). V1–V28 are PCA-transformed bank features
    that cannot be typed meaningfully by hand. Use the quick-fill buttons below to test with
    real sample values from the dataset.
    """)
 
    # ── Session state for quick fill ──────────────────────
    if 'amount_val' not in st.session_state:
        st.session_state.amount_val = 100.0
    if 'v_values' not in st.session_state:
        st.session_state.v_values = {f'V{i}': 0.0 for i in range(1, 29)}
 
    # Legitimate sample values (row 0 from dataset)
    LEGIT_SAMPLE = {
        'amount': 17982.10,
        'v': [-0.26064780, -0.46964845, 2.49626608, -0.08372391,
               0.12968124,  0.73289825, 0.51901362, -0.13000605,
               0.72715927,  0.63773454, -0.98702001, 0.29343810,
              -0.94138613,  0.54901989,  1.80487858,  0.21559799,
               0.51230666,  0.33364372,  0.12427016,  0.09120190,
              -0.11055168,  0.21760614, -0.13479449,  0.16595912,
               0.12627998, -0.43482398, -0.08123011, -0.15104549]
    }
 
    # Simulated fraud sample (known fraud patterns from dataset)
    FRAUD_SAMPLE = {
        'amount': 1.00,
        'v': [-4.7716, 3.4838, -7.2024,  4.4600, -2.9000, -0.6021, -8.3800,
               0.6630, -4.5100, -6.4300,  4.3000, -12.400,  2.1600, -9.4700,
               1.4400, -5.2800, -18.900,  5.4800, -8.2700, -0.2400,  0.7800,
               0.4500, -0.2300,  0.3100, -0.1400,  0.2900,  0.4200,  0.2500]
    }
 
    # Quick fill buttons
    c1, c2, c3 = st.columns([1, 1, 2])
 
    if c1.button("✅ Fill Legitimate Sample", use_container_width=True):
        st.session_state.amount_val = LEGIT_SAMPLE['amount']
        for i, v in enumerate(LEGIT_SAMPLE['v'], 1):
            st.session_state.v_values[f'V{i}'] = float(v)
        st.rerun()
 
    if c2.button("🚨 Fill Fraud Sample", use_container_width=True):
        st.session_state.amount_val = FRAUD_SAMPLE['amount']
        for i, v in enumerate(FRAUD_SAMPLE['v'], 1):
            st.session_state.v_values[f'V{i}'] = float(v)
        st.rerun()
 
    if c3.button("🔄 Reset to Zero", use_container_width=True):
        st.session_state.amount_val = 100.0
        for i in range(1, 29):
            st.session_state.v_values[f'V{i}'] = 0.0
        st.rerun()
 
    st.markdown("---")
 
    # ── Amount ────────────────────────────────────────────
    amount = st.number_input(
        "Transaction Amount (₹)",
        min_value=0.0,
        max_value=1000000.0,
        value=float(st.session_state.amount_val),
        format="%.2f"
    )
 
    # ── V1 to V28 in 4 columns ────────────────────────────
    st.markdown("**V1 – V28 Feature Values:**")
    grid = st.columns(4)
    v_inputs = {}
 
    for i in range(1, 29):
        col = grid[(i - 1) % 4]
        v_inputs[f'V{i}'] = col.number_input(
            label=f'V{i}',
            value=float(st.session_state.v_values.get(f'V{i}', 0.0)),
            format="%.4f",
            key=f'vinput_{i}'
        )
 
    st.markdown("---")
 
    # ── Predict button ────────────────────────────────────
    if st.button("🔍 Predict This Transaction", type="primary", use_container_width=True):
 
        # Build single-row dataframe
        row_data = {**v_inputs, 'Amount': amount}
        df_single = pd.DataFrame([row_data])
 
        # Scale Amount
        df_single['Amount_scaled'] = scaler.transform(df_single[['Amount']])
        df_single.drop(columns=['Amount'], inplace=True)
 
        # Order features
        try:
            df_single = df_single[features]
        except KeyError as e:
            st.error(f"Feature error: {e}")
            st.stop()
 
        # Predict
        proba    = model.predict_proba(df_single)[0][1]
        is_fraud = proba >= threshold
 
        # ── Result display ─────────────────────────────────
        st.markdown("### Prediction Result")
 
        r1, r2, r3 = st.columns(3)
        r1.metric("Fraud Probability",  f"{proba * 100:.2f}%")
        r2.metric("Transaction Amount", f"₹{amount:,.2f}")
        r3.metric("Threshold",          f"{threshold * 100:.0f}%")
 
        # Progress bar
        st.progress(float(proba))
 
        if is_fraud:
            st.error(f"""
            ### 🚨 FRAUD DETECTED
            This transaction has a **{proba*100:.2f}%** probability of being fraudulent.
            It exceeds the threshold of {threshold*100:.0f}%.
            **Recommended action: Block this transaction.**
            """)
        else:
            st.success(f"""
            ### ✅ LEGITIMATE TRANSACTION
            This transaction has only a **{proba*100:.2f}%** fraud probability.
            It is below the threshold of {threshold*100:.0f}%.
            **Recommended action: Approve this transaction.**
            """)
 
        # Risk gauge bar
        st.markdown("**Risk Level:**")
        risk_color = "#e74c3c" if proba > 0.7 else "#f39c12" if proba > 0.3 else "#27ae60"
        risk_label = "HIGH RISK" if proba > 0.7 else "MEDIUM RISK" if proba > 0.3 else "LOW RISK"
        st.markdown(f"""
        <div style="background:#eee; border-radius:10px; height:24px; margin:4px 0 12px 0;">
          <div style="background:{risk_color}; width:{proba*100:.1f}%; height:24px;
                      border-radius:10px; display:flex; align-items:center;
                      justify-content:center; color:white; font-size:12px; font-weight:700;">
            {risk_label} — {proba*100:.1f}%
          </div>
        </div>
        """, unsafe_allow_html=True)
 
 
# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:13px; padding: 10px 0;">
    Model: XGBoost &nbsp;|&nbsp; Dataset: Kaggle Credit Card Fraud (284,807 transactions)
    &nbsp;|&nbsp; Built with Streamlit &nbsp;|&nbsp; For educational purposes
</div>
""", unsafe_allow_html=True)



