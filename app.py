"""
Streamlit app for credit risk inference using a pre-trained model.
- Loads model and preprocessors from models/
- Single applicant form
- Batch scoring via CSV upload
- SHAP explanations (if SHAP is installed)
"""

import os
import json
from typing import Optional
from datetime import datetime
from io import BytesIO

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from ui.theme import apply_theme
from ui.components import section_card, button_bar
# PDF export library imports removed to prevent crashes when package is missing.
# If PDF export is reintroduced, wrap imports in try/except and gate features by a flag.

from src.models.credit_model import CreditRiskModel
from src.data.processor import DataProcessor

# Small helper for inline info popovers (fallback to expander if popover not available)
def info_tip(label: str, title: str, body_md: str):
    try:
        pop = st.popover(label)
        with pop:
            st.markdown(f"**{title}**")
            st.markdown(body_md)
    except Exception:
        with st.expander(label):
            st.markdown(f"**{title}**")
            st.markdown(body_md)

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(STORAGE_DIR, "credit_risk_model.pkl")
METADATA_PATH = os.path.join(STORAGE_DIR, "model_metadata.json")
SCALER_PATH = os.path.join(STORAGE_DIR, "preprocessor", "scaler.pkl")
ENCODERS_PATH = os.path.join(STORAGE_DIR, "preprocessor", "encoders.pkl")

st.set_page_config(
    page_title="Credit Risk Assessment Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Plotly global theming for professional look (brand colors only, skip template to avoid plotly.io import issues)
px.defaults.color_discrete_sequence = ['#1a4fa3']

# Inject theme-driven CSS (configurable via config/ui.json)
_TOKENS = apply_theme(os.path.join(os.path.dirname(__file__), "config", "ui.json"))

@st.cache_resource
def load_model_and_processor():
    model = None
    processor = None
    metadata = {}
    try:
        if os.path.exists(MODEL_PATH):
            model = CreditRiskModel.load(MODEL_PATH, METADATA_PATH if os.path.exists(METADATA_PATH) else None)
        if os.path.exists(SCALER_PATH) and os.path.exists(ENCODERS_PATH):
            processor = DataProcessor.load(SCALER_PATH, ENCODERS_PATH)
        if os.path.exists(METADATA_PATH):
            try:
                with open(METADATA_PATH, 'r') as f:
                    metadata = json.load(f)
            except Exception:
                metadata = {}
    except Exception as e:
        st.error(f"⚠️ Model loading failed: {e}")
        st.info("The app will run in demo mode without a trained model. Train a model first or check that model files exist.")
    return model, processor, metadata


# Improved splash loading: show header first, then reveal full app
import time
st.session_state.setdefault("splash_done", False)

model = None
processor = None
metadata = {}

if not st.session_state["splash_done"]:
    splash = st.empty()
    with splash.container():
        # Header shown immediately
        st.markdown('<div class="main-header">Credit Risk Assessment Platform</div>', unsafe_allow_html=True)
        st.markdown("### Professional Credit Decisioning System")
        st.markdown("**Powered by Advanced Machine Learning Analytics**")
        st.markdown("---")

        # Centered progress UI
        prog_col, txt_col = st.columns([4, 1])
        with prog_col:
            prog = st.progress(0)
        with txt_col:
            pct = st.empty()

        # Initial animation
        prog.progress(10)
        pct.text("10%")
        time.sleep(0.1)

        # Load resources (may take a moment)
        model, processor, metadata = load_model_and_processor()

        # Smooth finish animation
        for p, msg in [(45, "45%"), (60, "60%"), (80, "80%"), (100, "100%")]:
            prog.progress(p)
            pct.text(msg)
            time.sleep(0.1)

    # Hide splash and mark done
    splash.empty()
    st.session_state["splash_done"] = True
else:
    # Subsequent reruns: no splash
    model, processor, metadata = load_model_and_processor()

# Main header (shown after splash is cleared) - hero section
st.markdown("""
<div class="hero-section">
    <div class="hero-icon">🏦</div>
    <div class="hero-content">
        <h1 class="hero-title">Credit Risk Assessment Platform</h1>
        <p class="hero-subtitle">Professional Credit Decisioning System</p>
        <p class="hero-badge">Powered by Advanced Machine Learning Analytics</p>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

if model is None or processor is None:
    st.error("Model or preprocessor not found. Run the training script scripts/train_and_save_model.py to create artifacts.")
    st.stop()

threshold = float(metadata.get('threshold', 0.5))
model.optimal_threshold = threshold

# Helper: build a sample input form based on processor.feature_names
feature_names = processor.feature_names or metadata.get('feature_names') or []

# Hero overview section for a portfolio-like landing
with st.container():
    col_a, col_b = st.columns([2, 3])
    with col_a:
        st.markdown("#### Quick actions")
        qa1, qa2, qa3, qa4 = st.columns(4)
        with qa1:
            if st.button("Single", use_container_width=True, key="qa_single"):
                st.session_state["mode_override"] = "Single Applicant"
                st.rerun()
        with qa2:
            if st.button("Batch", use_container_width=True, key="qa_batch"):
                st.session_state["mode_override"] = "Batch Processing"
                st.rerun()
        with qa3:
            if st.button("Scenario", use_container_width=True, key="qa_scenario"):
                st.session_state["mode_override"] = "Scenario Analysis"
                st.rerun()
        with qa4:
            if st.button("Portfolio", use_container_width=True, key="qa_portfolio"):
                st.session_state["mode_override"] = "Portfolio Dashboard"
                st.rerun()
st.markdown("---")

# Sidebar with model info
with st.sidebar:
    # System Information removed per request
    st.header("ADVANCED FEATURES")
    show_comparison = st.checkbox("Show Scenario Comparison", value=False)
    show_trends = st.checkbox("Show Risk Sensitivity Analysis", value=False)
    export_pdf = st.checkbox("Generate PDF Report", value=False)
    
    st.markdown("---")
    st.markdown("**RISK CLASSIFICATION:**")
    st.markdown("**LOW RISK: < 40% default probability**")
    st.markdown("**MEDIUM RISK: 40-75% default probability**")
    st.markdown("**HIGH RISK: > 75% default probability**")

# Default mode is Single Applicant (mode selector removed)
mode = st.session_state.get("mode_override", "Single Applicant")


def create_risk_gauge(probability: float) -> go.Figure:
    """Create a professional gauge chart for risk visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Default Risk Level", 'font': {'size': 20, 'color': '#2c3e50', 'family': 'Arial'}},
        number={'suffix': "%", 'font': {'size': 48, 'color': '#2c3e50'}},
        delta={'reference': model.optimal_threshold * 100, 'increasing': {'color': '#1a4fa3'}, 'decreasing': {'color': '#2362c7'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#2c3e50", 'tickfont': {'size': 14}},
            'bar': {'color': "#1a4fa3", 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "#2c3e50",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(219, 234, 254, 0.8)'},
                {'range': [40, 75], 'color': 'rgba(147, 197, 253, 0.7)'},
                {'range': [75, 100], 'color': 'rgba(26, 79, 163, 0.5)'}
            ],
            'threshold': {
                'line': {'color': "#153b7c", 'width': 6},
                'thickness': 0.8,
                'value': model.optimal_threshold * 100
            }
        }
    ))
    fig.update_layout(
        height=350,
        margin=dict(l=30, r=30, t=60, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': "Arial, sans-serif"}
    )
    return fig


def create_score_breakdown(inputs: dict) -> go.Figure:
    """Create a horizontal bar chart showing key risk factors."""
    # Normalize key metrics for visualization
    factors = {
        'Credit Score': min(inputs.get('credit_score', 650) / 850 * 100, 100),
        'Income Level': min(inputs.get('income', 50000) / 200000 * 100, 100),
        'Debt Management': (1 - inputs.get('debt_to_income_ratio', 0.3)) * 100,
        'Credit History': min(inputs.get('credit_history_months', 60) / 300 * 100, 100),
        'Loan-to-Income': max(100 - (inputs.get('loan_amount', 20000) / inputs.get('income', 50000) * 100), 0)
    }
    
    df_factors = pd.DataFrame(list(factors.items()), columns=['Factor', 'Score'])
    df_factors = df_factors.sort_values('Score', ascending=True)
    
    # Single brand color with varying opacity based on score
    def blue_rgba(score):
        # opacity from 0.4 (low) to 1.0 (high)
        alpha = 0.4 + (max(0, min(score, 100)) / 100.0) * 0.6
        return f"rgba(26,79,163,{alpha:.2f})"
    colors = [blue_rgba(s) for s in df_factors['Score']]
    
    fig = go.Figure(go.Bar(
        x=df_factors['Score'],
        y=df_factors['Factor'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='#2c3e50', width=1)
        ),
        text=df_factors['Score'].round(1),
        texttemplate='%{text}%',
        textposition='outside',
        textfont=dict(size=12, color='#2c3e50')
    ))
    
    fig.update_layout(
        title={
            'text': 'Financial Profile Analysis',
            'font': {'size': 18, 'color': '#2c3e50', 'family': 'Arial'},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=380,
        showlegend=False,
        xaxis_title="Strength Score (%)",
        yaxis_title="",
        font={'family': "Arial, sans-serif", 'size': 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,249,250,0.5)",
        xaxis=dict(
            range=[0, 110],
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True
        ),
        yaxis=dict(
            tickfont={'size': 12, 'color': '#2c3e50'}
        ),
        margin=dict(l=20, r=40, t=60, b=40)
    )
    return fig


def generate_pdf_report(inputs: dict, prob: float, pred: int, risk: str, threshold: float) -> BytesIO:
    """Generate a professional PDF credit assessment report."""
    # Import reportlab lazily to avoid hard dependency when PDF export is not used
    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib import colors
        from reportlab.lib.units import inch
    except Exception as e:
        # Provide a clear error if reportlab isn't installed
        raise RuntimeError("PDF generation requires the 'reportlab' package. Please install it to enable PDF export.") from e

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a4fa3'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    # Title
    story.append(Paragraph("CREDIT RISK ASSESSMENT REPORT", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Decision banner (single brand blue)
    decision_text = "APPROVED" if pred == 0 else "DECLINED"
    decision_color = colors.HexColor('#1a4fa3')
    decision_style = ParagraphStyle(
        'Decision',
        parent=styles['Normal'],
        fontSize=18,
        textColor=colors.white,
        backColor=decision_color,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        spaceBefore=12,
        spaceAfter=12,
        leftIndent=0,
        rightIndent=0,
        leading=24
    )
    story.append(Paragraph(f"CREDIT APPLICATION {decision_text}", decision_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Risk Metrics
    story.append(Paragraph("RISK METRICS", heading_style))
    metrics_data = [
        ['Metric', 'Value'],
        ['Default Probability', f"{prob:.2%}"],
        ['Repayment Probability', f"{(1-prob)*100:.1f}%"],
        ['Risk Category', risk.upper()],
        ['Decision Threshold', f"{threshold*100:.1f}%"],
        ['Decision', decision_text]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[3*inch, 3*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2362c7')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Applicant Information
    story.append(Paragraph("APPLICANT INFORMATION", heading_style))
    applicant_data = [
        ['Field', 'Value'],
        ['Age', str(inputs.get('age', 'N/A'))],
        ['Annual Income', f"${inputs.get('income', 0):,.0f}"],
        ['Credit Score', str(inputs.get('credit_score', 'N/A'))],
        ['Credit History', f"{inputs.get('credit_history_months', 0)} months"],
        ['Existing Loans', str(inputs.get('existing_loans', 0))],
        ['Debt-to-Income Ratio', f"{inputs.get('debt_to_income_ratio', 0):.2%}"],
        ['Loan Amount Requested', f"${inputs.get('loan_amount', 0):,.0f}"],
        ['Employment Status', inputs.get('employment_status', 'N/A')],
        ['Housing Status', inputs.get('housing_status', 'N/A')],
        ['Loan Purpose', inputs.get('loan_purpose', 'N/A')]
    ]
    
    applicant_table = Table(applicant_data, colWidths=[3*inch, 3*inch])
    applicant_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2362c7')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    story.append(applicant_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Executive Summary
    story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
    repay_prob = (1 - prob) * 100
    
    if pred == 0:
        summary_text = f"""
        <b>Recommendation: APPROVE</b><br/><br/>
        The applicant demonstrates a <b>{risk.lower()} risk profile</b> with a <b>{prob:.2%}</b> probability of default, 
        which is below the institutional threshold of <b>{threshold*100:.1f}%</b>.<br/><br/>
        <b>Key Strengths:</b><br/>
        • Default risk: {prob:.2%} (threshold: {threshold*100:.1f}%)<br/>
        • Repayment probability: {repay_prob:.1f}%<br/>
        • Credit score: {inputs['credit_score']}<br/>
        • Annual income: ${inputs['income']:,.0f}<br/><br/>
        <b>Recommendation:</b> Proceed with loan approval subject to standard terms and conditions.
        """
    else:
        summary_text = f"""
        <b>Recommendation: DECLINE</b><br/><br/>
        The applicant demonstrates a <b>{risk.lower()} risk profile</b> with a <b>{prob:.2%}</b> probability of default, 
        which exceeds the institutional threshold of <b>{threshold*100:.1f}%</b>.<br/><br/>
        <b>Risk Factors:</b><br/>
        • Default risk: {prob:.2%} (threshold: {threshold*100:.1f}%)<br/>
        • Repayment probability: {repay_prob:.1f}%<br/>
        • Risk category: {risk}<br/><br/>
        <b>Recommendation:</b> Decline application or consider alternative lending products with adjusted terms.
        """
    
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Footer
    footer_text = """
    <i>This report is generated by an automated credit risk assessment system using machine learning. 
    All decisions should be reviewed by qualified credit officers before final approval. 
    This document is confidential and intended solely for internal use.</i>
    """
    story.append(Paragraph(footer_text, styles['Italic']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


def assess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure columns align
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        st.warning(f"Uploaded data is missing columns: {missing}. Attempting to continue with available columns.")
    
    # Add a dummy target column if not present (required by processor.transform)
    if 'target' not in df.columns:
        df = df.copy()
        df['target'] = 0
    
    X, _ = processor.transform(df, target_col='target')
    
    if X is None or X.empty:
        raise ValueError("Transform returned empty data")
    
    probs = model.predict_proba(X)
    
    if probs is None:
        raise ValueError("Model prediction returned None")
    
    preds = (probs >= model.optimal_threshold).astype(int)
    results = df.drop(columns=['target'], errors='ignore').copy()
    results['default_prob'] = probs
    results['prediction'] = preds
    def risk_label(p):
        if p >= 0.75:
            return 'High'
        if p >= 0.4:
            return 'Medium'
        return 'Low'
    results['risk'] = results['default_prob'].apply(risk_label)
    return results


if mode == "Single Applicant":
    st.header("INDIVIDUAL CREDIT ASSESSMENT")
    
    # Single clean layout without outer squares (no nested panel wrappers)
    with st.container():
        col_left, col_right = st.columns([3, 2])
    
    with col_left:
        with st.form(key='single_form'):
            st.subheader("Applicant Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### PERSONAL DETAILS")
                inputs = {}
                inputs['age'] = st.slider("Age", 18, 75, 35)
                inputs['income'] = st.number_input("Annual Income ($)", min_value=10000.0, max_value=500000.0, value=50000.0, step=5000.0)
                inputs['credit_score'] = st.slider("Credit Score", 300, 850, 650)
                inputs['credit_history_months'] = st.slider("Credit History (months)", 0, 360, 60)
                inputs['existing_loans'] = st.number_input("Existing Loans", min_value=0, max_value=10, value=1)
            
            with col2:
                st.markdown("##### FINANCIAL DETAILS")
                inputs['debt_to_income_ratio'] = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3, 0.01)
                inputs['loan_amount'] = st.number_input("Requested Loan Amount ($)", min_value=1000.0, max_value=500000.0, value=20000.0, step=1000.0)

                emp_opts = ['Employed', 'Self-Employed', 'Unemployed', 'Retired']
                emp_sel = st.selectbox("Employment Status", emp_opts)
                inputs['employment_status'] = emp_sel

                house_opts = ['Own', 'Rent', 'Mortgage']
                house_sel = st.selectbox("Housing Status", house_opts)
                inputs['housing_status'] = house_sel

                purpose_opts = ['Personal', 'Auto', 'Home', 'Education', 'Business']
                purpose_sel = st.selectbox("Loan Purpose", purpose_opts)
                inputs['loan_purpose'] = purpose_sel
            
            submit = st.form_submit_button("ASSESS CREDIT APPLICATION", use_container_width=True)
    
    with col_right:
        st.subheader("ASSESSMENT GUIDELINES")
        st.markdown(
        """
        **ASSESSMENT CRITERIA:**

        - Applications are APPROVED when default risk is below the threshold
        - Applications are REJECTED when default risk exceeds the threshold

        **KEY FACTORS EVALUATED:**
        - Credit history and score
        - Income stability and debt obligations
        - Employment status and history
        - Loan purpose and amount
        - Housing status and existing loans
        - Debt-to-income ratio
        """
        )

    # No outer panel wrapper to avoid empty squares
    
    if submit:
        # Build dataframe
        df_in = pd.DataFrame([inputs])
        
        # Debug: show what we're sending
        # st.write("Debug - Input data:", df_in)
        
        try:
            scored = assess_dataframe(df_in)
            
            if scored is None or scored.empty:
                st.error("Assessment returned no results")
            else:
                prob = float(scored['default_prob'].iloc[0])
                pred = int(scored['prediction'].iloc[0])
                risk = scored['risk'].iloc[0]
                
                st.markdown("---")
                st.markdown("## CREDIT DECISION REPORT")
                st.markdown("")
                
                # Visual decision banner (brand-styled)
                if pred == 0:
                    decision_bg = "#1a4fa3"
                    decision_text = "APPROVED"
                else:
                    decision_bg = "#1a4fa3"
                    decision_text = "DECLINED"
                st.markdown(
                    f"""
                    <div style="background:{decision_bg};color:#ffffff;padding:14px 18px;border-radius:0px;font-weight:700;font-size:1.1rem;letter-spacing:0.5px;">
                        CREDIT APPLICATION {decision_text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Create two columns for visualizations
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Risk gauge
                    st.plotly_chart(create_risk_gauge(prob), use_container_width=True)
                
                with viz_col2:
                    # Profile breakdown
                    st.plotly_chart(create_score_breakdown(inputs), use_container_width=True)
                
                st.markdown("---")
                
                # Detailed metrics
                st.subheader("RISK METRICS")
                info_tip("ⓘ What are these?", "Risk metrics explained", """
                - Default Risk: Estimated probability of default for this application.
                - Repayment Probability: 1 minus default risk.
                - Risk Category: Low/Medium/High bands for quick interpretation.
                - Decision: Outcome based on the current threshold policy.
                """)
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Default Risk", 
                        f"{prob:.2%}", 
                        delta=f"{(prob - model.optimal_threshold):.2%}" if prob > model.optimal_threshold else f"{(model.optimal_threshold - prob):.2%}",
                        delta_color="inverse",
                        help="Probability that the applicant will default on the loan"
                    )
                with col2:
                    repay_prob = (1 - prob) * 100
                    st.metric(
                        "Repayment Probability", 
                        f"{repay_prob:.1f}%",
                        help="Probability that the applicant will successfully repay the loan"
                    )
                with col3:
                    # Risk category with color indicator
                    st.metric(
                        "Risk Category", 
                        f"{risk.upper()}",
                        help="Overall risk classification based on default probability"
                    )
                with col4:
                    st.metric(
                        "Decision", 
                        decision_text,
                        help=f"Based on threshold of {model.optimal_threshold*100:.1f}%"
                    )
                
                # Executive summary
                st.markdown("---")
                st.subheader("EXECUTIVE SUMMARY")
                
                if pred == 0:
                    summary = f"""
                    **Recommendation:** **APPROVE**
                    
                    The applicant demonstrates a **{risk.lower()} risk profile** with a **{prob:.2%}** probability of default, 
                    which is **below** the institutional threshold of **{model.optimal_threshold*100:.1f}%**.
                    
                    **Key Strengths:**
                    - Default risk: {prob:.2%} (threshold: {model.optimal_threshold*100:.1f}%)
                    - Repayment probability: {repay_prob:.1f}%
                    - Credit score: {inputs['credit_score']}
                    - Annual income: ${inputs['income']:,.0f}
                    
                    **Recommendation:** Proceed with loan approval subject to standard terms and conditions.
                    """
                else:
                    summary = f"""
                    **Recommendation:** **DECLINE**
                    
                    The applicant demonstrates a **{risk.lower()} risk profile** with a **{prob:.2%}** probability of default, 
                    which **exceeds** the institutional threshold of **{model.optimal_threshold*100:.1f}%**.
                    
                    **Risk Factors:**
                    - Default risk: {prob:.2%} (threshold: {model.optimal_threshold*100:.1f}%)
                    - Repayment probability: {repay_prob:.1f}%
                    - Risk category: {risk}
                    
                    **Recommendation:** Decline application or consider alternative lending products with adjusted terms.
                    """
                
                st.markdown(summary)
                
                # Model confidence note
                st.info(f"""
                **Model Performance:** This assessment is generated by a machine learning model with 
                **{metadata.get('auc', 0)*100:.1f}% accuracy (AUC)** on historical data. 
                The model analyzes {len(feature_names)} key financial indicators to predict credit risk.
                """)
                
                # Advanced features
                if show_comparison:
                    st.markdown("---")
                    st.subheader("WHAT-IF SCENARIOS")
                    st.write("See how adjustments to key factors would affect the decision:")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    scenarios_quick = {
                        'Current': inputs,
                        'With +100 Credit': {**inputs, 'credit_score': min(inputs['credit_score'] + 100, 850)},
                        'With +$20K Income': {**inputs, 'income': inputs['income'] + 20000},
                        'With -10% Debt': {**inputs, 'debt_to_income_ratio': max(inputs['debt_to_income_ratio'] - 0.1, 0)},
                    }
                    
                    comparison_results = []
                    for sc_name, sc_inputs in scenarios_quick.items():
                        df_sc = pd.DataFrame([sc_inputs])
                        sc_result = assess_dataframe(df_sc)
                        comparison_results.append({
                            'Scenario': sc_name,
                            'Risk': f"{sc_result['default_prob'].iloc[0]:.2%}",
                            'Decision': 'APPROVED' if sc_result['prediction'].iloc[0] == 0 else 'DECLINED'
                        })
                    
                    comp_df = pd.DataFrame(comparison_results)
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)
                
                if show_trends:
                    st.markdown("---")
                    st.subheader("RISK SENSITIVITY ANALYSIS")
                    
                    # Credit score sensitivity
                    credit_range = range(max(300, inputs['credit_score'] - 150), min(850, inputs['credit_score'] + 150), 25)
                    credit_risks = []
                    
                    for cs in credit_range:
                        temp_inputs = {**inputs, 'credit_score': cs}
                        df_temp = pd.DataFrame([temp_inputs])
                        temp_result = assess_dataframe(df_temp)
                        credit_risks.append({
                            'Credit Score': cs,
                            'Default Risk': temp_result['default_prob'].iloc[0]
                        })
                    
                    risk_df = pd.DataFrame(credit_risks)
                    
                    fig_sens = px.line(
                        risk_df,
                        x='Credit Score',
                        y='Default Risk',
                        title='Impact of Credit Score on Default Risk',
                        markers=True
                    )
                    fig_sens.add_hline(y=model.optimal_threshold, line_dash="dash", line_color="#1a4fa3",
                                      annotation_text="Approval Threshold")
                    fig_sens.add_vline(x=inputs['credit_score'], line_dash="dot", line_color="#2362c7",
                                      annotation_text="Current Score")
                    fig_sens.update_layout(height=400, yaxis_tickformat='.1%')
                    st.plotly_chart(fig_sens, use_container_width=True)
                
                # PDF Report Generation
                if export_pdf:
                    st.markdown("---")
                    st.subheader("DOWNLOAD PDF REPORT")
                    
                    try:
                        pdf_buffer = generate_pdf_report(
                            inputs=inputs,
                            prob=prob,
                            pred=pred,
                            risk=risk,
                            threshold=model.optimal_threshold
                        )
                        
                        st.download_button(
                            label="Download Credit Assessment Report (PDF)",
                            data=pdf_buffer,
                            file_name=f"credit_assessment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            type="primary"
                        )
                        st.success("PDF report ready for download!")
                    except Exception as e:
                        st.error(f"Error generating PDF: {e}")

                # SHAP explanation
                try:
                    # Add dummy target for transform
                    df_explain = df_in.copy()
                    if 'target' not in df_explain.columns:
                        df_explain['target'] = 0
                    X_explain, _ = processor.transform(df_explain, target_col='target')
                    explain_data = model.explain(X_explain)
                    if explain_data.get('summary_plot') is not None:
                        st.subheader("Feature Importance")
                        st.pyplot(explain_data['summary_plot'])
                except Exception as e:
                    pass  # Silently skip SHAP if not available

        except Exception as e:
            import traceback
            st.error(f"Could not score input: {e}")
            st.code(traceback.format_exc())

elif mode == "Batch Processing":
    st.header("BATCH CREDIT ASSESSMENT")
    st.write("Upload a CSV file containing multiple loan applications for bulk processing.")
    
    # Show expected format
    with st.expander("VIEW REQUIRED CSV FORMAT"):
        sample_df = pd.DataFrame({
            'age': [35, 42],
            'income': [50000, 75000],
            'credit_score': [650, 720],
            'credit_history_months': [60, 120],
            'existing_loans': [1, 2],
            'debt_to_income_ratio': [0.3, 0.25],
            'loan_amount': [20000, 35000],
            'employment_status': ['Employed', 'Self-Employed'],
            'housing_status': ['Rent', 'Own'],
            'loan_purpose': ['Personal', 'Home']
        })
        st.dataframe(sample_df, use_container_width=True)
    
    uploaded = st.file_uploader("UPLOAD CSV FILE", type=['csv'], help="Select a CSV file with applicant data")
    
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"File uploaded successfully: {len(df)} applications found")
            
            with st.spinner("Processing applications..."):
                results = assess_dataframe(df)
            
            # Summary statistics
            st.subheader("BATCH SUMMARY")
            col1, col2, col3, col4 = st.columns(4)
            
            total_apps = len(results)
            approved = (results['prediction'] == 0).sum()
            declined = (results['prediction'] == 1).sum()
            avg_risk = results['default_prob'].mean()
            
            with col1:
                st.metric("Total Applications", total_apps)
            with col2:
                st.metric("Approved", approved, f"{approved/total_apps*100:.1f}%")
            with col3:
                st.metric("Declined", declined, f"{declined/total_apps*100:.1f}%")
            with col4:
                st.metric("Avg. Default Risk", f"{avg_risk:.2%}")
            
            # Risk distribution chart
            st.subheader("RISK DISTRIBUTION")
            risk_counts = results['risk'].value_counts()
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Application Risk Categories",
                color=risk_counts.index,
                color_discrete_map={'Low': '#1a4fa3', 'Medium': '#2362c7', 'High': '#153b7c'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.subheader("DETAILED RESULTS")
            
            # Format the results for display
            display_results = results.copy()
            display_results['decision'] = display_results['prediction'].map({0: 'APPROVED', 1: 'DECLINED'})
            display_results['default_prob'] = display_results['default_prob'].apply(lambda x: f"{x:.2%}")
            
            # Reorder columns
            cols_order = ['decision', 'risk', 'default_prob'] + [c for c in display_results.columns if c not in ['decision', 'risk', 'default_prob', 'prediction']]
            display_results = display_results[cols_order]
            
            st.dataframe(display_results, use_container_width=True, height=400)
            
            # Download button
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="DOWNLOAD RESULTS (CSV)",
                data=csv,
                file_name='credit_assessment_results.csv',
                mime='text/csv',
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Failed to process uploaded file: {e}")
            import traceback
            with st.expander("View Error Details"):
                st.code(traceback.format_exc())

elif mode == "Scenario Analysis":
    st.header("SCENARIO ANALYSIS & SENSITIVITY TESTING")
    st.write("Compare multiple what-if scenarios to understand how changes in applicant profile affect credit decisions.")
    
    # Base scenario
    st.subheader("BASE SCENARIO")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        base_age = st.number_input("Age", 18, 75, 35, key="base_age")
        base_income = st.number_input("Annual Income ($)", 10000.0, 500000.0, 50000.0, 5000.0, key="base_income")
        base_credit_score = st.number_input("Credit Score", 300, 850, 650, key="base_credit")
    
    with col2:
        base_credit_history = st.number_input("Credit History (months)", 0, 360, 60, key="base_history")
        base_existing_loans = st.number_input("Existing Loans", 0, 10, 1, key="base_loans")
        base_debt_ratio = st.number_input("Debt-to-Income Ratio", 0.0, 1.0, 0.3, 0.01, key="base_debt")
    
    with col3:
        base_loan_amount = st.number_input("Loan Amount ($)", 1000.0, 500000.0, 20000.0, 1000.0, key="base_loan")
        base_employment = st.selectbox("Employment Status", ['Employed', 'Self-Employed', 'Unemployed', 'Retired'], key="base_emp")
        base_housing = st.selectbox("Housing Status", ['Own', 'Rent', 'Mortgage'], key="base_housing")
    
    base_loan_purpose = st.selectbox("Loan Purpose", ['Personal', 'Auto', 'Home', 'Education', 'Business'], key="base_purpose")
    
    if st.button("RUN SCENARIO ANALYSIS", use_container_width=True):
        base_inputs = {
            'age': base_age, 'income': base_income, 'credit_score': base_credit_score,
            'credit_history_months': base_credit_history, 'existing_loans': base_existing_loans,
            'debt_to_income_ratio': base_debt_ratio, 'loan_amount': base_loan_amount,
            'employment_status': base_employment, 'housing_status': base_housing,
            'loan_purpose': base_loan_purpose
        }
        
        # Create scenarios
        scenarios = {
            'Base Case': base_inputs.copy(),
            'Improved Credit (+50)': {**base_inputs, 'credit_score': min(base_credit_score + 50, 850)},
            'Higher Income (+20%)': {**base_inputs, 'income': base_income * 1.2},
            'Lower Debt Ratio (-10%)': {**base_inputs, 'debt_to_income_ratio': max(base_debt_ratio - 0.1, 0)},
            'Smaller Loan (-25%)': {**base_inputs, 'loan_amount': base_loan_amount * 0.75},
            'Worst Case': {**base_inputs, 'credit_score': max(base_credit_score - 100, 300), 'debt_to_income_ratio': min(base_debt_ratio + 0.2, 1.0)}
        }
        
        # Evaluate all scenarios
        scenario_results = []
        for name, scenario in scenarios.items():
            df_scenario = pd.DataFrame([scenario])
            scored = assess_dataframe(df_scenario)
            scenario_results.append({
                'Scenario': name,
                'Default Risk': scored['default_prob'].iloc[0],
                'Decision': 'APPROVED' if scored['prediction'].iloc[0] == 0 else 'DECLINED',
                'Risk Category': scored['risk'].iloc[0]
            })
        
        results_df = pd.DataFrame(scenario_results)
        
        # Visualization
        st.markdown("---")
        st.subheader("SCENARIO COMPARISON")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig = px.bar(
                results_df,
                x='Scenario',
                y='Default Risk',
                title='Default Risk by Scenario',
                color_discrete_sequence=['#0d6efd'],
                pattern_shape='Decision',
                pattern_shape_sequence=['', 'x']
            )
            fig.add_hline(y=model.optimal_threshold, line_dash="dash", line_color="#153b7c", 
                         annotation_text="Decision Threshold")
            fig.update_layout(height=400, yaxis_tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Waterfall chart showing impact
            impacts = []
            base_risk = results_df[results_df['Scenario'] == 'Base Case']['Default Risk'].iloc[0]
            for _, row in results_df.iterrows():
                if row['Scenario'] != 'Base Case':
                    impact = (row['Default Risk'] - base_risk) * 100
                    impacts.append({'Scenario': row['Scenario'], 'Impact': impact})
            
            if impacts:
                impact_df = pd.DataFrame(impacts)
                fig2 = px.bar(
                    impact_df,
                    x='Scenario',
                    y='Impact',
                    title='Risk Impact vs Base Case (percentage points)',
                    color='Impact',
                    color_continuous_scale=['#e9ecef', '#1a4fa3']
                )
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
        
        # Results table
        st.subheader("DETAILED SCENARIO RESULTS")
        results_df['Default Risk'] = results_df['Default Risk'].apply(lambda x: f"{x:.2%}")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("APPROVAL BOUNDARY HEATMAP")
        st.caption("Explore how two factors jointly affect default risk. Darker colors indicate higher risk.")

        # Select variables for heatmap
        var_options = {
            'Credit Score': 'credit_score',
            'Debt-to-Income Ratio': 'debt_to_income_ratio',
            'Annual Income': 'income',
            'Loan Amount': 'loan_amount',
            'Credit History (months)': 'credit_history_months'
        }

        hcol1, hcol2, hcol3 = st.columns([2,2,1])
        with hcol1:
            x_label = st.selectbox("X-Axis Variable", list(var_options.keys()), index=0, key="heat_x")
        with hcol2:
            y_label = st.selectbox("Y-Axis Variable", list(var_options.keys()), index=1, key="heat_y")
        with hcol3:
            grid_res = st.slider("Resolution", min_value=10, max_value=40, value=20, step=5, key="heat_res")

        x_var = var_options[x_label]
        y_var = var_options[y_label]

        # Build ranges around base inputs
        def var_range(var, base_val):
            bounds = {
                'credit_score': (300, 850),
                'income': (10000.0, 500000.0),
                'debt_to_income_ratio': (0.0, 1.0),
                'loan_amount': (1000.0, 500000.0),
                'credit_history_months': (0, 360)
            }
            low, high = bounds[var]
            # Create a window around base value
            if var in ('credit_score', 'credit_history_months'):
                span = (high - low) * 0.4
                lo = max(low, base_val - span/2)
                hi = min(high, base_val + span/2)
                return np.linspace(lo, hi, grid_res)
            if var in ('income', 'loan_amount'):
                # log scale spacing for monetary variables
                lo = max(low, base_val * 0.5)
                hi = min(high, base_val * 1.5)
                return np.linspace(lo, hi, grid_res)
            if var == 'debt_to_income_ratio':
                lo = max(low, base_val - 0.2)
                hi = min(high, base_val + 0.2)
                return np.linspace(lo, hi, grid_res)
            return np.linspace(low, high, grid_res)

        x_vals = var_range(x_var, base_inputs.get(x_var, 0))
        y_vals = var_range(y_var, base_inputs.get(y_var, 0))

        # Generate grid of scenarios
        rows = []
        for xv in x_vals:
            for yv in y_vals:
                s = base_inputs.copy()
                s[x_var] = float(xv)
                s[y_var] = float(yv)
                rows.append(s)
        grid_df = pd.DataFrame(rows)
        grid_scored = assess_dataframe(grid_df)
        Z = grid_scored['default_prob'].values.reshape(len(x_vals), len(y_vals))

        heat = go.Figure(data=go.Heatmap(
            z=Z.T,  # transpose to align axes
            x=x_vals,
            y=y_vals,
            colorscale=[[0, '#e9ecef'], [0.5, '#93c5fd'], [1, '#1a4fa3']],
            colorbar=dict(title='Default Risk', tickformat='.0%')
        ))
        heat.update_layout(
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=450,
            margin=dict(l=40, r=20, t=10, b=40)
        )
        st.plotly_chart(heat, use_container_width=True)

        st.markdown("---")
        st.subheader("MINIMUM CHANGES TO REACH APPROVAL")
        st.caption("Calculated one variable at a time while keeping all others fixed at the base scenario.")

        # Helper: binary search to find required value
        def find_min_adjustment(base, var, low, high, better_when='increase', max_iters=25):
            target = float(model.optimal_threshold)
            start = base[var]
            lo, hi = (start, high) if better_when == 'increase' else (low, start)
            # Check feasibility
            test = base.copy()
            test[var] = hi if better_when == 'increase' else lo
            feas = assess_dataframe(pd.DataFrame([test]))
            if feas['default_prob'].iloc[0] > target:
                return None  # not achievable within bounds
            # Binary search
            best = None
            for _ in range(max_iters):
                mid = (lo + hi) / 2
                test[var] = mid
                prob = assess_dataframe(pd.DataFrame([test]))['default_prob'].iloc[0]
                if prob <= target:
                    best = mid
                    if better_when == 'increase':
                        hi = mid
                    else:
                        lo = mid
                else:
                    if better_when == 'increase':
                        lo = mid
                    else:
                        hi = mid
            return best

        # Compute recommendations
        bounds = {
            'credit_score': (300, 850, 'increase', 'pts'),
            'income': (10000.0, 500000.0, 'increase', '$'),
            'debt_to_income_ratio': (0.0, 1.0, 'decrease', 'ratio'),
            'loan_amount': (1000.0, 500000.0, 'decrease', '$'),
            'credit_history_months': (0, 360, 'increase', 'months')
        }

        recs = []
        # Only compute if base case is declined; otherwise show margins
        base_prob = assess_dataframe(pd.DataFrame([base_inputs]))['default_prob'].iloc[0]
        declined = base_prob > float(model.optimal_threshold)
        for var, (lo, hi, direction, unit) in bounds.items():
            best = find_min_adjustment(base_inputs.copy(), var, lo, hi, better_when=direction)
            if best is not None:
                current = base_inputs[var]
                delta = best - current
                if unit == '$':
                    change = f"{delta:,.0f}"
                    target_val = f"{best:,.0f}"
                elif unit == 'pts':
                    change = f"{delta:.0f}"
                    target_val = f"{best:.0f}"
                elif unit == 'months':
                    change = f"{delta:.0f}"
                    target_val = f"{best:.0f}"
                else:
                    change = f"{delta:.2f}"
                    target_val = f"{best:.2f}"
                action = 'Increase' if direction == 'increase' else 'Reduce'
                recs.append({
                    'Factor': var.replace('_', ' ').title(),
                    'Action': action,
                    'New Value Needed': target_val,
                    'Change Required': change
                })
            else:
                recs.append({
                    'Factor': var.replace('_', ' ').title(),
                    'Action': ('Increase' if direction == 'increase' else 'Reduce'),
                    'New Value Needed': 'Not achievable within limits',
                    'Change Required': 'N/A'
                })

        recs_df = pd.DataFrame(recs)
        if declined:
            st.dataframe(recs_df, use_container_width=True, hide_index=True)
        else:
            # If already approved, show margins to threshold for information
            st.caption("Applicant is already approved. Displaying informational margins to the decision threshold.")
            st.dataframe(recs_df, use_container_width=True, hide_index=True)

elif mode == "Portfolio Dashboard":
    st.header("PORTFOLIO RISK DASHBOARD")
    st.write("Simulate and analyze a portfolio of loan applications with advanced risk metrics.")
    
    # Portfolio simulation controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        portfolio_size = st.slider("Portfolio Size", 10, 1000, 100)
    with col2:
        portfolio_type = st.selectbox("Portfolio Type", ["Balanced", "Conservative", "Aggressive"])
    with col3:
        simulate = st.button("GENERATE PORTFOLIO", use_container_width=True)
    
    if simulate:
        with st.spinner("Generating portfolio..."):
            # Generate synthetic portfolio based on type
            np.random.seed(42)
            
            if portfolio_type == "Conservative":
                credit_scores = np.random.normal(720, 50, portfolio_size).clip(650, 850)
                income = np.random.lognormal(11.0, 0.3, portfolio_size)
                debt_ratio = np.random.uniform(0.1, 0.3, portfolio_size)
            elif portfolio_type == "Aggressive":
                credit_scores = np.random.normal(620, 80, portfolio_size).clip(500, 750)
                income = np.random.lognormal(10.3, 0.6, portfolio_size)
                debt_ratio = np.random.uniform(0.3, 0.6, portfolio_size)
            else:  # Balanced
                credit_scores = np.random.normal(670, 70, portfolio_size).clip(550, 800)
                income = np.random.lognormal(10.7, 0.5, portfolio_size)
                debt_ratio = np.random.uniform(0.2, 0.5, portfolio_size)
            
            portfolio_df = pd.DataFrame({
                'age': np.random.randint(25, 65, portfolio_size),
                'income': income,
                'credit_score': credit_scores.astype(int),
                'credit_history_months': np.random.randint(12, 240, portfolio_size),
                'existing_loans': np.random.poisson(1.2, portfolio_size),
                'debt_to_income_ratio': debt_ratio,
                'loan_amount': np.random.lognormal(9.8, 0.5, portfolio_size),
                'employment_status': np.random.choice(['Employed', 'Self-Employed'], portfolio_size, p=[0.8, 0.2]),
                'housing_status': np.random.choice(['Own', 'Rent', 'Mortgage'], portfolio_size, p=[0.3, 0.3, 0.4]),
                'loan_purpose': np.random.choice(['Personal', 'Auto', 'Home'], portfolio_size, p=[0.4, 0.35, 0.25])
            })
            
            # Assess portfolio
            portfolio_results = assess_dataframe(portfolio_df)
            
            # Portfolio metrics
            st.markdown("---")
            st.subheader("PORTFOLIO SUMMARY")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            total_apps = len(portfolio_results)
            approved = (portfolio_results['prediction'] == 0).sum()
            declined = (portfolio_results['prediction'] == 1).sum()
            avg_risk = portfolio_results['default_prob'].mean()
            total_exposure = portfolio_results[portfolio_results['prediction'] == 0]['loan_amount'].sum()
            expected_loss = (portfolio_results[portfolio_results['prediction'] == 0]['default_prob'] * 
                           portfolio_results[portfolio_results['prediction'] == 0]['loan_amount']).sum()
            
            with col1:
                st.metric("Total Applications", f"{total_apps:,}")
            with col2:
                st.metric("Approved", f"{approved:,}", f"{approved/total_apps*100:.1f}%")
            with col3:
                st.metric("Average Risk", f"{avg_risk:.2%}")
            with col4:
                st.metric("Total Exposure", f"${total_exposure:,.0f}")
            with col5:
                st.metric("Expected Loss", f"${expected_loss:,.0f}", f"{expected_loss/total_exposure*100:.2f}%")
            
            # Risk distribution charts
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk distribution histogram
                fig = px.histogram(
                    portfolio_results,
                    x='default_prob',
                    nbins=30,
                    title='Portfolio Risk Distribution',
                    labels={'default_prob': 'Default Probability'},
                    color_discrete_sequence=['#0d6efd']
                )
                fig.add_vline(x=model.optimal_threshold, line_dash="dash", line_color="#153b7c",
                            annotation_text="Threshold")
                fig.update_layout(height=350, xaxis_tickformat='.1%')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Credit score vs default risk scatter
                approved_data = portfolio_results[portfolio_results['prediction'] == 0]
                fig2 = px.scatter(
                    approved_data,
                    x='credit_score',
                    y='default_prob',
                    size='loan_amount',
                    color_discrete_sequence=['#1a4fa3'],
                    symbol='risk',
                    title='Approved Applications: Credit Score vs Risk',
                    symbol_map={'Low': 'circle', 'Medium': 'square', 'High': 'x'}
                )
                fig2.update_layout(height=350, yaxis_tickformat='.1%')
                st.plotly_chart(fig2, use_container_width=True)
            
            # Risk by segment
            st.markdown("---")
            st.subheader("RISK ANALYSIS BY SEGMENT")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # By employment status
                emp_analysis = portfolio_results.groupby('employment_status').agg({
                    'default_prob': 'mean',
                    'prediction': lambda x: (x == 0).sum()
                }).reset_index()
                emp_analysis.columns = ['Employment Status', 'Avg Risk', 'Approved Count']
                
                fig3 = px.bar(
                    emp_analysis,
                    x='Employment Status',
                    y='Avg Risk',
                    title='Average Risk by Employment Status',
                    text='Approved Count',
                    color='Avg Risk',
                    color_continuous_scale=[[0, '#e9ecef'], [0.5, '#93c5fd'], [1, '#0d6efd']]
                )
                fig3.update_layout(height=300, yaxis_tickformat='.1%')
                st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                # By loan purpose
                purpose_analysis = portfolio_results.groupby('loan_purpose').agg({
                    'default_prob': 'mean',
                    'loan_amount': 'sum'
                }).reset_index()
                purpose_analysis.columns = ['Loan Purpose', 'Avg Risk', 'Total Exposure']
                
                fig4 = px.bar(
                    purpose_analysis,
                    x='Loan Purpose',
                    y='Total Exposure',
                    title='Exposure by Loan Purpose',
                    color='Avg Risk',
                    color_continuous_scale=[[0, '#e9ecef'], [0.5, '#93c5fd'], [1, '#0d6efd']]
                )
                fig4.update_layout(height=300)
                st.plotly_chart(fig4, use_container_width=True)
            
            # Download portfolio results
            st.markdown("---")
            csv = portfolio_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="DOWNLOAD PORTFOLIO ANALYSIS (CSV)",
                data=csv,
                file_name=f'portfolio_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
                use_container_width=True
            )


