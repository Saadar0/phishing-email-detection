#app.py
import streamlit as st
import pandas as pd
import joblib
import time
from utils import TextPreprocessor

# 1. Page Configuration & Theme Optimization
st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# 2. Define custom UI styles
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #0e1117; }

    /* Custom Glassmorphism Card */
    .forensic-card {
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #30363d;
        background: rgba(22, 27, 34, 0.7);
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    /* Header Styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#58a6ff, #1f6feb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Badge/Chip Styling */
    .signal-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 15px;
        background: #21262d;
        border: 1px solid #30363d;
        color: #8b949e;
        font-size: 0.85rem;
        margin: 2px;
    }
    </style>
    """, unsafe_allow_html=True)


# 3. Resource Loading
@st.cache_resource
def load_artifacts():
    return (
        joblib.load('text_preprocessor2.pkl'),
        joblib.load('col_preprocessor2.pkl'),
        joblib.load('phishing_voting_model2.pkl'),
        joblib.load('selected_features2.pkl'),
        joblib.load('best_threshold2.pkl')
    )


try:
    preprocessor, cp, model, selected_feats, threshold = load_artifacts()
except Exception as e:
    st.error(f"System Offline: Ensure artifacts exist. Error: {e}")
    st.stop()

# 4. Sidebar - System Intel
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.markdown("### **System Status**")
    st.success("🟢 AI Core Active")
    st.info(f"🎯 Threshold: `{threshold:.2f}`")
    st.divider()
    st.markdown("### **Forensic Modules**")
    st.caption("✅ Lexical Pattern Matcher")
    st.caption("✅ Mixed-Alpha Scanner")
    st.caption("✅ Ensemble Voting")

# 5. Main Dashboard Layout
st.markdown('<h1 class="main-title">🛡️ PhishGuard AI</h1>', unsafe_allow_html=True)
st.markdown("---")

# Input Area
with st.container():
    email_input = st.text_area(
        "📩 **Suspicious Communication Content**",
        height=250,
        placeholder="Paste email or message body here for deep forensic scanning...",
        help="Our AI analyzes over 9,000 text features including urgency, technical tricks, and keyword signals."
    )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        audit_trigger = st.button("🚀 INITIATE SCAN", use_container_width=True, type="primary")

# 6. Audit Execution & Visual Results
if audit_trigger:
    if not email_input.strip():
        st.toast("Input required for scan.", icon="🛑")
    else:
        # Visual Loading Sequence
        with st.status("🔍 Scanning for Malicious Vectors...", expanded=False) as status:
            cleaned = preprocessor.transform([email_input])[0]
            raw_feats, highlights, mixed_words = preprocessor.extract_features(cleaned, email_input,
                                                                               return_ui_data=True)

            num_df = pd.DataFrame([raw_feats])[selected_feats]
            input_data = pd.concat([pd.Series([cleaned], name='processed_text'), num_df], axis=1)
            proba = model.predict_proba(cp.transform(input_data))[0][1]

            time.sleep(0.4)  # Aesthetic delay for "processing" feel
            status.update(label="Forensic Analysis Complete", state="complete")

        # Top-Level Risk Assessment
        st.markdown("### 📊 Threat Level Assessment")
        risk_percent = proba * 100

        # Determine Color Based on Risk
        if proba >= 0.75:
            risk_color = "#f85149"  # Crimson Red
            risk_label = "CRITICAL RISK"
        elif proba >= threshold:
            risk_color = "#dbab09"  # Warning Gold
            risk_label = "SUSPICIOUS ACTIVITY"
        else:
            risk_color = "#3fb950"  # Security Green
            risk_label = "LIKELY LEGITIMATE"

        # Large Risk Meter
        st.markdown(f"""
            <div style="background: {risk_color}22; padding: 20px; border-radius: 10px; border-left: 5px solid {risk_color};">
                <h2 style="color: {risk_color}; margin: 0;">{risk_label}: {risk_percent:.1f}%</h2>
            </div>
        """, unsafe_allow_html=True)
        st.progress(proba)

        # Detailed Breakdown
        st.write("##")
        col_main, col_stats = st.columns([2, 1])

        with col_main:
            st.markdown('<div class="forensic-card">', unsafe_allow_html=True)
            st.markdown("### 🔍 Forensic Signals Found")

            found_any = False
            # Loop through all highlights from utils.py
            for category, words in highlights.items():
                if words:
                    found_any = True
                    # Use unique colors for different categories
                    st.markdown(
                        f"**{category}:** " + " ".join([f'<span class="signal-badge">{w}</span>' for w in words]),
                        unsafe_allow_html=True)

            if mixed_words:
                found_any = True
                st.warning(f"⚠️ **Mixed Alphanumeric (Leetspeak):** {', '.join([f'`{w}`' for w in mixed_words])}")

            if not found_any:
                st.info("No common phishing keywords detected. Classification based on structural features.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_stats:
            st.markdown('<div class="forensic-card">', unsafe_allow_html=True)
            st.markdown("### ⚙️ Technical Check")

            # Simplified "Vocabulary Variety" (lexical_diversity)
            st.write(f"• **Repetitiveness:** `{100 - (raw_feats.get('lexical_diversity', 0) * 100):.0f}%`")
            st.caption("High repetitiveness is common in robotic, automated spam.")

            # Simplified "Urgent Shouting" (caps_word_ratio)
            st.write(f"• **Shouting Level:** `{raw_feats.get('caps_word_ratio', 0):.1%}`")
            st.caption("How much of the text is in ALL CAPS to scare you.")

            # Simplified "Link Density" (url_count)
            st.write(f"• **Links Found:** `{raw_feats.get('url_count', 0)}` detected")

            # Simplified "Excitement Level" (exclamation_count)
            st.write(f"• **Panic Level:** `{raw_feats.get('exclamation_count', 0)}` exclamations")
            st.caption("Phishers use lots of '!!!' to make you act fast.")

            st.markdown('</div>', unsafe_allow_html=True)

        # Full Data View
        with st.expander("🛠️ View Raw Model Features"):
            st.dataframe(pd.DataFrame([raw_feats]).T, use_container_width=True)