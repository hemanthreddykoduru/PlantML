"""
app.py
------
Streamlit web application for the Plant Identification & Benefit Prediction System.

Features:
  ✅ Upload plant image (JPG / PNG / WEBP)
  ✅ Display uploaded image with a clean card layout
  ✅ Predict plant class with confidence percentage
  ✅ Display full plant info (scientific name, benefits, uses, Vrikshayurveda)
  ✅ Loading spinner during model inference
  ✅ Error handling for invalid uploads
  ✅ Responsive, modern dark-themed UI via custom CSS

Run:
  streamlit run app.py
"""

import os
import json
import time
import numpy as np
from PIL import Image

import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌿 Plant Identifier — Vrikshayurveda AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark Premium Theme
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ─────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0d1117;
        color: #e6edf3;
    }
    .stApp { background: linear-gradient(135deg, #0d1117, #161b22); }

    /* ── Hero header ─────────────────────────────────────────── */
    .hero {
        background: linear-gradient(135deg, #1a4731 0%, #0d6832 50%, #145a32 100%);
        border-radius: 18px;
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,255,100,0.08);
    }
    .hero h1 {
        font-family: 'Playfair Display', serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: #82f5b2;
        margin-bottom: 0.5rem;
    }
    .hero p {
        font-size: 1.1rem;
        color: #a7f3cc;
        font-weight: 300;
        letter-spacing: 0.03em;
    }

    /* ── Cards ───────────────────────────────────────────────── */
    .card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: box-shadow 0.2s;
    }
    .card:hover { box-shadow: 0 6px 28px rgba(0,255,100,0.08); }

    /* ── Prediction badge ────────────────────────────────────── */
    .pred-badge {
        display: inline-block;
        background: linear-gradient(90deg, #1e6b3e, #27a85a);
        color: white;
        font-size: 1.05rem;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
        border-radius: 30px;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .confidence-bar-wrap {
        background: #21262d;
        border-radius: 8px;
        height: 12px;
        margin: 0.6rem 0 0.3rem;
        overflow: hidden;
    }
    .confidence-bar {
        background: linear-gradient(90deg, #27a85a, #82f5b2);
        height: 100%;
        border-radius: 8px;
        transition: width 0.8s ease;
    }

    /* ── Benefit pill tags ───────────────────────────────────── */
    .tag {
        display: inline-block;
        background: #1c3a28;
        color: #82f5b2;
        font-size: 0.8rem;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        margin: 0.2rem;
        border: 1px solid #2a6b3e;
    }

    /* ── Section headings ────────────────────────────────────── */
    .section-heading {
        font-size: 1rem;
        font-weight: 600;
        color: #82f5b2;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        border-left: 3px solid #27a85a;
        padding-left: 0.6rem;
    }

    /* ── Sidebar ─────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #161b22 !important;
        border-right: 1px solid #21262d;
    }

    /* ── Upload area ─────────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        border: 2px dashed #27a85a !important;
        border-radius: 12px;
        background: rgba(39,168,90,0.04);
    }

    /* ── Divider ─────────────────────────────────────────────── */
    hr { border-color: #21262d; }

    /* ── Footer ──────────────────────────────────────────────── */
    .footer {
        text-align: center;
        color: #484f58;
        font-size: 0.82rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #21262d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS (after page config)
# ──────────────────────────────────────────────────────────────────────────────
from utils import (
    CLASS_NAMES,
    PLANT_INFO,
    preprocess_image,
    predict_plant,
    get_plant_info,
    load_experiments,
    save_experiment,
    analyze_nlp_proof
)

import pandas as pd
import plotly.express as px

MODEL_PATH   = os.path.join("model", "plant_model.h5")
CLASSES_PATH = os.path.join("model", "class_names.json")

# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADER (cached so it loads once per session)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the Keras model and class names from disk."""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)

        # Override class names with saved order (if available)
        if os.path.isfile(CLASSES_PATH):
            with open(CLASSES_PATH) as f:
                class_names = json.load(f)
        else:
            class_names = CLASS_NAMES

        return model, class_names, None   # (model, class_names, error)
    except Exception as exc:
        return None, CLASS_NAMES, str(exc)


# ──────────────────────────────────────────────────────────────────────────────
# HELPER — CONFIDENCE BAR HTML
# ──────────────────────────────────────────────────────────────────────────────

def confidence_bar_html(confidence: float) -> str:
    pct = int(confidence * 100)
    color = "#27a85a" if pct >= 70 else "#f0a500" if pct >= 40 else "#e05252"
    return f"""
    <div class="confidence-bar-wrap">
      <div class="confidence-bar" style="width:{pct}%; background:linear-gradient(90deg,{color},{color}aa);"></div>
    </div>
    <small style="color:#8b949e;">Confidence: <b style="color:{color};">{pct}%</b></small>
    """


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌿 Plant Identifier")
    st.markdown("---")
    
    # ── Navigation Selection ──
    page = st.radio("📌 Navigation", ["🌿 Identify Plant", "📝 Log Experiment", "📊 Validation Dashboard", "💬 Ask Vrikshayurveda AI"])
    
    st.markdown("---")
    st.markdown("### 📋 Supported Plants")
    for cls in CLASS_NAMES:
        info = get_plant_info(cls)
        st.markdown(f"- **{info['common_name']}** _{info['scientific_name']}_")
    st.markdown("---")
    st.markdown("### ⚙️ Model Info")
    st.markdown("- **Architecture**: MobileNetV2")
    st.markdown("- **Input size**: 224 × 224 px")
    st.markdown("- **Transfer Learning**: ImageNet")
    st.markdown("- **Framework**: TensorFlow / Keras")
    st.markdown("---")
    st.markdown("---")
    st.markdown(
        "<small style='color:#484f58;'>Built for Academic Internship Demo<br>© 2026 PlantML Project</small>",
        unsafe_allow_html=True,
    )

# (Navigation moved to top of sidebar)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE 2: LOG EXPERIMENT
# ──────────────────────────────────────────────────────────────────────────────
if page == "📝 Log Experiment":
    st.markdown("## 📝 Log Vrikshayurveda Experiment")
    st.write("Record real-world applications of traditional agricultural practices for experimental validation.")
    
    with st.form("experiment_form"):
        col1, col2 = st.columns(2)
        plant_name = col1.selectbox("Target Plant", CLASS_NAMES)
        soil_type = col2.selectbox("Soil Type", ["Clay", "Sandy", "Loam", "Sandy Loam", "Silt", "Peat"])
        
        treatment = st.text_input("Vrikshayurveda Treatment Applied", placeholder="e.g. Neem leaf slurry for pest control")
        duration = st.number_input("Observation Duration (Days)", min_value=1, max_value=365, value=14)
        
        st.info("🤖 **AI LLM Verification**: The final outcome will be audited and computationally concluded by the intelligent OpenRouter AI Engine by reading your descriptive notes.")
        notes = st.text_area("Detailed Observational Evidence (Proof of Efficacy)", placeholder="e.g. The neem cake completely eliminated the nematodes resulting in vibrantly green, healthy foliage.")
        
        submitted = st.form_submit_button("✅ Audit & Submit Evidence via AI", type="primary")
        
        if submitted:
            if treatment and notes:
                # Capture API key securely from local state
                api_key = st.secrets.get("OPENROUTER_API_KEY", st.session_state.get("open_router_key", None))
                
                with st.spinner("🤖 AI Agent is auditing your text using scientific LLM comprehension..."):
                    nlp_res = analyze_nlp_proof(notes, api_key)
                    
                save_experiment(plant_name, soil_type, treatment, duration, nlp_res['outcome'], notes, nlp_res['score'])
                
                if nlp_res.get("ai_verified"):
                    st.success(f"**🤖 Verified by AI Copilot!** The LLM confidently classified this treatment as a **{nlp_res['outcome']}**.")
                    st.info(f"**💡 AI Audit Reasoning:** {nlp_res['reasoning']}")
                else:
                    st.warning(f"⚡ Offline Math Fallback Mode Used! Computed Outcome: **{nlp_res['outcome']}** ({nlp_res['positivity']:.1f}% positive metric).")
            else:
                st.error("⚠️ Please specify the treatment and provide textual evidence.")
                
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# PAGE 3: VALIDATION DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────
elif page == "📊 Validation Dashboard":
    st.markdown("## 📊 Experimental Validation Dashboard")
    st.write("Statistical analysis of all crowdsourced Vrikshayurveda experimental validations.")
    
    experiments = load_experiments()
    if not experiments:
        st.info("No experiments logged yet. Go to 'Log Experiment' to add data.")
        st.stop()
        
    df = pd.DataFrame(experiments)
    
    # KPIs
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Validations Logged", len(df))
    success_rate = (len(df[df['outcome'] == 'Success']) / len(df)) * 100
    kpi2.metric("Overall Success Efficacy", f"{success_rate:.1f}%")
    kpi3.metric("Avg. Observation Duration", f"{df['duration_days'].mean():.1f} days")
    
    st.markdown("---")
    
    colA, colB = st.columns(2)
    with colA:
        # Pie chart of outcomes
        fig1 = px.pie(df, names='outcome', title='Experimental Outcomes', 
                     color='outcome', color_discrete_map={'Success':'#27a85a', 'Partial Success':'#f0a500', 'Failure':'#e05252', 'Ongoing':'#58a6ff'})
        fig1.update_layout(plot_bgcolor='#0d1117', paper_bgcolor='#0d1117', font_color='#e6edf3')
        st.plotly_chart(fig1, use_container_width=True)
        
    with colB:
        # Bar chart of treatments per plant
        fig2 = px.histogram(df, x='plant_name', color='outcome', title='Validations by Plant Target', barmode='group',
                           color_discrete_map={'Success':'#27a85a', 'Partial Success':'#f0a500', 'Failure':'#e05252'})
        fig2.update_layout(plot_bgcolor='#0d1117', paper_bgcolor='#0d1117', font_color='#e6edf3')
        st.plotly_chart(fig2, use_container_width=True)
        
    st.markdown('<div class="section-heading">NLP-Validated Raw Logs</div>', unsafe_allow_html=True)
    
    # Format the timestamp to be human readable
    display_df = df[['timestamp', 'plant_name', 'treatment', 'outcome', 'nlp_score', 'notes']].copy()
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp'], format='ISO8601').dt.strftime('%d %b %Y, %H:%M')
    
    st.dataframe(display_df, use_container_width=True)
    
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# PAGE 4: VRIKSHAYURVEDA AI COPILOT
# ──────────────────────────────────────────────────────────────────────────────
elif page == "💬 Ask Vrikshayurveda AI":
    st.markdown("## 💬 Vrikshayurveda AI Copilot")
    st.write("Ask our supercharged AI anything about agricultural practices, traditional texts, or biological crop treatments using its vast internet knowledge!")
    
    # ── API Key Management ──
    api_key = None
    if "OPENROUTER_API_KEY" in st.secrets:
        api_key = st.secrets["OPENROUTER_API_KEY"]
    else:
        api_key = st.session_state.get("open_router_key", "")
        if not api_key:
            st.warning("⚠️ Internet Knowledge connection requires an OpenRouter API Key.")
            key_input = st.text_input("Enter your API Key securely:", type="password")
            st.markdown("[Get a free OpenRouter connection key here](https://openrouter.ai/keys)")
            if st.button("Connect to Internet Knowledge", type="primary"):
                if key_input:
                    st.session_state["open_router_key"] = key_input
                    st.rerun()
                else:
                    st.error("Please enter a valid key.")
            st.stop()
            
    # ── AI Chat Interface ──
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
    except ImportError:
        st.error("openai module is missing. Please restart Streamlit.")
        st.stop()
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Namaste! I am your Vrikshayurveda AI Copilot embedded with vast internet knowledge. What agricultural or medicinal questions can I answer for you today?"}
        ]
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    prompt = st.chat_input("Ask about crops, soils, or ancient texts...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            try:
                # Build chat history for memory
                chat_history = [{"role": "system", "content": "You are a master of Vrikshayurveda, modern agriculture, and internet knowledge. Provide short, brilliant answers."}]
                for m in st.session_state.messages:
                    chat_history.append({"role": m["role"], "content": m["content"]})
                chat_history.append({"role": "user", "content": prompt})
                
                # Fetch intelligence from OpenRouter
                completion = client.chat.completions.create(
                    model="google/gemini-2.5-flash",
                    messages=chat_history,
                    max_tokens=1500
                )
                answer = completion.choices[0].message.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"❌ AI Connection Error. Did you enter the correct OpenRouter API key? Details: {e}")
                
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# PAGE 1: IDENTIFY PLANT (Existing code flows naturally here)
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero">
        <h1>🌿 Plant Identification & Benefit Prediction</h1>
        <p>Powered by MobileNetV2 Transfer Learning &amp; Vrikshayurveda Knowledge Base</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────────────────────────────────────

model, class_names, model_error = load_model()

if model_error:
    st.warning(
        f"⚠️ Trained model not found at `{MODEL_PATH}`. "
        "Please run `python train.py` first to train and save the model.\n\n"
        f"_Technical detail: {model_error}_"
    )
    st.info(
        "**Demo Mode**: You can still explore the plant knowledge base below. "
        "Upload an image once the model is trained."
    )
    model = None   # Continue in demo mode


# ──────────────────────────────────────────────────────────────────────────────
# MAIN — TWO COLUMNS: UPLOAD | RESULTS
# ──────────────────────────────────────────────────────────────────────────────

col_upload, col_results = st.columns([1, 1.4], gap="large")

# ── LEFT COLUMN: IMAGE UPLOAD ─────────────────────────────────────────────────
with col_upload:
    st.markdown('<div class="section-heading">📤 Upload Plant Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="Choose an image file",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        # Validate the file is a real image
        try:
            image = Image.open(uploaded_file)
            image.verify()                    # Check image integrity
            image = Image.open(uploaded_file) # Re-open after verify (seek resets)
        except Exception:
            st.error("❌ Invalid image file. Please upload a valid JPG, PNG, or WEBP image.")
            image = None

        if image is not None:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.markdown(
                f"<small style='color:#8b949e;'>Format: {image.format or 'Unknown'} · "
                f"Size: {image.size[0]}×{image.size[1]} px</small>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # ── PREDICT BUTTON ─────────────────────────────────────────────
            if model is not None:
                predict_btn = st.button("🔍 Identify Plant", type="primary", use_container_width=True)
            else:
                predict_btn = False
                st.button("🔍 Identify Plant", disabled=True, use_container_width=True,
                          help="Train the model first (python train.py)")
    else:
        image = None
        predict_btn = False
        # Placeholder card
        st.markdown(
            """
            <div class="card" style="text-align:center; padding: 3rem 1rem;">
              <div style="font-size:3.5rem;">🌱</div>
              <p style="color:#8b949e; margin-top:1rem;">
                Upload a clear photo of a plant leaf or the whole plant.<br>
                Supported: JPG, PNG, WEBP
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        predict_btn = False


# ── RIGHT COLUMN: PREDICTION RESULTS ─────────────────────────────────────────
with col_results:
    st.markdown('<div class="section-heading">🌿 Identification Results</div>', unsafe_allow_html=True)

    if predict_btn and image is not None and model is not None:
        with st.spinner("🔬 Analysing plant image..."):
            time.sleep(0.4)   # Brief delay for UX feel
            try:
                class_name, confidence = predict_plant(model, image, class_names)
                info = get_plant_info(class_name)
                st.session_state["last_result"] = (class_name, confidence, info)
            except Exception as exc:
                st.error(f"❌ Prediction error: {exc}")
                st.session_state.pop("last_result", None)

    # Display results (from session state so they persist between reruns)
    if "last_result" in st.session_state:
        class_name, confidence, info = st.session_state["last_result"]

        # ── Prediction summary ──────────────────────────────────────────────
        st.markdown(f'<div class="pred-badge">🌿 {info["common_name"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<i style="color:#8b949e;">_{info["scientific_name"]}_</i>', unsafe_allow_html=True)
        st.markdown(confidence_bar_html(confidence), unsafe_allow_html=True)

        high, med, low = confidence >= 0.7, 0.4 <= confidence < 0.7, confidence < 0.4
        if high:
            st.success("✅ High confidence prediction")
        elif med:
            st.warning("⚠️ Moderate confidence — verify visually")
        else:
            st.error("❌ Low confidence — try a clearer image")

        st.markdown("---")

        # ── Plant description ───────────────────────────────────────────────
        st.markdown('<div class="section-heading">📖 About this Plant</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="card"><p style="line-height:1.7">{info["description"]}</p>'
            f'<small style="color:#8b949e;">Family: <b>{info.get("family","—")}</b></small></div>',
            unsafe_allow_html=True,
        )

        # ── Benefits (pill tags) ────────────────────────────────────────────
        st.markdown('<div class="section-heading">✨ Key Benefits</div>', unsafe_allow_html=True)
        tags_html = "".join(f'<span class="tag">{b}</span>' for b in info["benefits"])
        st.markdown(f'<div class="card">{tags_html}</div>', unsafe_allow_html=True)

        # ── Medicinal uses ──────────────────────────────────────────────────
        st.markdown('<div class="section-heading">💊 Medicinal Uses</div>', unsafe_allow_html=True)
        uses_html = "".join(f"<li style='margin-bottom:4px'>{u}</li>" for u in info["medicinal_uses"])
        st.markdown(
            f'<div class="card"><ul style="padding-left:1.2rem; color:#c9d1d9">{uses_html}</ul></div>',
            unsafe_allow_html=True,
        )

        # ── Agricultural uses ───────────────────────────────────────────────
        st.markdown('<div class="section-heading">🌾 Agricultural Uses</div>', unsafe_allow_html=True)
        ag_html = "".join(f"<li style='margin-bottom:4px'>{u}</li>" for u in info["agricultural_uses"])
        st.markdown(
            f'<div class="card"><ul style="padding-left:1.2rem; color:#c9d1d9">{ag_html}</ul></div>',
            unsafe_allow_html=True,
        )

        # ── Vrikshayurveda ──────────────────────────────────────────────────
        st.markdown('<div class="section-heading">📜 Vrikshayurveda Reference</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="card" style="border-left:3px solid #c9a227; background:#1a1710;">'
            f'<p style="color:#e8d5a0; font-style:italic; line-height:1.8">'
            f'🌿 {info["vrikshayurveda"]}</p></div>',
            unsafe_allow_html=True,
        )

    else:
        # Placeholder when no prediction yet
        st.markdown(
            """
            <div class="card" style="text-align:center; padding: 3.5rem 1rem; color:#8b949e;">
              <div style="font-size:3rem;">🔍</div>
              <p style="margin-top:1rem;">
                Upload a plant image and click <b>Identify Plant</b><br>
                to see results here.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# EXPLORE SECTION — Browse all plants in the knowledge base
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## 🌍 Explore the Plant Knowledge Base")
st.markdown("Browse all 10 plants in our database, even without uploading an image.")

cols = st.columns(5)
for idx, (cls, col) in enumerate(zip(CLASS_NAMES, cols * 2)):
    info = get_plant_info(cls)
    if idx < len(CLASS_NAMES):
        col = cols[idx % 5]
        with col:
            with st.expander(f"🌿 {info['common_name']}"):
                st.markdown(f"**Scientific**: _{info['scientific_name']}_")
                st.markdown(f"**Family**: {info.get('family','—')}")
                st.markdown("**Top Benefits:**")
                for b in info["benefits"][:3]:
                    st.markdown(f"- {b}")

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer">
      🌿 Plant Identification & Benefit Prediction System &nbsp;|&nbsp;
      Built with TensorFlow/Keras + Streamlit &nbsp;|&nbsp;
      Academic Internship Project &nbsp;|&nbsp; 2025
    </div>
    """,
    unsafe_allow_html=True,
)
