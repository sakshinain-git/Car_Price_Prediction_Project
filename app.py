"""
app.py - Car Price Prediction System
Main Streamlit application entry point.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import datetime

# Page configuration
st.set_page_config(
    page_title="AutoValue AI | Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom modular imports
from src.utils.theme import inject_theme
from src.utils.formatters import fmt_num, fmt_inr_short
from src.models.trainer import auto_train_if_needed
from src.models.predictor import load_artifacts, load_dataset

from src.pages.dashboard import render_dashboard
from src.pages.predictor import render_predictor
from src.pages.comparison import render_comparison
from src.pages.analytics import render_analytics
from src.pages.intelligence import render_intelligence

def main():
    # Initialize theme state
    if "theme" not in st.session_state:
        st.session_state.theme = "Dark"

    # Inject CSS theme
    inject_theme(st.session_state.theme)
    
    # Auto-train if artifacts are missing
    auto_train_if_needed()
    
    # Load ML artifacts and dataset
    artifacts, le_dict, feature_cols, model_results, stats, reference_year = load_artifacts()
    df = load_dataset()
    model_ready = artifacts is not None and len(artifacts) > 0

    PAGES = ["🏠 Dashboard", "🔮 Price Predictor", "⚡ Car Comparison", "📊 Analytics Hub", "🤖 Model Intelligence"]
    if "page" not in st.session_state:
        st.session_state.page = PAGES[0]

    # ── SIDEBAR navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:20px 0 10px 0;">
          <div style="font-family:'Orbitron',monospace;font-size:1.4rem;font-weight:900;
                      background:linear-gradient(135deg,#00d4ff,#7b2ff7);-webkit-background-clip:text;
                      -webkit-text-fill-color:transparent;background-clip:text;">🚗 AUTOVALUE AI</div>
          <div style="font-size:0.65rem;color:rgba(0,212,255,0.5);letter-spacing:3px;text-transform:uppercase;margin-top:4px;">Car Intelligence System</div>
        </div><hr>""", unsafe_allow_html=True)

        sidebar_page = st.selectbox("📍 Navigate", PAGES,
                                    index=PAGES.index(st.session_state.page),
                                    label_visibility="collapsed")
        if sidebar_page != st.session_state.page:
            st.session_state.page = sidebar_page
            st.rerun()

        # Theme Selector
        theme_choice = st.selectbox(
            "🎨 Theme Mode",
            ["🌙 Dark Mode", "🌞 Light Mode"],
            index=0 if st.session_state.theme == "Dark" else 1
        )
        new_theme = "Dark" if "Dark" in theme_choice else "Light"
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        if not model_ready:
            st.markdown('<div class="warning-box">⚠️ Models not trained.<br>Run: <code>python model_trainer.py</code></div>', unsafe_allow_html=True)
        else:
            n_rec = stats.get('total_records', 0)
            p_mean = stats.get('price_mean', 0)
            st.markdown(
                f'<div class="info-box">✅ <strong>{len(artifacts)} models</strong> loaded<br>'
                f'📊 <strong>{fmt_num(n_rec)}</strong> training records<br>'
                f'💰 Avg price: <strong>{fmt_inr_short(p_mean)}</strong></div>', 
                unsafe_allow_html=True
            )
            
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.7rem;color:rgba(224,224,255,0.3);text-align:center;letter-spacing:1px;">'
                    'POWERED BY MACHINE LEARNING<br>Random Forest · XGBoost · GBM</div>', unsafe_allow_html=True)

    page = st.session_state.page

    # Route navigation to sub-pages
    if "Dashboard" in page:
        render_dashboard(df, stats, theme=st.session_state.theme)
    elif "Predictor" in page:
        render_predictor(artifacts, le_dict, feature_cols, model_results, stats, reference_year, theme=st.session_state.theme)
    elif "Comparison" in page:
        render_comparison(artifacts, le_dict, feature_cols, stats, reference_year, theme=st.session_state.theme)
    elif "Analytics" in page:
        render_analytics(df, stats, theme=st.session_state.theme)
    elif "Intelligence" in page:
        render_intelligence(artifacts, model_results, feature_cols, theme=st.session_state.theme)

    # Footer
    st.markdown('<div class="footer">🚗 AUTOVALUE AI · Powered by Machine Learning · Built with Python & Streamlit</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()