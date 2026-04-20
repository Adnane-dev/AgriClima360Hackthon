# src/utils.py
import streamlit as st
import plotly.graph_objects as go

def load_css():
    """Charge le CSS personnalisé pour l'application."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background: #f5f7f5;
        color: #1a2e24;
    }
    [data-testid="stSidebar"] {
        background: #eef3ee !important;
        border-right: 1px solid #c8d9c8;
    }
    [data-testid="stSidebar"] * { color: #1a3a2a !important; }
    
    h1 { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; color: #2d6a4f; }
    h2 { font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem; color: #40916c; }
    
    .tag {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        color: #2d6a4f;
        border: 1px solid #c8d9c8;
        padding: 2px 8px;
        border-radius: 3px;
        display: inline-block;
        margin-bottom: 12px;
    }
    .step-box {
        background: #ffffff;
        border: 1px solid #d0e6d0;
        border-left: 3px solid #2d6a4f;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 10px 0;
    }
    .warn-box {
        background: #fff8f0;
        border: 1px solid #ffd8b0;
        border-left: 3px solid #f4a261;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
    }
    .ok-box {
        background: #f0f9f0;
        border: 1px solid #c8e6c8;
        border-left: 3px solid #2d6a4f;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
    }
    .kpi {
        background: #ffffff;
        border: 1px solid #d0e6d0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .kpi-v {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        color: #2d6a4f;
    }
    .kpi-l {
        font-size: 11px;
        color: #6b9e7a;
        text-transform: uppercase;
    }
    </style>
    """, unsafe_allow_html=True)

def plot_theme(fig):
    """Applique le thème Plotly cohérent avec l'application."""
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8faf8",
        font=dict(family="IBM Plex Sans", color="#2d4a3a", size=12),
        xaxis=dict(gridcolor="#e0ece0", linecolor="#c8d9c8"),
        yaxis=dict(gridcolor="#e0ece0", linecolor="#c8d9c8"),
        title=dict(font=dict(family="IBM Plex Mono", color="#2d6a4f", size=14)),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", font=dict(color="#2d4a3a"))
    )
    return fig