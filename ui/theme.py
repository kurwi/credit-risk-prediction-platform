from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any
import streamlit as st


DEFAULT_TOKENS: Dict[str, Any] = {
    "brand_color": "#1a4fa3",
    "brand_light": "#2362c7",
    "brand_dark": "#153b7c",
    "text_color": "#1e293b",
    "muted_text": "#64748b",
    "bg": "#f8fafc",
    "bg_alt": "#f1f5f9",
    "surface": "#ffffff",
    "border": "#cbd5e1",
    "border_accent": "#93c5fd",
    "radius": 2,
    "font_family": "-apple-system, Segoe UI, Roboto, Arial, sans-serif",
    "max_width": 1400,
}


@dataclass
class ThemeTokens:
    brand_color: str = DEFAULT_TOKENS["brand_color"]
    brand_light: str = DEFAULT_TOKENS["brand_light"]
    brand_dark: str = DEFAULT_TOKENS["brand_dark"]
    text_color: str = DEFAULT_TOKENS["text_color"]
    muted_text: str = DEFAULT_TOKENS["muted_text"]
    bg: str = DEFAULT_TOKENS["bg"]
    bg_alt: str = DEFAULT_TOKENS["bg_alt"]
    surface: str = DEFAULT_TOKENS["surface"]
    border: str = DEFAULT_TOKENS["border"]
    border_accent: str = DEFAULT_TOKENS["border_accent"]
    radius: int = DEFAULT_TOKENS["radius"]
    font_family: str = DEFAULT_TOKENS["font_family"]
    max_width: int = DEFAULT_TOKENS["max_width"]

    @classmethod
    def load(cls, path: str | None) -> "ThemeTokens":
        data: Dict[str, Any] = {}
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        merged = {**DEFAULT_TOKENS, **data}
        return cls(**merged)

    def inject_css(self) -> None:
        tokens = asdict(self)
        css = f"""
        <style>
        :root {{
            --brand: {tokens['brand_color']};
            --brand-light: {tokens['brand_light']};
            --brand-dark: {tokens['brand_dark']};
            --text: {tokens['text_color']};
            --muted: {tokens['muted_text']};
            --bg: {tokens['bg']};
            --bg-alt: {tokens['bg_alt']};
            --surface: {tokens['surface']};
            --border: {tokens['border']};
            --border-accent: {tokens['border_accent']};
            --radius: {tokens['radius']}px;
            --font: {tokens['font_family']};
            --maxw: {tokens['max_width']}px;
        }}

        .block-container {{ max-width: var(--maxw); padding-top: 1rem; padding-bottom: 3rem; }}
        
        /* Dark mode support - detect system preference */
        @media (prefers-color-scheme: dark) {{
            :root {{
                --bg: #0f172a;
                --bg-alt: #1e293b;
                --surface: #1e293b;
                --text: #e2e8f0;
                --muted: #94a3b8;
                --border: #334155;
                --border-accent: #475569;
            }}
            body, .reportview-container {{ 
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%) !important;
                color: var(--text) !important;
            }}
        }}
        
        body, .reportview-container {{ 
            background: linear-gradient(135deg, #f8fafc 0%, #e0f2fe 50%, #f1f5f9 100%);
            color: var(--text); font-family: var(--font); 
        }}

        /* Hero section styling - sharp & modern */
        .hero-section {{
            display: flex;
            align-items: center;
            gap: 2.5rem;
            padding: 2.5rem 0;
            margin-bottom: 1rem;
            animation: fadeIn 0.6s ease-in;
            border-bottom: 3px solid var(--brand);
        }}
        
        .hero-icon {{
            font-size: 5rem;
            background: linear-gradient(135deg, var(--brand-light), var(--brand-dark));
            padding: 2rem;
            border-radius: 0;
            box-shadow: 0 12px 48px rgba(26,79,163,0.3),
                        inset 0 0 0 2px rgba(255,255,255,0.1);
            animation: float 3s ease-in-out infinite;
            border: 3px solid var(--brand-dark);
        }}
        
        .hero-content {{
            flex: 1;
        }}
        
        .hero-title {{
            font-size: 3.5rem !important;
            font-weight: 900 !important;
            background: linear-gradient(135deg, var(--brand-light) 0%, var(--brand-dark) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0 !important;
            padding: 0 !important;
            letter-spacing: -1.5px;
            line-height: 1.1;
            text-transform: uppercase;
        }}
        
        .hero-subtitle {{
            font-size: 1.4rem;
            color: var(--muted);
            margin: 0.8rem 0 0 0;
            font-weight: 600;
            letter-spacing: 0.3px;
        }}
        
        .hero-badge {{
            display: inline-block;
            background: linear-gradient(90deg, var(--brand-light), var(--brand-dark));
            color: white;
            padding: 0.5rem 1.2rem;
            border-radius: 0;
            font-size: 0.9rem;
            font-weight: 700;
            margin-top: 1.2rem;
            box-shadow: 0 6px 16px rgba(26,79,163,0.35);
            letter-spacing: 1px;
            text-transform: uppercase;
            border: 2px solid var(--brand-dark);
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(-20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        @keyframes float {{
            0%, 100% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-10px); }}
        }}
        
        .main-header {{
            font-size: 2.5rem; font-weight: 700; 
            background: linear-gradient(90deg, var(--brand-light), var(--brand-dark));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: left; padding: 1rem 0; 
            border-bottom: 3px solid var(--brand); 
            margin-bottom: 1.5rem; letter-spacing: -0.5px;
        }}
        
        /* Sharp cards with strong visual hierarchy */
        .stMetric {{ 
            background: linear-gradient(135deg, var(--surface) 0%, #e0f2fe 100%);
            padding: 1.5rem;
            border-radius: 0 !important;
            border-left: 5px solid var(--brand) !important;
            border-top: 2px solid var(--border-accent);
            border-right: 2px solid var(--border-accent);
            border-bottom: 2px solid var(--border-accent);
            box-shadow: 0 4px 12px rgba(0,0,0,0.08),
                        0 2px 4px rgba(26,79,163,0.1);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        .stMetric:hover {{ 
            box-shadow: 0 16px 40px rgba(26,79,163,0.2),
                        0 8px 16px rgba(0,0,0,0.1);
            border-left-width: 7px !important;
            border-left-color: var(--brand-dark) !important;
            transform: translateX(4px) translateY(-4px);
        }}
        
        .metric-card {{ 
            background: linear-gradient(135deg, var(--surface) 0%, #e0f2fe 100%);
            padding: 2rem;
            border-radius: 0;
            color: var(--text);
            text-align: left; 
            border-left: 5px solid var(--brand);
            border-top: 2px solid var(--border-accent);
            border-right: 2px solid var(--border-accent);
            border-bottom: 2px solid var(--border-accent);
            box-shadow: 0 4px 12px rgba(0,0,0,0.08),
                        0 2px 4px rgba(26,79,163,0.1);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        .metric-card:hover {{ 
            box-shadow: 0 16px 40px rgba(26,79,163,0.2),
                        0 8px 16px rgba(0,0,0,0.1);
            border-left-width: 7px;
            border-left-color: var(--brand-dark);
            transform: translateX(4px) translateY(-4px);
        }}
        
        /* Sharp, powerful button design */
        .stButton>button, .stFormSubmitButton>button {{ 
            background: linear-gradient(90deg, var(--brand), var(--brand-dark)) !important;
            color: #ffffff !important;
            font-weight: 700;
            font-size: 0.85rem;
            border-radius: 0 !important;
            padding: 0.8rem 1rem;
            border: 3px solid var(--brand-dark) !important;
            box-shadow: 0 4px 12px rgba(26,79,163,0.3),
                        0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            letter-spacing: 0.3px;
            text-transform: uppercase;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            text-align: center !important;
            display: block !important;
        }}
        .stButton>button:hover, .stFormSubmitButton>button:hover {{ 
            background: linear-gradient(90deg, var(--brand-light), var(--brand)) !important;
            box-shadow: 0 12px 32px rgba(26,79,163,0.4),
                        0 6px 12px rgba(0,0,0,0.15);
            transform: translateY(-3px) scale(1.02);
            border-color: var(--brand-light) !important;
        }}
        .stButton>button:active, .stFormSubmitButton>button:active {{
            transform: translateY(-1px) scale(1.01);
            box-shadow: 0 4px 12px rgba(26,79,163,0.3);
        }}
        
        h1, h2, h3 {{ 
            color: var(--text);
            font-weight: 800;
            font-family: var(--font);
            letter-spacing: -0.5px;
            text-transform: uppercase;
        }}
        h2 {{ 
            position: relative;
            display: inline-block;
            padding-bottom: 0.8rem;
            margin-bottom: 1.5rem !important;
        }}
        h2::after {{
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 80px;
            height: 5px;
            background: linear-gradient(90deg, var(--brand), var(--brand-dark));
            box-shadow: 0 2px 8px rgba(26,79,163,0.3);
        }}
        h3 {{
            font-size: 1.3rem !important;
            margin: 1.5rem 0 1rem 0 !important;
            padding-left: 0 !important;
            border-left: none !important;
        }}
        
        div[data-testid="stSidebar"] {{ 
            background: linear-gradient(180deg, var(--surface) 0%, var(--bg-alt) 100%);
            border-right: 4px solid var(--brand);
            box-shadow: 4px 0 16px rgba(26,79,163,0.1);
        }}
        div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
        div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
        div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {{
            border-left: 4px solid var(--brand);
            padding-left: 0.8rem;
            margin-left: -0.8rem;
        }}
        
        .stAlert {{ 
            border-radius: 0 !important;
            border-left: 5px solid var(--brand) !important; 
            border-top: 2px solid var(--border-accent) !important;
            border-right: 2px solid var(--border-accent) !important;
            border-bottom: 2px solid var(--border-accent) !important;
            background: linear-gradient(135deg, var(--surface) 0%, #e0f2fe 100%) !important;
            color: var(--text) !important;
            padding: 1.2rem 1.5rem !important;
            box-shadow: 0 4px 12px rgba(26,79,163,0.15) !important;
        }}
        
        div[data-baseweb="select"] {{ 
            border-radius: 0 !important;
            border-color: var(--brand) !important;
            border-width: 2px !important;
        }}
        input[type="checkbox"], input[type="radio"] {{ 
            accent-color: var(--brand) !important;
            transform: scale(1.2);
        }}
        .stSlider [role="slider"] {{ 
            background-color: var(--brand) !important;
            border-color: var(--brand-dark) !important;
            border-width: 3px !important;
            border-radius: 0 !important;
        }}
        
        [data-testid="stTable"] > div {{ 
            border: 3px solid var(--brand);
            border-radius: 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        [data-testid="stTable"] th {{ 
            background: linear-gradient(135deg, var(--brand) 0%, var(--brand-dark) 100%);
            color: white !important;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 1rem !important;
        }}
        [data-testid="stTable"] td {{ 
            border-color: var(--border-accent) !important;
            padding: 0.8rem !important;
        }}
        [data-testid="stTable"] tr:hover {{ background: var(--bg-alt) !important; }}
        
        #MainMenu {{ visibility: hidden; }} footer {{ visibility: hidden; }}
        
        /* Sharp square for form containers (Applicant Information) - dark mode compatible */
        [data-testid="stForm"] {{
            background: var(--surface) !important;
            border: 3px solid var(--brand) !important;
            border-radius: 0 !important;
            padding: 2rem 2.5rem !important;
            box-shadow: 0 8px 24px rgba(0,0,0,0.1),
                        0 4px 8px rgba(26,79,163,0.15),
                        inset 0 0 0 1px rgba(26,79,163,0.05) !important;
            margin-bottom: 2rem !important;
            position: relative;
        }}
        [data-testid="stForm"]::before {{
            content: '';
            position: absolute;
            top: -3px;
            left: -3px;
            right: -3px;
            bottom: -3px;
            background: linear-gradient(135deg, var(--brand-light), var(--brand-dark));
            z-index: -1;
            opacity: 0;
            transition: opacity 0.3s ease;
        }}
        [data-testid="stForm"]:hover::before {{
            opacity: 0.1;
        }}
        [data-testid="stForm"] * {{
            color: var(--text) !important;
        }}
        /* Ensure primary CTA buttons keep white text in dark mode */
        [data-testid="stForm"] .stFormSubmitButton>button,
        [data-testid="stForm"] .stFormSubmitButton>button * ,
        [data-testid="stForm"] .stButton>button,
        [data-testid="stForm"] .stButton>button * {{
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }}
        [data-testid="stForm"] label {{
            color: var(--text) !important;
        }}
        [data-baseweb="input"] input, [data-baseweb="textarea"] textarea {{ 
            border-color: var(--brand) !important;
            border-width: 2px !important;
            border-radius: 0 !important;
            background: var(--bg-alt) !important;
            color: var(--text) !important;
            padding: 0.8rem !important;
            transition: all 0.3s ease;
        }}
        [data-baseweb="input"] input:focus, [data-baseweb="textarea"] textarea:focus {{
            border-color: var(--brand-dark) !important;
            box-shadow: 0 0 0 3px rgba(26,79,163,0.1) !important;
        }}
        [data-baseweb="select"] > div {{ 
            border-color: var(--brand) !important;
            border-width: 2px !important;
            border-radius: 0 !important;
            background: var(--bg-alt) !important;
            color: var(--text) !important;
        }}
        .stNumberInput > div > div > input {{
            background: var(--bg-alt) !important;
            color: var(--text) !important;
            border-radius: 0 !important;
            border-width: 2px !important;
        }}
        .stSlider > div > div > div {{
            background: var(--bg-alt) !important;
        }}
        .stSlider [data-baseweb="slider"] {{
            border-radius: 0 !important;
        }}

        /* Applicant Information outer sharp square */
        .applicant-panel {{
            background: var(--surface);
            border: 1px solid var(--border-accent);
            border-radius: var(--radius);
            padding: 1rem 1.25rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)


def apply_theme(config_path: str | None = None) -> ThemeTokens:
    tokens = ThemeTokens.load(config_path)
    tokens.inject_css()
    return tokens
