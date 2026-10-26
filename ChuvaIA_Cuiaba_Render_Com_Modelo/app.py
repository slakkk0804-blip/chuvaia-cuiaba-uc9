from __future__ import annotations

import calendar
import html
import json
import sqlite3
from datetime import date, datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from modelo_utils import LATITUDE, LONGITUDE, METEO, TIMEZONE, preparar_variaveis

BASE = Path(__file__).resolve().parent
METRICAS = BASE / "artefatos" / "metricas.json"
MODELO = BASE / "artefatos" / "modelo_chuva.joblib"
HISTORICO = BASE / "dados" / "historico_previsoes.sqlite3"
CACHE_TEMPO = BASE / "dados" / "ultimo_tempo_valido.json"
CACHE_BLOQUEIO = BASE / "dados" / "limite_openmeteo.json"
LOGO = BASE / "assets" / "logo_chuvaia.png"

st.set_page_config(
    page_title="ChuvaIA Campo | Cuiabá",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root{
  --bg:#f6f7fb;
  --panel:#ffffff;
  --panel-soft:#fbfbfd;
  --border:#ebeaf2;
  --text:#26263a;
  --muted:#8f90a6;
  --orange:#f4a66f;
  --purple:#8f84ef;
  --cyan:#4cc8df;
  --blue:#6979d9;
  --shadow:0 10px 30px rgba(94,108,132,.08);
  --radius:20px;
}
html,body,[data-testid="stAppViewContainer"],.stApp,
[data-testid="stAppViewContainer"] *:not(svg):not(path){font-family:'Plus Jakarta Sans','Segoe UI',sans-serif!important;}
body,.stApp,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;}
#MainMenu, footer, header{visibility:hidden!important;}
.block-container{max-width:1400px!important;padding:24px 26px 26px!important;}
[data-testid="stVerticalBlock"]{gap:1rem;}
[data-testid="stHorizontalBlock"]{gap:1rem;}

/* Generic cards */
.soft-card{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow);padding:20px;}
.tiny-label{font-size:.72rem;font-weight:700;letter-spacing:.02em;color:var(--muted);}
.big-number{font-size:2.55rem;font-weight:800;line-height:1;color:var(--text);margin-top:8px;}
.section-title{font-size:1.22rem;font-weight:800;color:var(--text);letter-spacing:-.03em;margin:6px 0 10px;}
.section-subtitle{font-size:.86rem;color:var(--muted);line-height:1.6;margin-top:-4px;}

/* Top layout */
.shell{display:flex;gap:18px;align-items:flex-start;}
.sidebar-frame{position:sticky;top:24px;align-self:flex-start;}
.sidebar-card{background:var(--panel);border:1px solid var(--border);border-radius:26px;box-shadow:var(--shadow);padding:16px 12px 14px;min-height:760px;}
.logo-wrap{display:flex;align-items:center;gap:10px;padding:8px 10px 18px;}
.logo-badge{width:34px;height:34px;border-radius:11px;background:#fff4ea;color:#eb9b61;display:grid;place-items:center;font-size:18px;font-weight:700;}
.logo-title{font-size:1rem;font-weight:800;color:#2f3142;line-height:1.1;}
.logo-sub{font-size:.66rem;color:#a1a2b5;margin-top:3px;}
.side-nav-spacer{height:8px;}
.side-footer{margin-top:18px;padding:12px 10px 8px;border-top:1px solid var(--border);font-size:.74rem;color:var(--muted);}

.main-wrap{flex:1;min-width:0;}
.topbar{display:flex;align-items:center;justify-content:space-between;gap:18px;margin-bottom:16px;}
.topbar-title{font-size:1.45rem;font-weight:800;letter-spacing:-.04em;color:var(--text);}
.topbar-right{display:flex;align-items:center;gap:12px;}
.ghost-pill{background:var(--panel);border:1px solid var(--border);height:42px;padding:0 15px;border-radius:14px;display:flex;align-items:center;color:var(--muted);font-size:.85rem;font-weight:700;box-shadow:var(--shadow);}
.user-chip{display:flex;align-items:center;gap:9px;background:var(--panel);border:1px solid var(--border);padding:7px 11px;border-radius:999px;box-shadow:var(--shadow);}
.user-avatar{width:31px;height:31px;border-radius:50%;background:linear-gradient(135deg,#a6b6ff,#8b86ea);display:grid;place-items:center;color:white;font-weight:800;font-size:.85rem;}
.user-name{font-size:.8rem;font-weight:700;color:#46495c;}

/* highlight cards */
.quick-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin-bottom:16px;}
.quick-card{position:relative;overflow:hidden;color:#fff;padding:18px 18px 16px;border-radius:18px;min-height:104px;box-shadow:0 12px 24px rgba(125,121,167,.17);}
.quick-card.orange{background:linear-gradient(135deg,#f7b385,#ef965e);}
.quick-card.purple{background:linear-gradient(135deg,#9a93f0,#7f74dd);}
.quick-card.cyan{background:linear-gradient(135deg,#43d0e3,#25b5ce);}
.quick-card.blue{background:linear-gradient(135deg,#7b89da,#5d68b5);}
.quick-icon{position:absolute;right:16px;top:16px;width:38px;height:38px;border-radius:12px;background:rgba(255,255,255,.28);display:grid;place-items:center;font-size:18px;}
.quick-value{font-size:1.9rem;font-weight:800;letter-spacing:-.05em;margin-top:18px;}
.quick-label{font-size:.75rem;font-weight:700;opacity:.95;}
.quick-sub{font-size:.72rem;opacity:.88;margin-top:4px;}

.grid-2-1{display:grid;grid-template-columns:2.1fr 1fr;gap:16px;align-items:start;}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start;}
.small-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start;}

.mini-toolbar{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:10px;}
.mini-pill-row{display:flex;gap:7px;flex-wrap:wrap;}
.mini-pill{background:#f7f7fb;border:1px solid var(--border);padding:5px 10px;border-radius:999px;font-size:.72rem;color:var(--muted);font-weight:700;}
.big-prob{font-size:3rem;font-weight:800;letter-spacing:-.06em;color:var(--text);line-height:1;margin:8px 0;}
.risk-badge{display:inline-flex;align-items:center;gap:7px;background:#eef8f2;color:#31936e;border:1px solid #d4ebdd;border-radius:999px;padding:7px 12px;font-size:.76rem;font-weight:800;}
.risk-badge.medium{background:#fff6e9;border-color:#f2e0bc;color:#c08831;}
.risk-badge.high{background:#fff0f1;border-color:#f3d7da;color:#bf6771;}
.progress-track{width:100%;height:11px;border-radius:999px;background:#eeedf6;overflow:hidden;margin:14px 0 10px;}
.progress-fill{height:100%;background:linear-gradient(90deg,#9f95f1,#7267db);border-radius:999px;}
.helper-text{font-size:.84rem;line-height:1.7;color:var(--muted);}
.weather-list{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px;}
.weather-item{background:#fafafe;border:1px solid var(--border);border-radius:16px;padding:14px 15px;}
.weather-item .value{font-size:1.28rem;font-weight:800;color:var(--text);margin-top:6px;}

.calendar-card{background:linear-gradient(180deg,#7786dc,#5f6bbc);color:#fff;border-radius:22px;padding:16px;box-shadow:0 16px 35px rgba(95,107,188,.28);}
.calendar-header{display:flex;align-items:center;justify-content:space-between;font-size:.82rem;font-weight:700;margin-bottom:10px;}
.calendar-grid{display:grid;grid-template-columns:repeat(7,1fr);gap:6px;font-size:.74rem;text-align:center;}
.calendar-grid div{padding:6px 0;border-radius:10px;}
.calendar-dayname{color:rgba(255,255,255,.7);font-weight:700;}
.calendar-empty{opacity:.24;}
.calendar-today{background:rgba(255,255,255,.18);font-weight:800;}
.calendar-pill{margin-top:12px;padding-top:12px;border-top:1px solid rgba(255,255,255,.2);font-size:.74rem;}

.people-list{margin:6px 0 0;padding:0;list-style:none;}
.people-list li{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:10px 0;border-bottom:1px solid var(--border);}
.people-list li:last-child{border-bottom:0;}
.person{display:flex;align-items:center;gap:10px;}
.person-avatar{width:34px;height:34px;border-radius:50%;background:#f1f0fe;color:#7368d8;display:grid;place-items:center;font-weight:800;font-size:.8rem;}
.person-meta{font-size:.78rem;color:var(--muted);}
.person-name{font-size:.85rem;color:var(--text);font-weight:700;}
.table-note{font-size:.75rem;color:var(--muted);margin-top:8px;}

[data-testid="stPlotlyChart"]{background:transparent!important;border:0!important;box-shadow:none!important;padding:0!important;}
[data-testid="stDataFrame"]{border-radius:16px;overflow:hidden;border:1px solid var(--border);}
[data-testid="stDataFrame"] *{font-size:.83rem;}
[data-testid="stButton"] button{width:100%;min-height:42px;border-radius:14px!important;border:1px solid transparent!important;background:transparent!important;color:#82849a!important;font-weight:700!important;box-shadow:none!important;justify-content:flex-start!important;padding-left:14px!important;}
[data-testid="stButton"] button:hover{background:#f6f6fb!important;color:#5d60aa!important;border-color:#eceaf8!important;}
[data-testid="stButton"] button[kind="primary"]{background:#f6f4ff!important;color:#6357c3!important;border:1px solid #e8e4ff!important;}
[data-testid="stDownloadButton"] button{border-radius:14px!important;border:1px solid var(--border)!important;background:var(--panel)!important;color:#5d63a0!important;font-weight:700!important;}
[data-testid="stExpander"]{border:1px solid var(--border)!important;background:var(--panel)!important;border-radius:18px!important;}
[data-testid="stAlert"]{border-radius:16px!important;}
.stTabs [data-baseweb="tab-list"]{gap:8px;}
.stTabs [data-baseweb="tab"]{background:#fff;border:1px solid var(--border);padding:8px 14px;border-radius:12px;}
.stTabs [aria-selected="true"]{background:#f6f4ff;color:#6357c3;}

@media(max-width:1100px){
  .shell{display:block;}
  .sidebar-frame{position:relative;top:auto;margin-bottom:16px;}
  .sidebar-card{min-height:auto;}
  .grid-2-1,.grid-2,.small-grid{grid-template-columns:1fr;}
  .quick-grid{grid-template-columns:repeat(2,minmax(0,1fr));}
}
@media(max-width:700px){
  .block-container{padding:14px!important;}
  .quick-grid{grid-template-columns:1fr;}
  .weather-list{grid-template-columns:1fr;}
  .topbar{display:block;}
  .topbar-right{margin-top:10px;justify-content:flex-start;flex-wrap:wrap;}
}

/* Layout corrigido: Streamlit gerencia os painéis e o grid, sem HTML atravessando blocos. */
[data-testid="stMainBlockContainer"],.block-container{
  width:100%!important;max-width:1540px!important;margin-inline:auto!important;
  padding:22px 24px 42px!important;
}
[data-testid="stAppViewBlockContainer"]{max-width:1540px!important;}
[data-testid="stHorizontalBlock"]{align-items:stretch!important;gap:16px!important;}
[data-testid="stColumn"]{min-width:0!important;}
.st-key-side_panel{
  position:sticky!important;top:16px!important;min-height:calc(100vh - 46px);
  background:#fff;border:1px solid #eaeaf1;border-radius:22px;padding:18px 13px!important;
  box-shadow:0 9px 26px rgba(78,89,120,.07);
}
.st-key-side_panel [data-testid="stVerticalBlock"]{gap:.42rem!important;}
.st-key-side_panel [data-testid="stButton"] button{min-height:45px!important;width:100%;font-size:.84rem!important;}
.st-key-main_panel{min-width:0!important;}
.st-key-forecast_panel,.st-key-chart_models,.st-key-chart_importance,
.st-key-analytics_a,.st-key-analytics_b,.st-key-analytics_c,.st-key-analytics_d,
.st-key-data_table,.st-key-report_intro,.st-key-report_method,.st-key-report_limits,
.st-key-recent_panel,.st-key-weather_panel{
  background:#fff!important;border:1px solid #e9eaf1;border-radius:20px;
  padding:19px 20px!important;box-shadow:0 8px 24px rgba(70,82,112,.065);
}
.st-key-calendar_panel{padding:0!important;}
.st-key-forecast_panel [data-testid="stPlotlyChart"],.st-key-chart_models [data-testid="stPlotlyChart"]{
  width:100%!important;
}
.st-key-main_panel .section-title{margin:0 0 5px;}
.st-key-main_panel .section-subtitle{font-size:.79rem;line-height:1.5;margin-bottom:10px;}
.st-key-side_panel .logo-wrap{padding:7px 5px 21px;}
.st-key-side_panel .side-footer{margin-top:23px;line-height:1.7;}
.header-status{display:flex;align-items:center;gap:9px;justify-content:flex-end;flex-wrap:wrap;}
.header-status .ghost-pill{font-size:.75rem;}
.small-print{font-size:.74rem;color:#9295a6;line-height:1.5;margin:8px 0 5px;}
.forecast-copy{display:flex;align-items:center;justify-content:space-between;gap:15px;}
.forecast-emoji{font-size:3.3rem;line-height:1;}
.forecast-line{font-size:.9rem;color:#6f7485;line-height:1.55;margin:10px 0 8px;}
.quick-card{min-height:118px;}
.quick-value{font-size:clamp(1.1rem,2.2vw,1.88rem);white-space:normal;word-break:break-word;}
@media(max-width:1050px){
  .st-key-side_panel{position:relative!important;min-height:0!important;padding:12px!important;}
  .st-key-side_panel [data-testid="stVerticalBlock"]{gap:.4rem!important;}
}
@media(max-width:800px){
  .block-container{padding:12px 12px 32px!important;}
  .quick-grid{grid-template-columns:repeat(2,minmax(0,1fr));}
  .quick-card{min-height:105px;}
  .st-key-side_panel{min-height:0!important;}
  .st-key-main_panel .section-title{font-size:1.04rem;}
}
@media(max-width:520px){
  .quick-card{padding:14px 12px;}
  .quick-value{font-size:1.4rem;}
  .quick-icon{width:31px;height:31px;right:9px;top:11px;}
  .ghost-pill{height:34px;padding:0 9px;}
  .st-key-forecast_panel,.st-key-chart_models,.st-key-chart_importance,
  .st-key-analytics_a,.st-key-analytics_b,.st-key-analytics_c,.st-key-analytics_d,
  .st-key-data_table,.st-key-report_intro,.st-key-report_method,.st-key-report_limits,
  .st-key-recent_panel,.st-key-weather_panel{padding:14px!important;}
}


/* Versão 4 — sidebar nativa: a logo e os botões ficam no topo do painel branco. */
section[data-testid="stSidebar"]{
  background:#fff!important;
  border-right:1px solid #e9eaf2!important;
  box-shadow:5px 0 24px rgba(67,75,115,.045)!important;
  min-width:245px!important;
  max-width:245px!important;
  width:245px!important;
}
section[data-testid="stSidebar"] > div{background:#fff!important;}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"]{
  background:#fff!important;
  padding:21px 15px 24px!important;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{
  gap:0.48rem!important;
  align-items:stretch!important;
  justify-content:flex-start!important;
}
section[data-testid="stSidebar"] [data-testid="stImage"]{
  margin:0 0 10px!important;
}
section[data-testid="stSidebar"] [data-testid="stImage"] img{
  display:block!important;max-width:100%!important;max-height:110px!important;object-fit:contain!important;
}
section[data-testid="stSidebar"] [data-testid="stButton"] button{
  justify-content:flex-start!important;
  text-align:left!important;
  width:100%!important;
  min-height:43px!important;
  padding:8px 13px!important;
  margin:0!important;
  border-radius:12px!important;
  font-size:.85rem!important;
  font-weight:700!important;
  color:#6b6f85!important;
  background:transparent!important;
  border:1px solid transparent!important;
}
section[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"]{
  color:#665bc9!important;
  background:#f0edff!important;
  border:1px solid #e6e1ff!important;
}
section[data-testid="stSidebar"] [data-testid="stButton"] button:hover{
  color:#5c56aa!important;
  background:#f6f5fc!important;
}
.nav-section-label{
  color:#a5a9b7;font-size:.67rem;font-weight:800;letter-spacing:.14em;
  padding:8px 9px 4px;
}
.sidebar-note{
  margin-top:25px;border-top:1px solid #eaebf2;padding:17px 8px 0;
  display:flex;flex-direction:column;gap:6px;color:#979baa;font-size:.70rem;line-height:1.5;
}
.sidebar-note strong{font-weight:800;color:#777c93;font-size:.67rem;letter-spacing:.06em;}
.brand-fallback{font-size:1.23rem;font-weight:800;padding:15px 8px;}
.brand-fallback span{display:block;color:#8a8b9f;font-size:.86rem;}
/* Torna a área central ampla, sem a coluna em branco criada antes. */
section[data-testid="stMain"] .block-container{
  max-width:1480px!important;
  width:100%!important;
  padding:23px 27px 50px!important;
}
@media(max-width:920px){
  section[data-testid="stSidebar"]{min-width:230px!important;max-width:230px!important;width:230px!important;}
  section[data-testid="stMain"] .block-container{padding:16px 15px 35px!important;}
}

/* ChuvaIA: revisão de legibilidade para browsers com tema escuro e modo claro do app.
   A regra de contraste fica no fim para prevalecer sobre os estilos antigos. */
:root{color-scheme:light!important;--text:#21283a!important;--muted:#566075!important;}
[data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stSidebar"], .stApp {color-scheme:light!important;}
.stMarkdown,[data-testid="stMarkdownContainer"],
[data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"]{
  color:#253047!important;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"] p,
[data-testid="stWidgetLabel"] p {color:#445168!important;}
.section-title,.topbar-title,.person-name,.weather-item .value,
.big-prob,.user-name,.section-title strong {color:#1d2638!important;}
.section-subtitle,.tiny-label,.small-print,.helper-text,.forecast-line,
.person-meta,.table-note,.sidebar-note,.sidebar-note span,
.mini-pill,.ghost-pill,.quick-legend,.nav-section-label,
[data-testid="stCaptionContainer"] p{color:#535e72!important;opacity:1!important;}
/* Quatro cards mantêm suas cores originais com texto escuro e nítido. */
.quick-card{color:#1c2940!important;}
.quick-card .quick-label,.quick-card .quick-value,.quick-card .quick-sub,
.quick-card .quick-icon{color:#1d2b44!important;opacity:1!important;
 text-shadow:none!important;}
.quick-card .quick-label{font-size:.78rem!important;}
.quick-card .quick-sub{font-size:.74rem!important;}
.quick-card.orange{background:linear-gradient(135deg,#ffe2cf,#f6b886)!important;}
.quick-card.purple{background:linear-gradient(135deg,#e4ddff,#b3a8f5)!important;}
.quick-card.cyan{background:linear-gradient(135deg,#d1f6fc,#8ee4ef)!important;}
.quick-card.blue{background:linear-gradient(135deg,#e0e7ff,#a5b4ec)!important;}
.quick-icon{background:#ffffff80!important;}
/* A agenda usa fundo um pouco mais escuro para texto branco legível. */
.calendar-card{background:linear-gradient(180deg,#6377c3,#495ca8)!important;}
.calendar-card,.calendar-card *{color:#fff!important;}
.calendar-dayname{color:#e8edff!important;opacity:1!important;}
.calendar-empty{opacity:.55!important;}
/* Rótulos e botões do Streamlit fora dos cards coloridos */
section[data-testid="stSidebar"] [data-testid="stButton"] button{color:#3a4965!important;}
section[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"]{color:#463aab!important;}
[data-testid="stExpander"] summary,[data-testid="stExpander"] p{color:#2b3853!important;}
[data-testid="stDataFrame"]{color-scheme:light!important;}
/* Modos de alto contraste e acessibilidade */
@media(prefers-reduced-motion:reduce){.quick-card,.progress-fill{transition:none!important;}}
</style>
""",
    unsafe_allow_html=True,
)


def pt_num(valor: float, digitos: int = 1) -> str:
    return f"{valor:.{digitos}f}".replace(".", ",")


def v_float(dicionario: dict, chave: str) -> float:
    valor = float(dicionario[chave])
    if not np.isfinite(valor):
        raise ValueError(f"Valor meteorológico inválido: {chave}")
    return valor


@st.cache_data(show_spinner=False)
def abrir_metricas():
    """Lê apenas o JSON; abas de análise não precisam carregar o modelo pesado."""
    if not METRICAS.exists():
        raise FileNotFoundError("Falta o arquivo artefatos/metricas.json.")
    return json.loads(METRICAS.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def abrir_modelo():
    """Carrega o Random Forest uma única vez por processo, sob demanda."""
    if not MODELO.exists():
        raise FileNotFoundError("Falta o arquivo artefatos/modelo_chuva.joblib.")
    return joblib.load(MODELO), abrir_metricas()


@st.cache_data(ttl=900, show_spinner=False)
def buscar_tempo() -> dict:
    """Consulta compartilhada por visitantes, com cooldown para erros 429.

    Tanto sucesso quanto indisponibilidade são guardados 15 minutos na cache
    do Streamlit. Isso impede que cada visitante/refresh faça nova chamada
    quando o IP da hospedagem estiver temporariamente limitado.

    A última resposta válida também é salva localmente para quedas passageiras.
    Em um redeploy ou restart no Render gratuito, essa cópia pode desaparecer.
    """
    parametros = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "timezone": TIMEZONE,
        "current": ",".join([*METEO, "weather_code"]),
        "forecast_days": 1,
    }
    motivo = "Não foi possível acessar a fonte meteorológica neste momento."
    # Evita solicitar novamente antes do prazo informado em Retry-After.
    try:
        estado_bloqueio = json.loads(CACHE_BLOQUEIO.read_text(encoding="utf-8"))
        limite = datetime.fromisoformat(estado_bloqueio["proxima_tentativa_utc"])
        if limite > datetime.now(timezone.utc):
            motivo = "A Open-Meteo está limitando temporariamente o acesso (HTTP 429)"
            bloqueado = True
        else:
            bloqueado = False
    except (OSError, ValueError, KeyError, TypeError):
        bloqueado = False
    try:
        if bloqueado:
            raise requests.HTTPError(motivo)
        resposta = requests.get(
            "https://api.open-meteo.com/v1/forecast", params=parametros,
            timeout=18, headers={"User-Agent": "ChuvaIA-Campo-SENAC-UC9/1.0"},
        )
        if resposta.status_code == 429:
            motivo = "A Open-Meteo limitou temporariamente o acesso (HTTP 429)"
            # Respeita Retry-After quando disponível e, na falta dele,
            # mantém um intervalo mínimo de quinze minutos.
            espera = 900
            indicado = resposta.headers.get("Retry-After", "").strip()
            try:
                if indicado.isdecimal():
                    espera = max(espera, int(indicado))
                elif indicado:
                    espera = max(espera, int((parsedate_to_datetime(indicado)
                                             - datetime.now(timezone.utc)).total_seconds()))
            except (ValueError, TypeError, OverflowError):
                pass
            espera = min(espera, 86400)
            try:
                CACHE_BLOQUEIO.parent.mkdir(parents=True, exist_ok=True)
                CACHE_BLOQUEIO.write_text(json.dumps({
                    "proxima_tentativa_utc": (datetime.now(timezone.utc)
                                           + timedelta(seconds=espera)).isoformat()
                }), encoding="utf-8")
            except OSError:
                pass
            raise requests.HTTPError(motivo, response=resposta)
        resposta.raise_for_status()
        atual = resposta.json().get("current", {})
        faltando = [c for c in [*METEO, "time"] if atual.get(c) is None]
        if faltando:
            raise ValueError("A fonte não retornou todas as variáveis meteorológicas")
        snapshot = {
            "atual": atual,
            "consultado_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            CACHE_TEMPO.parent.mkdir(parents=True, exist_ok=True)
            temporario = CACHE_TEMPO.with_suffix(".tmp")
            temporario.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
            temporario.replace(CACHE_TEMPO)
        except OSError:
            pass  # Cache em disco é opcional; não bloqueia uma previsão válida.
        return {**snapshot, "estado": "atual", "mensagem": ""}
    except requests.HTTPError as erro:
        if not bloqueado and getattr(erro.response, "status_code", None) != 429:
            motivo = f"O serviço meteorológico retornou HTTP {erro.response.status_code if erro.response is not None else 'indisponível'}"
    except requests.RequestException:
        motivo = "Falha de comunicação com a fonte meteorológica"
    except (ValueError, TypeError, KeyError):
        motivo = "A fonte retornou dados meteorológicos incompletos ou inválidos"

    # Não inventar uma previsão em caso de erro. Só usar informação real recente.
    try:
        snapshot = json.loads(CACHE_TEMPO.read_text(encoding="utf-8"))
        recebido = datetime.fromisoformat(snapshot["consultado_utc"])
        idade_s = (datetime.now(timezone.utc) - recebido).total_seconds()
        if 0 <= idade_s <= 3600 and all(
            snapshot["atual"].get(c) is not None for c in [*METEO, "time"]
        ):
            return {**snapshot, "estado": "anterior", "mensagem": motivo}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    return {"atual": None, "consultado_utc": None,
            "estado": "indisponivel", "mensagem": motivo}


def prever(atual: dict, pacote: dict) -> float:
    entrada = {"time": atual["time"], **{campo: atual[campo] for campo in METEO}}
    x = preparar_variaveis(pd.DataFrame([entrada]))
    if x.columns.tolist() != pacote["colunas"]:
        raise ValueError("As colunas do treinamento e da previsão não coincidem.")
    estimativa = float(pacote["modelo"].predict_proba(x)[0, 1])
    if not np.isfinite(estimativa) or not 0 <= estimativa <= 1:
        raise ValueError("O modelo não produziu uma probabilidade válida.")
    return estimativa


def nivel_chuva(p: float) -> tuple[str, str, str, str]:
    if p < .30:
        return "Baixa chance de chuva", "Baixo", "", "🌤️"
    if p < .65:
        return "Chance moderada de chuva", "Moderado", "medium", "⛅"
    return "Chance elevada de chuva", "Elevado", "high", "🌧️"


def registrar_previsao(hora_fonte: str, modelo: str, chance: float, atual: dict) -> None:
    HISTORICO.parent.mkdir(parents=True, exist_ok=True)
    versao = str(MODELO.stat().st_mtime_ns) if MODELO.exists() else "desconhecida"
    with sqlite3.connect(HISTORICO, timeout=5) as conexao:
        conexao.execute("""CREATE TABLE IF NOT EXISTS previsoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hora_fonte TEXT NOT NULL,
            registrado_utc TEXT NOT NULL,
            modelo TEXT NOT NULL,
            versao_modelo TEXT NOT NULL,
            chance REAL NOT NULL,
            temperatura REAL,
            umidade REAL,
            vento REAL,
            nebulosidade REAL,
            UNIQUE(hora_fonte, versao_modelo)
        )""")
        conexao.execute("""INSERT OR IGNORE INTO previsoes
            (hora_fonte, registrado_utc, modelo, versao_modelo, chance, temperatura, umidade, vento, nebulosidade)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
            hora_fonte,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            modelo,
            versao,
            chance,
            v_float(atual, "temperature_2m"),
            v_float(atual, "relative_humidity_2m"),
            v_float(atual, "wind_speed_10m"),
            v_float(atual, "cloud_cover"),
        ))


def ler_historico(limite: int = 60) -> pd.DataFrame:
    if not HISTORICO.exists():
        return pd.DataFrame()
    with sqlite3.connect(HISTORICO, timeout=5) as conexao:
        df = pd.read_sql_query(
            """SELECT hora_fonte, registrado_utc, modelo, chance, temperatura, umidade, vento, nebulosidade
               FROM previsoes ORDER BY id DESC LIMIT ?""",
            conexao,
            params=(limite,),
        )
    return df.iloc[::-1].reset_index(drop=True)


def plot_base(titulo: str, altura: int = 280) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=altura,
        margin=dict(l=18, r=18, t=46, b=18),
        title=dict(text=titulo, x=.03, font=dict(size=15, color="#2f3142")),
        font=dict(family="Plus Jakarta Sans, Segoe UI, sans-serif", size=12, color="#4d566c"),
        xaxis=dict(showgrid=False, zeroline=False, showline=False),
        yaxis=dict(gridcolor="#f1f1f5", zeroline=False, showline=False),
        hoverlabel=dict(bgcolor="#2f3142", font_color="#fff"),
        showlegend=False,
    )
    return fig


def graf_linha_historico(df: pd.DataFrame) -> go.Figure:
    base = df.copy()
    base["instante"] = pd.to_datetime(base["hora_fonte"], errors="coerce")
    base = base.dropna(subset=["instante"])
    fig = plot_base("Histórico das previsões", 285)
    if not base.empty:
        y = base["chance"] * 100
        fig.add_trace(go.Scatter(
            x=base["instante"],
            y=y,
            mode="lines+markers",
            line=dict(color="#d8b28f", width=3, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(218,188,160,.18)",
            hovertemplate="%{x|%d/%m %H:%M}<br><b>%{y:.1f}%</b><extra></extra>",
        ))
        fig.update_xaxes(tickformat="%d/%m")
        fig.update_yaxes(ticksuffix="%")
    return fig


def graf_donut(prob: float) -> go.Figure:
    fig = plot_base("Rain chance", 260)
    restante = max(0, 100 - prob * 100)
    fig.add_trace(go.Pie(
        values=[prob * 100, restante],
        hole=.73,
        marker=dict(colors=["#f1a46d", "#eff0f7"]),
        textinfo="none",
        sort=False,
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
        labels=["Chuva", "Restante"],
    ))
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), annotations=[dict(
        text=f"<b>{pt_num(prob*100,1)}%</b><br><span style='font-size:12px;color:#8f90a6'>chance</span>",
        x=.5, y=.5, showarrow=False, font=dict(size=22, color="#2f3142")
    )])
    return fig


@st.cache_data(show_spinner=False, max_entries=8)
def graf_barras_modelos(meta: dict) -> go.Figure:
    nomes = list(meta["metricas_teste"].keys())
    valores = [meta["metricas_teste"][n]["f1_macro"] * 100 for n in nomes]
    fig = plot_base("F1 macro por algoritmo", 280)
    fig.add_trace(go.Bar(
        x=valores,
        y=nomes,
        orientation="h",
        marker=dict(color=["#f3a76f", "#8b82ef", "#3ec8df", "#6879d8"], cornerradius=6),
        hovertemplate="%{y}<br><b>%{x:.2f}%</b><extra></extra>",
    ))
    fig.update_xaxes(ticksuffix="%")
    fig.update_yaxes(autorange="reversed")
    return fig


@st.cache_data(show_spinner=False, max_entries=8)
def graf_importancia(meta: dict) -> go.Figure:
    itens = meta["importancia_permutacao"][:5][::-1]
    fig = plot_base("Feature importance", 270)
    fig.add_trace(go.Bar(
        x=[i["queda_f1_macro"] for i in itens],
        y=[i["variavel"] for i in itens],
        orientation="h",
        marker=dict(color="#8b82ef", cornerradius=6),
        hovertemplate="%{y}<br>Queda no F1: %{x:.4f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    return fig


@st.cache_data(show_spinner=False, max_entries=8)
def graf_matriz(meta: dict) -> go.Figure:
    matriz = np.asarray(meta["matriz_confusao_teste"])
    fig = plot_base("Matriz de confusão", 285)
    fig.add_trace(go.Heatmap(
        z=matriz,
        x=["Sem chuva", "Chuva"],
        y=["Sem chuva", "Chuva"],
        colorscale=[[0, "#f4f2ff"], [1, "#8d82ed"]],
        text=matriz,
        texttemplate="%{text}",
        textfont=dict(color="#1d2440", size=15),
        showscale=False,
        hovertemplate="Real: %{y}<br>Previsto: %{x}<br>Casos: %{z}<extra></extra>",
    ))
    fig.update_layout(margin=dict(l=40, r=15, t=48, b=25))
    return fig


def render_plot(fig: go.Figure, key: str):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "responsive": True}, key=key)


def render_calendar() -> str:
    today = date.today()
    cal = calendar.Calendar(firstweekday=6)
    meses_pt = ["", "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
                "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]
    month_name = f"{meses_pt[today.month]} {today.year}"
    month_days = cal.monthdayscalendar(today.year, today.month)
    parts = [
        '<div class="calendar-card">',
        f'<div class="calendar-header"><span>{html.escape(month_name.title())}</span><span>🗓️</span></div>',
        '<div class="calendar-grid">',
    ]
    for name in ["DOM", "SEG", "TER", "QUA", "QUI", "SEX", "SÁB"]:
        parts.append(f'<div class="calendar-dayname">{name}</div>')
    for week in month_days:
        for d in week:
            if d == 0:
                parts.append('<div class="calendar-empty">·</div>')
            else:
                cls = "calendar-today" if d == today.day else ""
                parts.append(f'<div class="{cls}">{d}</div>')
    parts.extend([
        '</div>',
        '<div class="calendar-pill"><strong>Atualização automática</strong><br>Dados atuais da Open-Meteo + previsão do modelo em tempo real.</div>',
        '</div>'
    ])
    return "".join(parts)


def sidebar_nav() -> None:
    """Navegação nativa dentro da barra branca, da parte superior para baixo."""
    if LOGO.exists():
        st.image(str(LOGO), width="stretch")
    else:
        st.markdown('<div class="brand-fallback">ChuvaIA <span>Cuiabá</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="nav-section-label">MENU</div>', unsafe_allow_html=True)
    paginas = [
        ("dashboard", "▦  Dashboard"),
        ("analytics", "▤  Analytics"),
        ("relatorio", "▧  Projeto UC9"),
    ]
    for key, label in paginas:
        if st.button(
            label,
            key=f"nav_{key}",
            type="primary" if st.session_state.pagina == key else "secondary",
            width="stretch",
        ):
            if st.session_state.pagina != key:
                st.session_state.pagina = key
                st.rerun()

    st.markdown(
        '<div class="sidebar-note">'
        '<strong>SENAC · UC9 · GRUPO 6</strong>'
        '<span>Previsão automática de chuva para Cuiabá.</span>'
        '</div>',
        unsafe_allow_html=True,
    )

def topbar(title: str) -> None:
    st.markdown(
        '<div class="topbar">'
        f'<div class="topbar-title">{html.escape(title)}</div>'
        '<div class="topbar-right">'
        '<div class="ghost-pill">⌖ Cuiabá, MT</div>'
        '<div class="user-chip"><div class="user-avatar">6</div><div class="user-name">Grupo 6</div></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def quick_cards(atual: dict, prob: float) -> None:
    cards = [
        ("orange", f"{pt_num(v_float(atual, 'temperature_2m'))}°C", "Temperatura", "Condição atual", "☀"),
        ("purple", f"{pt_num(prob * 100, 1)}%", "Chance de chuva", "Próximas 3 horas", "☁"),
        ("cyan", f"{pt_num(v_float(atual, 'relative_humidity_2m'), 0)}%", "Umidade do ar", "Condição atual", "◉"),
        ("blue", f"{pt_num(v_float(atual, 'wind_speed_10m'))} km/h", "Vento", "Velocidade atual", "≈"),
    ]
    content = ''.join(
        f'<div class="quick-card {c}"><div class="quick-label">{html.escape(lab)}</div>'
        f'<div class="quick-value">{html.escape(val)}</div>'
        f'<div class="quick-sub">{html.escape(sub)}</div><div class="quick-icon">{ic}</div></div>'
        for c, val, lab, sub, ic in cards
    )
    st.markdown(f'<div class="quick-grid">{content}</div>', unsafe_allow_html=True)


@st.fragment(run_every="15min")
def dashboard_page() -> None:
    topbar("Dashboard")
    refresh_slot, status_slot = st.columns([1.2, 3], vertical_alignment="center")
    with refresh_slot:
        if st.button("↻ Atualizar previsão", key="force_refresh", width="stretch"):
            # Não limpe o cache aqui: isso provocaria novas chamadas e mais erros 429.
            st.rerun(scope="fragment")
    with status_slot:
        st.caption("Dados estimados pela Open-Meteo · renovação automática a cada 15 minutos")

    try:
        atual = buscar_tempo()
    except (requests.RequestException, ValueError, KeyError, TypeError) as err:
        st.error("Não foi possível consultar o clima. Confira a conexão e clique em Atualizar previsão.")
        st.caption(f"Detalhes: {err}")
        return
    try:
        pacote, meta = abrir_modelo()
        prob = prever(atual, pacote)
    except (FileNotFoundError, OSError, ValueError, KeyError, IndexError, AttributeError) as err:
        st.warning("O clima foi consultado, mas o modelo treinado não está disponível ou é incompatível.")
        st.caption(f"Detalhes: {err}")
        return
    try:
        registrar_previsao(str(atual["time"]), str(pacote["nome"]), prob, atual)
        hist = ler_historico()
        historico_ok = True
    except (OSError, sqlite3.Error, ValueError):
        hist = pd.DataFrame()
        historico_ok = False

    quick_cards(atual, prob)
    desc, nivel, badge_class, icone = nivel_chuva(prob)
    esquerda, direita = st.columns([2.1, 1], gap="medium")
    with esquerda:
        with st.container(key="forecast_panel"):
            st.markdown(
                '<div class="section-title">Previsão de chuva</div>'
                '<div class="section-subtitle">Próximas três horas · classificação automática do Random Forest</div>'
                '<div class="forecast-copy">'
                f'<div><div class="big-prob">{pt_num(prob * 100, 1)}%</div>'
                f'<div class="risk-badge {badge_class}">● Nível {html.escape(nivel.lower())}</div></div>'
                f'<div class="forecast-emoji" role="img" aria-label="Tempo">{icone}</div>'
                '</div>'
                f'<div class="progress-track"><div class="progress-fill" style="width:{max(1, prob * 100):.2f}%"></div></div>'
                f'<div class="forecast-line">{html.escape(desc)}. Esta é uma estimativa de chuva acumulada de pelo menos 0,2 mm nas três horas seguintes.</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="section-title" style="margin-top:17px">Histórico das previsões</div>', unsafe_allow_html=True)
            if not hist.empty:
                render_plot(graf_linha_historico(hist), "history_main")
            else:
                st.info("O histórico será exibido aqui após a primeira consulta salva.")
            st.markdown(
                '<div class="small-print">A probabilidade não é calibrada nem constitui alerta oficial. '
                'O histórico mostra as estimativas registradas, não os acertos.</div>',
                unsafe_allow_html=True,
            )
    with direita:
        with st.container(key="calendar_panel"):
            st.markdown(render_calendar(), unsafe_allow_html=True)
        with st.container(key="recent_panel"):
            st.markdown('<div class="section-title">Últimas consultas</div>'
                        '<div class="section-subtitle">Estimativas salvas automaticamente.</div>',
                        unsafe_allow_html=True)
            if hist.empty:
                st.caption("Nenhuma previsão registrada ainda.")
            else:
                itens = []
                for _, row in hist.tail(4).iloc[::-1].iterrows():
                    when = pd.to_datetime(row["hora_fonte"], errors="coerce")
                    label = when.strftime("%d/%m às %H:%M") if pd.notna(when) else "Consulta"
                    itens.append(
                        f'<li><div class="person"><div class="person-avatar">☁</div>'
                        f'<div><div class="person-name">{html.escape(label)}</div>'
                        f'<div class="person-meta">{html.escape(str(row["modelo"]))}</div></div></div>'
                        f'<strong>{pt_num(float(row["chance"]) * 100,1)}%</strong></li>'
                    )
                st.markdown(f'<ul class="people-list">{"".join(itens)}</ul>', unsafe_allow_html=True)
            if not historico_ok:
                st.caption("O armazenamento local não estava disponível nesta consulta.")

    charts_a, charts_b = st.columns(2, gap="medium")
    with charts_a:
        with st.container(key="chart_models"):
            st.markdown('<div class="section-title">Comparação dos algoritmos</div>'
                        '<div class="section-subtitle">F1 macro no teste final.</div>', unsafe_allow_html=True)
            render_plot(graf_barras_modelos(meta), "models_main")
    with charts_b:
        with st.container(key="chart_importance"):
            st.markdown('<div class="section-title">Importância das variáveis</div>'
                        '<div class="section-subtitle">Queda no F1 macro ao embaralhar cada variável.</div>', unsafe_allow_html=True)
            render_plot(graf_importancia(meta), "importance_main")


def analytics_page() -> None:
    topbar("Analytics")
    try:
        meta = abrir_metricas()
    except (FileNotFoundError, OSError, ValueError) as err:
        st.warning("Ainda não há métricas disponíveis. Execute o treinamento.")
        st.caption(str(err))
        return
    selected = str(meta["algoritmo_escolhido"])
    met = meta["metricas_teste"][selected]
    cards = [
        ("orange", f"{pt_num(met['acuracia'] * 100,2)}%", "Acurácia", "Teste final", "✓"),
        ("purple", f"{pt_num(met['f1_macro'] * 100,2)}%", "F1 macro", "Teste final", "◎"),
        ("cyan", f"{pt_num(met['recall_chuva'] * 100,2)}%", "Recall de chuva", "Teste final", "☂"),
        ("blue", selected, "Modelo escolhido", "F1 na validação", "★"),
    ]
    st.markdown('<div class="quick-grid">' + ''.join(
        f'<div class="quick-card {color}"><div class="quick-label">{html.escape(label)}</div>'
        f'<div class="quick-value" style="font-size:clamp(1rem,1.85vw,1.65rem)">{html.escape(value)}</div>'
        f'<div class="quick-sub">{html.escape(sub)}</div><div class="quick-icon">{icon}</div></div>'
        for color,value,label,sub,icon in cards
    ) + '</div>', unsafe_allow_html=True)
    a, b = st.columns(2, gap="medium")
    with a:
        with st.container(key="analytics_a"):
            st.markdown('<div class="section-title">Chuva por horário</div>', unsafe_allow_html=True)
            hours = meta["chuva_horaria_historica"]
            fig = plot_base("Incidência histórica", 290)
            fig.add_trace(go.Bar(
                x=[f"{r['hora']:02d}h" for r in hours],
                y=[r["taxa"] * 100 for r in hours],
                marker=dict(color="#8e85ee",cornerradius=5),
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
            ))
            fig.update_yaxes(ticksuffix="%")
            render_plot(fig, "hours_chart")
    with b:
        with st.container(key="analytics_b"):
            st.markdown('<div class="section-title">Chuva por mês</div>', unsafe_allow_html=True)
            months = meta["chuva_mensal_historica"]
            fig = plot_base("Incidência histórica", 290)
            fig.add_trace(go.Bar(
                x=[calendar.month_abbr[int(r["mes"])].upper() for r in months],
                y=[r["taxa"] * 100 for r in months],
                marker=dict(color="#3ec6da",cornerradius=5),
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
            ))
            fig.update_yaxes(ticksuffix="%")
            render_plot(fig, "months_chart")
    c, d = st.columns(2, gap="medium")
    with c:
        with st.container(key="analytics_c"):
            st.markdown('<div class="section-title">Matriz de confusão</div>', unsafe_allow_html=True)
            render_plot(graf_matriz(meta), "confusion_chart")
    with d:
        with st.container(key="analytics_d"):
            st.markdown('<div class="section-title">Explicabilidade</div>', unsafe_allow_html=True)
            render_plot(graf_importancia(meta), "importance_analytics")
    with st.container(key="data_table"):
        st.markdown('<div class="section-title">Comparação final</div>'
                    '<div class="section-subtitle">Resultados medidos no teste cronológico.</div>', unsafe_allow_html=True)
        rows = []
        for name, m in meta["metricas_teste"].items():
            rows.append({"Algoritmo":name,"Acurácia":f"{pt_num(m['acuracia']*100,2)}%",
                         "F1 macro":f"{pt_num(m['f1_macro']*100,2)}%",
                         "Recall chuva":f"{pt_num(m['recall_chuva']*100,2)}%",
                         "Amostras":m["suporte"]})
        st.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch")
    with st.expander("Avaliação por estação e período do dia"):
        period_rows=[]
        for category, groups in meta.get("recortes_desempenho_teste",{}).items():
            for item in groups:
                period_rows.append({"Recorte":category,"Grupo":item["grupo"],"Amostras":item["suporte"],
                                    "F1 macro":f"{pt_num(item['f1_macro']*100,2)}%",
                                    "Recall chuva":f"{pt_num(item['recall_chuva']*100,2)}%"})
        if period_rows:
            st.dataframe(pd.DataFrame(period_rows),hide_index=True,width="stretch")
        st.caption("Diferenças entre períodos revelam limitações temporais, não viés demográfico.")


def report_page() -> None:
    topbar("Projeto UC9")
    with st.container(key="report_intro"):
        st.markdown('<div class="section-title">Objetivo do ChuvaIA</div>'
                    '<div class="helper-text">Prever se ocorrerá chuva nas próximas três horas em Cuiabá, '
                    'sem entrada manual, usando variáveis meteorológicas atuais da Open-Meteo. '
                    'A variável-alvo corresponde a pelo menos 0,2 mm de precipitação acumulada nas três horas seguintes.</div>',
                    unsafe_allow_html=True)
    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(key="report_method"):
            st.markdown('<div class="section-title">Dados e metodologia</div>'
                        '<div class="helper-text">Histórico Open-Meteo/ERA5 (2020–2025). '
                        'Comparação entre baseline, Regressão Logística, Random Forest e Gradient Boosting. '
                        'Random Forest ajustado por GridSearchCV com TimeSeriesSplit. '
                        'Seleção pelo F1 macro de validação e teste cronológico independente.</div>',
                        unsafe_allow_html=True)
    with right:
        with st.container(key="report_limits"):
            st.markdown('<div class="section-title">Limitações</div>'
                        '<div class="helper-text">Dados atuais e históricos resultam de modelos meteorológicos '
                        'potencialmente diferentes. Não há calibração de probabilidades nem dados de radar local. '
                        'O desempenho varia entre estação seca e chuvosa. Não substitui alertas oficiais.</div>',
                        unsafe_allow_html=True)
    try:
        meta = abrir_metricas()
    except (FileNotFoundError, OSError, ValueError) as err:
        st.warning("Execute o treinamento para gerar as métricas reais.")
        st.caption(str(err))
        return
    escolhido = meta["algoritmo_escolhido"]
    selected = meta["metricas_teste"][escolhido]
    cards = [
        ("orange", f"{int(meta['amostras']):,}", "Amostras", "Histórico analisado", "#"),
        ("purple", escolhido, "Modelo", "Selecionado na validação", "★"),
        ("cyan", f"{pt_num(selected['acuracia']*100,2)}%", "Acurácia", "No teste", "✓"),
        ("blue", f"{pt_num(meta['taxa_historica_chuva']*100,1)}%", "Incidência de chuva", "Na base histórica", "☂"),
    ]
    st.markdown('<div class="quick-grid">' + ''.join(
        f'<div class="quick-card {c}"><div class="quick-label">{html.escape(l)}</div>'
        f'<div class="quick-value" style="font-size:clamp(1rem,1.8vw,1.6rem)">{html.escape(v)}</div>'
        f'<div class="quick-sub">{html.escape(sub)}</div><div class="quick-icon">{ico}</div></div>'
        for c,v,l,sub,ico in cards
    )+'</div>',unsafe_allow_html=True)
    cases = int(meta["matriz_confusao_teste"][1][1])
    simulated_min = cases * .20 * 10
    with st.expander("Simulação didática de impacto e análise crítica"):
        st.write(f"O modelo identificou corretamente **{cases}** casos de chuva no conjunto de teste. "
                 f"Se 20% desses casos representassem deslocamentos reagendados com economia de "
                 f"10 minutos por pessoa, a economia simulada seria **{pt_num(simulated_min / 60,1)} horas**. "
                 "Essa premissa é ilustrativa, não um benefício observado.")
        for group in meta.get("recortes_desempenho_teste",{}).values():
            for item in group:
                if str(item.get("grupo", "")).lower().startswith("seca"):
                    st.write(f"Na estação seca, o recall de chuva foi {pt_num(item['recall_chuva']*100,2)}%, "
                             "limitação importante para uso real.")
    st.download_button("Baixar métricas reais (JSON)",
                       data=json.dumps(meta,ensure_ascii=False,indent=2).encode("utf-8"),
                       file_name="metricas_uc9_cuiaba.json",mime="application/json")


def main() -> None:
    if "pagina" not in st.session_state:
        st.session_state.pagina = "dashboard"

    # A barra lateral real posiciona logo + abas DENTRO da área branca.
    # Não desenhe uma coluna branca falsa ao lado da página.
    with st.sidebar:
        sidebar_nav()

    if st.session_state.pagina == "dashboard":
        dashboard_page()
    elif st.session_state.pagina == "analytics":
        analytics_page()
    else:
        report_page()



# --- CHUVAIA CAMPO: PAGINA UNICA ---
ATIVIDADES_CAMPO = {
    "Colheita e produtos expostos": [
        ("Proteger produtos", "Confira cobertura de caixas, sacarias e produtos já colhidos."),
        ("Priorizar áreas cobertas", "Se possível, transfira produtos sensíveis à água para uma área protegida."),
        ("Reavaliar a colheita", "Verifique condições da cultura e do terreno antes de continuar."),
    ],
    "Preparação do solo e máquinas": [
        ("Inspecionar o solo", "Avalie umidade, compactação e acesso de máquinas."),
        ("Proteger equipamentos", "Confira máquinas e materiais vulneráveis à água."),
        ("Organizar alternativas", "Considere inspeção ou manutenção em local protegido."),
    ],
    "Carregamento e transporte": [
        ("Conferir a carga", "Revise lonas e embalagens contra umidade."),
        ("Reduzir a exposição", "Evite manter cargas sensíveis em áreas descobertas."),
        ("Verificar o trajeto", "Consulte as condições das estradas e informações meteorológicas locais."),
    ],
    "Armazenamento temporário": [
        ("Conferir a cobertura", "Inspecione telhados, lonas, calhas e pontos de infiltração."),
        ("Proteger itens vulneráveis", "Cuide de sementes, insumos e produtos sensíveis à umidade."),
        ("Reorganizar materiais", "Avalie levar os produtos para áreas cobertas e ventiladas."),
    ],
}


def moeda(valor):
    return "R$ " + f"{valor:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


@st.fragment(run_every="15min")
def pagina_campo():
    topbar("ChuvaIA Campo")
    st.markdown('<div class="section-subtitle">Proteção da produção rural · Planejamento de atividades em Cuiabá, MT</div>', unsafe_allow_html=True)
    bot, aviso = st.columns([1, 3], vertical_alignment="center")
    with bot:
        if st.button("↻ Recarregar painel", width="stretch", key="update_campo"):
            buscar_tempo.clear()
            st.rerun(scope="fragment")
    with aviso:
        st.caption("Dados Open-Meteo · atualização automática a cada ~15 minutos")
    dados = buscar_tempo()
    atual, prob = dados["atual"], None
    try:
        meta = abrir_metricas()
    except (FileNotFoundError, OSError, ValueError, KeyError):
        meta = None

    idade_min = None
    if atual is not None and dados["consultado_utc"]:
        try:
            idade_min = max(0, int((datetime.now(timezone.utc) - datetime.fromisoformat(
                dados["consultado_utc"])).total_seconds() / 60))
        except (ValueError, TypeError):
            idade_min = None

    if atual is not None and idade_min is not None and idade_min <= 45:
        try:
            pacote, _ = abrir_modelo()
            prob = prever(atual, pacote)
        except (FileNotFoundError, OSError, ValueError, KeyError, IndexError,
                AttributeError, TypeError) as erro:
            st.error("Não foi possível executar o modelo treinado nesta consulta.")
            st.caption(f"Detalhe técnico: {erro}")
    elif atual is not None:
        st.warning("A última consulta foi feita há mais de 45 minutos. "
                   "Para não exibir uma previsão antiga como atual, o risco foi ocultado.")

    if dados["estado"] == "anterior":
        st.warning(f"{dados['mensagem']}. Exibindo dados consultados há "
                   f"{idade_min if idade_min is not None else '?'} minutos, "
                   "de forma identificada. A previsão pode estar desatualizada.")
    elif dados["estado"] == "indisponivel":
        st.warning(f"{dados['mensagem']}. A atualização será tentada novamente "
                   "após a janela de cache de aproximadamente 15 minutos. "
                   "As orientações de proteção rural continuam disponíveis.")

    if prob is not None:
        quick_cards(atual, prob)
        desc, nivel, badge_class, icone = nivel_chuva(prob)
        risco = ("Monitorar condições" if prob < .30 else
                 "Revisar medidas de proteção" if prob < .65 else
                 "Priorizar preparação")
    else:
        risco = "Previsão temporariamente indisponível"
    a, b = st.columns([1.25, 1], gap="medium")
    with a:
        with st.container(key="forecast_panel"):
            st.markdown('<div class="section-title">Previsão para as próximas 3 horas</div>'
                        '<div class="section-subtitle">Estimativa produzida pelo Random Forest para Cuiabá.</div>',
                        unsafe_allow_html=True)
            if prob is None:
                st.info("**Sem previsão atualizada.** A fonte meteorológica está temporariamente "
                        "indisponível ou os dados recentes não puderam ser processados. "
                        "Não usamos valores fictícios para orientar trabalhos no campo.")
            else:
                st.markdown(
                    f'<div class="forecast-copy"><div><div class="big-prob">{pt_num(prob*100)}%</div>'
                    f'<div class="risk-badge {badge_class}">● Nível {html.escape(nivel.lower())}</div></div>'
                    f'<div class="forecast-emoji">{icone}</div></div>'
                    f'<div class="progress-track"><div class="progress-fill" style="width:{prob*100:.2f}%"></div></div>'
                    '<div class="helper-text">Chance estimada de pelo menos 0,2 mm de chuva acumulada '
                    'nas três horas seguintes à hora de referência. '
                    '<strong>Não informa intensidade nem garante que choverá.</strong></div>'
                    f'<div class="small-print">Hora meteorológica: '
                    f'{html.escape(str(atual["time"]).replace("T", " às "))} · Cuiabá. '
                    f'Consulta feita há {idade_min} minuto(s). '
                    f'{"Dados recuperados de consulta anterior." if dados["estado"] == "anterior" else ""}</div>',
                    unsafe_allow_html=True)
    with b:
        with st.container(key="weather_panel"):
            st.markdown('<div class="section-title">🚜 Atividade no campo</div>'
                        '<div class="section-subtitle">Qual atividade você pretende realizar?</div>',
                        unsafe_allow_html=True)
            atividade = st.selectbox("Escolha a atividade", list(ATIVIDADES_CAMPO), key="atividade_campo")
            st.caption("A atividade define orientações preventivas. Ela não altera a previsão do modelo.")
            if prob is None:
                st.info("**Sem nível de risco calculado no momento.** Você ainda pode consultar "
                        "as medidas de proteção para a atividade selecionada; considere previsões "
                        "meteorológicas oficiais antes de tomar decisões.")
            else:
                st.info(f"**{risco}.** Considere as condições reais na propriedade e a previsão meteorológica local.")

    st.markdown('<div class="section-title" style="margin-top:10px">🛡️ Proteção da produção rural</div>'
                '<div class="section-subtitle">Medidas gerais para avaliar antes do trabalho — não são decisões automáticas da IA.</div>',
                unsafe_allow_html=True)
    with st.container(key="recent_panel"):
        for i,(nome, detalhe) in enumerate(ATIVIDADES_CAMPO[atividade],1):
            st.markdown(f'<div style="display:flex;gap:13px;padding:13px 4px;border-bottom:1px solid #edf0f5">'
                        f'<div class="person-avatar">{i}</div><div><div class="person-name">{html.escape(nome)}</div>'
                        f'<div class="person-meta" style="color:#536076!important;line-height:1.6">{html.escape(detalhe)}</div></div></div>',
                        unsafe_allow_html=True)
        st.caption("A ocorrência de chuva, isoladamente, não informa intensidade, umidade do solo nem riscos específicos da atividade.")

    with st.expander("💰 Simulação de impacto econômico (hipotética)"):
        st.write("Simule o prejuízo **caso a chuva afete a atividade**; isto não representa economia comprovada.")
        x,y,z = st.columns(3)
        with x: perda=st.number_input("Prejuízo sem proteção (R$)", 0.0, 1000000.0, 500.0, 50.0)
        with y: parcela=st.number_input("Parcela evitável (%)", 0, 100, 50, 5)
        with z: custo=st.number_input("Custo da proteção (R$)", 0.0, 1000000.0, 80.0, 10.0)
        protegido=perda*(1-parcela/100)+custo
        ca,cb,cc=st.columns(3)
        ca.metric("Sem proteção",moeda(perda))
        cb.metric("Com proteção",moeda(protegido))
        cc.metric("Diferença simulada",moeda(perda-protegido))
        st.caption("Premissas definidas pelo usuário: prejuízo com proteção = prejuízo sem proteção × (1 − fração evitável) + custo. Não é economia estatisticamente prevista.")

    with st.expander("📊 Projeto UC9 · IA, métricas e limitações"):
        st.write("**Problema:** ajudar produtores a priorizar proteção de colheitas, máquinas e mercadorias diante do risco de chuva.")
        st.write("**Método:** Open-Meteo/ERA5 (2020–2025); baseline, Regressão Logística, Random Forest e Gradient Boosting com validação temporal.")
        if meta:
            modelo=meta["algoritmo_escolhido"]
            m=meta["metricas_teste"][modelo]
            c1,c2,c3=st.columns(3)
            c1.metric("Acurácia",f"{pt_num(m['acuracia']*100,2)}%")
            c2.metric("F1 macro",f"{pt_num(m['f1_macro']*100,2)}%")
            c3.metric("Recall chuva",f"{pt_num(m['recall_chuva']*100,2)}%")
            st.caption("Resultados históricos em teste cronológico, sem garantia de desempenho futuro.")
            tabela=[{"Modelo":nome,"F1 macro":f"{pt_num(v['f1_macro']*100,2)}%"} for nome,v in meta["metricas_teste"].items()]
            st.dataframe(pd.DataFrame(tabela),hide_index=True,width="stretch")
        st.write("**Limitações:** probabilidade não calibrada; menor desempenho na estação seca; não estima milímetros, intensidade, condições por propriedade ou umidade do solo. "
                 "Orientações são regras preventivas gerais, e não previsões do Random Forest.")
        st.caption("Fontes meteorológicas: Open-Meteo/ERA5. Demonstração acadêmica SENAC UC9 · Grupo 6.")
    st.caption("Fonte dos dados meteorológicos: Open-Meteo (CC BY 4.0). ChuvaIA Campo é um "
               "protótipo educacional de apoio à decisão e não substitui alertas oficiais ou avaliação agronômica.")


def inicio_campo():
    with st.sidebar:
        if LOGO.exists():
            st.image(str(LOGO), width="stretch")
        else:
            st.markdown("### 🌱 ChuvaIA Campo")
        st.markdown('<div class="nav-section-label">PAINEL ÚNICO</div>'
                    '<div class="sidebar-note" style="margin-top:8px;padding-top:8px">'
                    '<strong>PROTEÇÃO DA PRODUÇÃO RURAL</strong>'
                    '<span>Previsão automática e planejamento preventivo no campo.</span>'
                    '</div><div class="sidebar-note"><strong>LOCALIDADE</strong>'
                    '<span>Cuiabá, Mato Grosso — previsão pontual para a cidade.</span>'
                    '</div><div class="sidebar-note"><strong>SENAC · UC9 · GRUPO 6</strong>'
                    '<span>Machine Learning aplicado ao clima.</span></div>',
                    unsafe_allow_html=True)
    pagina_campo()


inicio_campo()
