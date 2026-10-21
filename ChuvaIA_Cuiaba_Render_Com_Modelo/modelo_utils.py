"""Contrato de dados único para treino e inferência (evita divergência de features)."""
from __future__ import annotations
import numpy as np
import pandas as pd

LATITUDE = -15.601
LONGITUDE = -56.097
TIMEZONE = "America/Cuiaba"
LIMIAR_CHUVA_MM = 0.2
HORIZONTE_HORAS = 3
METEO = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "surface_pressure", "cloud_cover", "wind_speed_10m", "wind_direction_10m",
]
COLUNAS_HISTORICAS = METEO + ["precipitation"]
COLUNAS_MODELO = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "surface_pressure", "cloud_cover", "wind_speed_10m",
    "direcao_vento_sin", "direcao_vento_cos", "hora_sin", "hora_cos",
    "ano_sin", "ano_cos", "fim_semana",
]


def preparar_variaveis(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma campos meteorológicos e timestamp em variáveis preditoras.

    Importante: NÃO inclui precipitação presente/futura como feature.
    """
    if "time" not in df.columns:
        raise ValueError("Dataset precisa conter coluna 'time'")
    faltantes = sorted(set(METEO) - set(df.columns))
    if faltantes:
        raise ValueError(f"Variáveis meteorológicas ausentes: {faltantes}")
    tempos = pd.to_datetime(df["time"], errors="coerce")
    if tempos.isna().any():
        raise ValueError("Coluna time tem datas inválidas")
    x = df[METEO].apply(pd.to_numeric, errors="coerce").copy()
    angulo = np.deg2rad(x.pop("wind_direction_10m"))
    x["direcao_vento_sin"] = np.sin(angulo)
    x["direcao_vento_cos"] = np.cos(angulo)
    horas = tempos.dt.hour + tempos.dt.minute / 60
    x["hora_sin"] = np.sin(2 * np.pi * horas / 24)
    x["hora_cos"] = np.cos(2 * np.pi * horas / 24)
    dias = tempos.dt.dayofyear
    x["ano_sin"] = np.sin(2 * np.pi * dias / 365.25)
    x["ano_cos"] = np.cos(2 * np.pi * dias / 365.25)
    x["fim_semana"] = (tempos.dt.weekday >= 5).astype(int)
    return x.loc[:, COLUNAS_MODELO].reset_index(drop=True)


def construir_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Alvo: soma das precipitações em t+1, t+2 e t+3 >= limiar.

    Reindexação horária evita tratar tempos desconexos como horas vizinhas.
    As três últimas linhas e lacunas no alvo não são rotuladas.
    """
    if "precipitation" not in df.columns:
        raise ValueError("Faltou a coluna precipitation para calcular rótulos")
    base = df.copy()
    base["time"] = pd.to_datetime(base["time"], errors="raise")
    base = base.sort_values("time").drop_duplicates("time").set_index("time").asfreq("h")
    base["precipitation"] = pd.to_numeric(base["precipitation"], errors="coerce")
    proximas = pd.concat(
        [base["precipitation"].shift(-h) for h in range(1, HORIZONTE_HORAS + 1)],
        axis=1,
    )
    total_futuro = proximas.sum(axis=1, min_count=HORIZONTE_HORAS)
    base["chuva_proximas_3h_mm"] = total_futuro
    base = base.loc[total_futuro.notna()].reset_index()
    base["alvo_chuva"] = (base["chuva_proximas_3h_mm"] >= LIMIAR_CHUVA_MM).astype(int)
    x = preparar_variaveis(base)
    y = base["alvo_chuva"].reset_index(drop=True)
    return x, y, base


def periodo_do_dia(hora: int) -> str:
    if 0 <= hora < 6:
        return "Madrugada (0–5h)"
    if 6 <= hora < 12:
        return "Manhã (6–11h)"
    if 12 <= hora < 18:
        return "Tarde (12–17h)"
    return "Noite (18–23h)"


def estacao_local(mes: int) -> str:
    # Aproximação didática para Cuiabá; não substitui análise climatológica.
    return "Seca (mai–set)" if 5 <= mes <= 9 else "Chuvosa (out–abr)"
