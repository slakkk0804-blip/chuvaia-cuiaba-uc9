"""Baixa histórico ERA5 da Open-Meteo, por ano, em CSV local."""
from __future__ import annotations
import argparse
from datetime import date
from pathlib import Path
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from modelo_utils import LATITUDE, LONGITUDE, TIMEZONE, COLUNAS_HISTORICAS

URL = "https://archive-api.open-meteo.com/v1/archive"


def baixar_historico(inicio: date, fim: date, destino: Path) -> pd.DataFrame:
    if inicio > fim or fim >= date.today():
        raise ValueError("Use um intervalo válido e totalmente passado")
    destino = Path(destino)
    frames = []
    sessao = requests.Session()
    politica = Retry(total=4, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504])
    sessao.mount("https://", HTTPAdapter(max_retries=politica))
    for ano in range(inicio.year, fim.year + 1):
        d_ini = max(inicio, date(ano, 1, 1))
        d_fim = min(fim, date(ano, 12, 31))
        params = {
            "latitude": LATITUDE, "longitude": LONGITUDE,
            "start_date": d_ini.isoformat(), "end_date": d_fim.isoformat(),
            "hourly": ",".join(COLUNAS_HISTORICAS),
            "timezone": TIMEZONE,
            "models": "era5",  # reanálise consistente, sem mesclar modelos históricos
        }
        print(f"Consultando {ano}: {d_ini} a {d_fim}", flush=True)
        resposta = sessao.get(URL, params=params, timeout=90)
        resposta.raise_for_status()
        dados = resposta.json()
        if dados.get("error") or "hourly" not in dados:
            raise RuntimeError(f"A API não entregou dados de {ano}: {dados.get('reason', dados)}")
        frame = pd.DataFrame(dados["hourly"])
        esperadas = ["time", *COLUNAS_HISTORICAS]
        if any(c not in frame for c in esperadas):
            raise RuntimeError(f"Campos inesperados na resposta de {ano}: {frame.columns.tolist()}")
        frames.append(frame[esperadas])
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("A API não retornou registros")
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino, index=False)
    print(f"OK: {len(df):,} registros. CSV salvo em {destino}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Dados históricos horários de Cuiabá")
    p.add_argument("--inicio", default="2020-01-01")
    p.add_argument("--fim", default="2025-12-31")
    p.add_argument("--saida", default="dados/cuiaba_historico.csv")
    a = p.parse_args()
    baixar_historico(date.fromisoformat(a.inicio), date.fromisoformat(a.fim), Path(a.saida))
