"""Testes locais com dados inventados SOMENTE para garantir a correção do código."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modelo_utils import (METEO, COLUNAS_MODELO, preparar_variaveis,
                          construir_dataset, periodo_do_dia, estacao_local)


def pequeno_historico():
    base = pd.DataFrame({"time": pd.date_range("2025-05-01", periods=12, freq="h")})
    for campo in METEO:
        base[campo] = 40.0
    base["precipitation"] = 0.0
    base.loc[3, "precipitation"] = .25
    return base


def test_futuro_e_exclusao_da_precipitacao():
    x, y, base = construir_dataset(pequeno_historico())
    assert len(x) == 9  # últimas 3 h sem futuro completo
    assert all(c in x for c in COLUNAS_MODELO)
    assert "precipitation" not in x.columns
    assert y.tolist()[:5] == [1, 1, 1, 0, 0]
    assert base.loc[0, "chuva_proximas_3h_mm"] == .25


def test_preenchimento_lacuna_sem_virar_hora_vizinha():
    df = pequeno_historico()
    df = df[df["time"] != pd.Timestamp("2025-05-01 03:00")]
    x, y, base = construir_dataset(df)
    # Sem posição observada às 3h, o alvo de 0h-2h não pode ser calculado.
    assert pd.Timestamp("2025-05-01 00:00") not in set(base["time"])
    assert len(x) == len(y)


def test_transformacao_para_consulta_atual():
    dado = {c: 20.0 for c in METEO}
    linha = pd.DataFrame([{"time": "2026-10-21T10:00", **dado}])
    x = preparar_variaveis(linha)
    assert x.shape == (1, len(COLUNAS_MODELO))
    assert np.isfinite(x.to_numpy()).all()
    assert periodo_do_dia(6) == "Manhã (6–11h)"
    assert estacao_local(6).startswith("Seca")
