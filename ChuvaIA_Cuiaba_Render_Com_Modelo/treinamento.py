"""Treinamento auditável: divisão temporal, GridSearchCV e métricas reais."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modelo_utils import (
    COLUNAS_MODELO, HORIZONTE_HORAS, LIMIAR_CHUVA_MM,
    construir_dataset, periodo_do_dia, estacao_local,
)


def montar_modelos(rapido: bool = False):
    """O imputador é ajustado DENTRO de cada fold, sem vazamento de dados."""
    simples = lambda modelo: Pipeline([("imputar", SimpleImputer(strategy="median")), ("estimar", modelo)])
    modelos = {
        "Baseline (classe majoritária)": simples(DummyClassifier(strategy="most_frequent")),
        "Regressão Logística": Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
            ("estimar", LogisticRegression(max_iter=1800, class_weight="balanced", random_state=42)),
        ]),
        "Random Forest": simples(RandomForestClassifier(
            n_estimators=120, random_state=42, n_jobs=-1, class_weight="balanced_subsample"
        )),
        "Gradient Boosting": simples(HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.08, max_leaf_nodes=15, random_state=42,
        )),
    }
    grade = {
        "estimar__n_estimators": [60, 120] if rapido else [100, 160],
        "estimar__min_samples_leaf": [1] if rapido else [1, 3],
        "estimar__max_depth": [14],
    }
    return modelos, grade


def metricas(y, pred, proba=None):
    obj = {
        "acuracia": round(float(accuracy_score(y, pred)), 5),
        "acuracia_balanceada": round(float(balanced_accuracy_score(y, pred)), 5),
        "f1_macro": round(float(f1_score(y, pred, average="macro", zero_division=0)), 5),
        "f1_chuva": round(float(f1_score(y, pred, pos_label=1, zero_division=0)), 5),
        "precisao_chuva": round(float(precision_score(y, pred, pos_label=1, zero_division=0)), 5),
        "recall_chuva": round(float(recall_score(y, pred, pos_label=1, zero_division=0)), 5),
        "suporte": int(len(y)),
    }
    if proba is not None:
        obj["brier"] = round(float(brier_score_loss(y, proba)), 5)
    return obj


def teste_subgrupos(base, y_real, y_pred):
    tmp = base[["time"]].copy().reset_index(drop=True)
    tmp["y"] = np.asarray(y_real)
    tmp["pred"] = np.asarray(y_pred)
    tmp["periodo"] = tmp["time"].dt.hour.map(periodo_do_dia)
    tmp["estacao"] = tmp["time"].dt.month.map(estacao_local)
    saida = {}
    for nome in ("periodo", "estacao"):
        saida[nome] = []
        for grupo, frag in tmp.groupby(nome, sort=True):
            if len(frag) < 20:
                continue
            met = metricas(frag["y"], frag["pred"])
            saida[nome].append({"grupo": str(grupo), **met})
    return saida


def executar(arquivo: Path, artefatos: Path, rapido: bool = False):
    df = pd.read_csv(arquivo)
    x, y, base = construir_dataset(df)
    if len(y) < 1000:
        raise ValueError("Dataset insuficiente para o projeto (mínimo: 1.000 linhas)")
    n = len(y)
    lim_treino = int(n * .70)
    lim_validacao = int(n * .85)
    # Purga fronteiriça: os alvos usam chuva das próximas 3 horas.
    treino = np.arange(0, lim_treino - HORIZONTE_HORAS)
    validacao = np.arange(lim_treino, lim_validacao - HORIZONTE_HORAS)
    teste = np.arange(lim_validacao, n)
    x_train, y_train = x.iloc[treino], y.iloc[treino]
    x_val, y_val = x.iloc[validacao], y.iloc[validacao]
    x_test, y_test = x.iloc[teste], y.iloc[teste]
    x_treino_final = x.iloc[:lim_validacao-HORIZONTE_HORAS]
    y_treino_final = y.iloc[:lim_validacao-HORIZONTE_HORAS]
    if any(len(np.unique(a)) < 2 for a in (y_train, y_val, y_test)):
        raise ValueError("Divisão contém somente uma classe. Amplie o período de dados.")

    modelos, grade = montar_modelos(rapido)
    busca = GridSearchCV(
        modelos["Random Forest"], grade,
        scoring="f1_macro", cv=TimeSeriesSplit(n_splits=3, gap=HORIZONTE_HORAS),
        n_jobs=-1, refit=True, error_score="raise", verbose=0,
    )
    print("Ajustando Random Forest com GridSearchCV e validação temporal...", flush=True)
    busca.fit(x_train, y_train)
    modelos["Random Forest"] = busca.best_estimator_
    resumo_validacao = {}
    for nome, modelo in modelos.items():
        if nome != "Random Forest":
            print("Treinando:", nome, flush=True)
            modelo.fit(x_train, y_train)
        previsao = modelo.predict(x_val)
        resumo_validacao[nome] = metricas(y_val, previsao)

    # Escolha por F1 macro na validação. O teste fica isolado até aqui.
    eleito = max(resumo_validacao, key=lambda nome: resumo_validacao[nome]["f1_macro"])
    print("Escolhido na validação:", eleito, flush=True)
    resultados_teste = {}
    matrizes_teste = {}
    for nome, modelo in modelos.items():
        modelo.fit(x_treino_final, y_treino_final)
        prev = modelo.predict(x_test)
        proba = modelo.predict_proba(x_test)[:, 1] if hasattr(modelo, "predict_proba") else None
        resultados_teste[nome] = metricas(y_test, prev, proba)
        matrizes_teste[nome] = confusion_matrix(y_test, prev, labels=[0, 1]).astype(int).tolist()
    vencedor = modelos[eleito]
    pred_final = vencedor.predict(x_test)
    matriz = confusion_matrix(y_test, pred_final, labels=[0, 1]).astype(int).tolist()

    # Permutação em amostra do teste, explicável para qualquer algoritmo vencedor.
    amostra = x_test.sample(n=min(1800, len(x_test)), random_state=42)
    perm = permutation_importance(
        vencedor, amostra, y_test.loc[amostra.index],
        scoring="f1_macro", n_repeats=3, random_state=42, n_jobs=1,
    )
    importantes = sorted(
        ({"variavel": c, "queda_f1_macro": round(float(imp), 5)}
         for c, imp in zip(COLUNAS_MODELO, perm.importances_mean)),
        key=lambda o: o["queda_f1_macro"], reverse=True,
    )
    distribuicao_horas = (
        base.assign(hora=base["time"].dt.hour).groupby("hora")["alvo_chuva"]
        .agg(["count", "mean"]).reset_index()
    )
    distribuicao_meses = (
        base.assign(mes=base["time"].dt.month).groupby("mes")["alvo_chuva"]
        .agg(["count", "mean"]).reset_index()
    )
    ausentes = {c: int(df[c].isna().sum()) for c in df.columns}
    clima_desc = {c: {"media": round(float(df[c].mean()), 3),
                       "min": round(float(df[c].min()), 3),
                       "max": round(float(df[c].max()), 3)}
                  for c in ("temperature_2m", "relative_humidity_2m", "precipitation")}
    resultado = {
        "titulo": "ChuvaIA Cuiabá — UC9 Grupo 6",
        "data_treinamento_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "origem": "Open-Meteo Historical Weather API / ERA5",
        "intervalo": [str(base["time"].iloc[0]), str(base["time"].iloc[-1])],
        "amostras": int(n),
        "amostras_por_particao": {"treino": len(treino), "validacao": len(validacao), "teste": len(teste)},
        "definicao_alvo": f"Precipitação acumulada de t+1 a t+{HORIZONTE_HORAS} horas >= {LIMIAR_CHUVA_MM} mm",
        "taxa_historica_chuva": round(float(y.mean()), 5),
        "classes_historicas": {"sem_chuva": int((y == 0).sum()), "chuva": int((y == 1).sum())},
        "colunas_modelo": COLUNAS_MODELO,
        "algoritmo_escolhido": eleito,
        "criterio_escolha": "Maior F1 macro no período de validação, sem olhar o teste",
        "metricas_validacao": resumo_validacao,
        "metricas_teste": resultados_teste,
        "matriz_confusao_teste": matriz,
        "matrizes_por_modelo_teste": matrizes_teste,
        "gridsearch_rf": {"melhores_parametros": busca.best_params_, "melhor_f1_macro_cv": round(float(busca.best_score_), 5)},
        "importancia_permutacao": importantes,
        "recortes_desempenho_teste": teste_subgrupos(base.iloc[teste], y_test, pred_final),
        "chuva_horaria_historica": [{"hora": int(r.hora), "registros": int(r.count), "taxa": round(float(r.mean), 5)} for r in distribuicao_horas.itertuples()],
        "chuva_mensal_historica": [{"mes": int(r.mes), "registros": int(r.count), "taxa": round(float(r.mean), 5)} for r in distribuicao_meses.itertuples()],
        "ausentes_original": ausentes,
        "estatisticas_descritivas": clima_desc,
        "notas": [
            "Alvo construído somente com precipitação futura; não é usada como entrada.",
            "Dados históricos de reanálise (ERA5); ao vivo, a Open-Meteo fornece estimativas meteorológicas de outros modelos. Pode haver mudança de distribuição.",
            "Probabilidades classificadas não foram calibradas separadamente.",
            "Resultados correspondem ao período e à origem de dados apresentados; não garantem desempenho operacional futuro.",
        ],
    }
    artefatos = Path(artefatos)
    artefatos.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": vencedor, "colunas": COLUNAS_MODELO, "nome": eleito}, artefatos / "modelo_chuva.joblib")
    (artefatos / "metricas.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nRESULTADOS NO TESTE (último período cronológico):")
    for nome, met in resultados_teste.items():
        print(f"  {nome}: acurácia={met['acuracia']:.3f}; F1 macro={met['f1_macro']:.3f}; recall chuva={met['recall_chuva']:.3f}")
    print(f"\nArquivos salvos em {artefatos.resolve()}")
    return resultado


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compara modelos e salva o melhor")
    parser.add_argument("--arquivo", default="dados/cuiaba_historico.csv")
    parser.add_argument("--artefatos", default="artefatos")
    parser.add_argument("--rapido", action="store_true", help="Grade reduzida para teste do fluxo, não é avaliação final")
    args = parser.parse_args()
    executar(Path(args.arquivo), Path(args.artefatos), args.rapido)
