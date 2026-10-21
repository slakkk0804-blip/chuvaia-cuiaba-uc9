"""Gera relatório preenchido somente com métricas calculadas do teste histórico."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARQUIVO = ROOT / "artefatos" / "metricas.json"
SAIDA = ROOT / "RELATORIO_RESULTADOS.md"


def percentual(num):
    return f"{100*num:.2f}%".replace(".", ",")


def produzir(meta: dict) -> str:
    m = meta["metricas_teste"][meta["algoritmo_escolhido"]]
    matriz = meta["matriz_confusao_teste"]
    tn, fp = matriz[0]
    fn, tp = matriz[1]
    cab = "| Algoritmo | Acurácia | F1 macro | Recall de chuva |\n|---|---:|---:|---:|\n"
    def linhas(chave):
        return "".join(f"| {nome} | {percentual(res['acuracia'])} | {percentual(res['f1_macro'])} | {percentual(res['recall_chuva'])} |\n" for nome,res in meta[chave].items())
    importancia = "\n".join(f"- **{v['variavel']}**: redução do F1 macro = {v['queda_f1_macro']:.4f}" for v in meta["importancia_permutacao"][:8])
    vies = "\n".join(f"- **{chave} / {g['grupo']}**: {g['suporte']} horas, F1 macro {percentual(g['f1_macro'])}, recall chuva {percentual(g['recall_chuva'])}." for chave,grupos in meta["recortes_desempenho_teste"].items() for g in grupos)
    out = meta["ausentes_original"]
    ausentes = ", ".join(f"{col} = {n}" for col,n in out.items())
    base = meta["matrizes_por_modelo_teste"].get("Baseline (classe majoritária)", [[0,0],[0,0]])
    _, fp_base = base[0]
    _, tp_base = base[1]
    adesao=.20; evitados=10; transtorno_fp=3
    pontos_modelo = adesao * (tp*evitados - fp*transtorno_fp)
    pontos_base = adesao * (tp_base*evitados - fp_base*transtorno_fp)
    descricao = f"""# Relatório técnico — ChuvaIA Cuiabá / UC9 (rascunho gerado a partir de resultados)

> **Conferir e complementar:** o texto é um rascunho acadêmico baseado em medições feitas pelo script. Os integrantes devem acrescentar interpretação, figuras, revisão e referências conforme instruções da UC9. Não submeter sem revisão.

## 1. Identificação

**Projeto:** ChuvaIA Cuiabá. **Grupo:** 6 (confirmar nomes e divisão de tarefas). **Área:** séries temporais e previsão de tendência, formulado como classificação supervisionada.

## 2. Problema e justificativa

Antecipar se ocorrerá chuva nas próximas três horas em Cuiabá sem que o cidadão precise digitar informações meteorológicas. A aplicação pode apoiar planejamento de horários de deslocamento e atividades externas.

## 3. Solução proposta

O serviço consulta automaticamente a Open-Meteo, transforma o estado meteorológico atual em atributos e o envia a um modelo de Machine Learning já treinado. A previsão exibida é do classificador treinado, não da probabilidade de chuva retornada por um fornecedor externo.

## 4. Dados e variável-alvo

- Origem: **{meta['origem']}**.
- Período efetivo: **{meta['intervalo'][0]} até {meta['intervalo'][1]}**.
- Exemplos válidos: **{meta['amostras']:,}**.
- Definição do alvo: **{meta['definicao_alvo']}**.
- Registros de chuva: **{meta['classes_historicas']['chuva']:,}**; sem chuva: **{meta['classes_historicas']['sem_chuva']:,}**.
- Frequência de chuva no alvo: **{percentual(meta['taxa_historica_chuva'])}**.
- Colunas preditoras derivadas: **{', '.join(meta['colunas_modelo'])}**.

## 5. Análise exploratória (EDA)

A análise automatizada determinou a distribuição do evento de chuva por hora do dia e mês do ano; conferir os gráficos na aba *Analytics*. Além disso, as estatísticas descritivas incluem:

- Temperatura: média **{meta['estatisticas_descritivas']['temperature_2m']['media']:.2f} °C**, mínimo **{meta['estatisticas_descritivas']['temperature_2m']['min']:.2f} °C**, máximo **{meta['estatisticas_descritivas']['temperature_2m']['max']:.2f} °C**.
- Umidade: média **{meta['estatisticas_descritivas']['relative_humidity_2m']['media']:.2f}%**.
- Precipitação horária média **{meta['estatisticas_descritivas']['precipitation']['media']:.3f} mm**.

**Interpretação a adicionar pela equipe:** identificar os meses e as horas com maior ocorrência de chuva e relacionar à sazonalidade local.

## 6. Pré-processamento

Ausentes identificados no CSV original: **{ausentes}**.

O conjunto foi alinhado por hora, rótulos incertos foram descartados, as variáveis de direção do vento, hora e calendário receberam transformações cíclicas e as ausências das entradas foram tratadas por mediana calculada dentro de cada pipeline, sem vazamento de dados. Os extremos plausíveis foram mantidos por poderem representar eventos meteorológicos verdadeiros.

## 7. Modelagem e validação

Foram utilizados um baseline e três algoritmos supervisionados: Regressão Logística, Random Forest e Gradient Boosting. O conjunto foi dividido em ordem temporal: {meta['amostras_por_particao']['treino']:,} exemplos de treino, {meta['amostras_por_particao']['validacao']:,} de validação e {meta['amostras_por_particao']['teste']:,} de teste, com três horas purgadas nas fronteiras.

O Random Forest passou por GridSearchCV com TimeSeriesSplit. Melhores hiperparâmetros: `{meta['gridsearch_rf']['melhores_parametros']}`. Melhor F1 macro médio nos folds: **{percentual(meta['gridsearch_rf']['melhor_f1_macro_cv'])}**.

### Métricas da validação (usadas para escolher modelo)

{cab}{linhas('metricas_validacao')}

### Métricas do teste (resultado final; não usar para escolher modelo)

{cab}{linhas('metricas_teste')}

**Algoritmo escolhido:** **{meta['algoritmo_escolhido']}**. Critério: **{meta['criterio_escolha']}**.

## 8. Métricas e matriz de confusão

No período de teste, o modelo apresentou **acurácia {percentual(m['acuracia'])}**, **acurácia balanceada {percentual(m['acuracia_balanceada'])}**, **F1 macro {percentual(m['f1_macro'])}**, **precisão de chuva {percentual(m['precisao_chuva'])}**, **recall de chuva {percentual(m['recall_chuva'])}** e **F1 de chuva {percentual(m['f1_chuva'])}**.

Matriz de confusão (real nas linhas; previsto nas colunas): TN={tn}, FP={fp}, FN={fn}, TP={tp}. Falsos negativos deixam de alertar chuva real; falsos positivos podem alterar planos desnecessariamente.

## 9. Explicabilidade

A permutação de cada variável sobre uma amostra do teste estimou quanto o desempenho cai ao embaralhar um atributo. Resultados principais:

{importancia}

**Interpretação a adicionar pela equipe:** valores maiores indicam variáveis mais úteis para o classificador, mas não provam causalidade meteorológica.

## 10. Análise de viés temporal

Recortes de validação fora da amostra:

{vies}

A diferença entre períodos representa um risco potencial de erro desigual por sazonalidade e horário. Não há variáveis demográficas ou por bairro, portanto não é possível estudar vieses entre grupos sociais ou intraurbanos.

## 11. Simulação de impacto (hipóteses, não economia real)

Premissas estritamente acadêmicas: cada verdadeiro positivo equivale a uma atividade hipotética sob chuva; **20%** dos avisados mudariam de plano; essa mudança evitaria hipoteticamente **10 minutos** por acerto; cada falso positivo gera um custo hipotético de **3 minutos** para quem reage.

- Modelo selecionado: `20% × (TP × 10 min − FP × 3 min)` = **{pontos_modelo:.1f} minutos líquidos hipotéticos**.
- Baseline: `20% × (TP × 10 min − FP × 3 min)` = **{pontos_base:.1f} minutos líquidos hipotéticos**.
- Diferença: **{pontos_modelo-pontos_base:.1f} minutos hipotéticos** favoráveis ao modelo (se positiva).

Esses resultados dependem integralmente das premissas arbitrárias e **não** constituem comprovação de economia em deslocamentos reais.

## 12. Demonstração

Abrir a URL pública do projeto (inserir após deploy), demonstrar coleta automática de dados, percentual do modelo, gráficos, relatório e erro explícito quando a internet estiver indisponível.

## 13. Limitações e próximos passos

Histórico ERA5 de reanálise e estimativas meteorológicas atuais por previsão são processos diferentes, com possível *dataset shift*. A resolução da grade histórica não garante precipitação em cada bairro. Probabilidades ainda não receberam calibração específica e podem falhar em chuvas intensas. Ampliar base com estação local e avaliar prospectivamente.

## 14. Equipe e referências

Preencher nomes e contribuições de cada integrante.

- SENAC. *UC9 — Projeto Final: ML para Problemas Reais*. PDF de orientações disponibilizado pelo instrutor.
- Open-Meteo. *Historical Weather API*. https://open-meteo.com/en/docs/historical-weather-api.
- Open-Meteo. *Forecast API*. https://open-meteo.com/en/docs.
- scikit-learn. *User Guide*. https://scikit-learn.org/stable/user_guide.html.
- Streamlit. *Community Cloud deployment*. https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy.

**Data de treinamento (UTC):** {meta['data_treinamento_utc']}.
"""
    return descricao


if __name__ == "__main__":
    if not ARQUIVO.exists():
        raise SystemExit("Nenhuma métrica real: execute primeiro `python treinamento.py`.")
    metadata = json.loads(ARQUIVO.read_text(encoding="utf-8"))
    SAIDA.write_text(produzir(metadata), encoding="utf-8")
    print(f"Relatório criado: {SAIDA}")
