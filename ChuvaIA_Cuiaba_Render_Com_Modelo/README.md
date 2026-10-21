# ChuvaIA Cuiabá — Projeto UC9 · Grupo 6

Dashboard automático de previsão de chuva nas próximas 3 horas para Cuiabá. Projeto educacional.

## O que este pacote contém
- `app.py`: interface aprovada pelo grupo, incluindo navegação lateral, gráficos e histórico local.
- `modelo_utils.py`: preparação das mesmas variáveis usadas no treinamento.
- `artefatos/modelo_chuva.joblib`: Random Forest real recebido do usuário e compactado sem alteração lógica (32 MB → ~10 MB).
- `artefatos/metricas.json`: métricas reais do treinamento.
- `assets/logo_chuvaia.png`: logo aprovada.
- `.python-version`: Python 3.14.3 (Render).
- `requirements.txt`: scikit-learn 1.9.1, versão serializada no modelo.

## Publicação no Render
1. **Descompacte** o ZIP fornecido pelo ChatGPT. NÃO envie o ZIP dentro do repositório.
2. Crie um repositório GitHub chamado `chuvaia-cuiaba-uc9`.
3. Envie **o conteúdo da pasta** do projeto ao GitHub; `app.py` deve estar na raiz. Inclua `artefatos/modelo_chuva.joblib`, a pasta `assets` e `.streamlit/config.toml`.
4. Envie o endereço do repositório nesta conversa. O ChatGPT poderá configurar o serviço Render, desde que você confirme o workspace desejado.

**Configuração de serviço Render:**
- Tipo: Web Service, runtime Python, plano Free (caso disponível)
- Build: `pip install -r requirements.txt`
- Start: `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`
- Python: lê `.python-version` (3.14.3)
- `app.py` na raiz do repositório.

## Avisos
- `modelo_chuva.joblib` é um arquivo Python serializado; execute somente arquivos de origem confiável.
- Histórico usa SQLite local e **pode desaparecer** em reinicializações da hospedagem gratuita. Para persistência real use banco externo.
- A atualização automática depende do site estar aberto e dos dados mais recentes da API.
- Previsões são educacionais; não substituem avisos oficiais e suas probabilidades não foram calibradas separadamente.
