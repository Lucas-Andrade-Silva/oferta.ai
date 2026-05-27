# Oferta.Ai

Oferta.Ai e uma aplicacao Streamlit para acompanhar ofertas primarias, fundos imobiliarios, credito privado imobiliario e contexto macroeconomico brasileiro. O projeto combina dados publicos da CVM, dados de mercado de FIIs, indicadores macro, cache local de documentos SRE e agentes LLM para relatorios, debates e chat analitico.

> Este projeto nao e recomendacao de investimento. As respostas e analises dependem das fontes carregadas no momento da consulta.

## Principais recursos

- Painel inicial com ofertas primarias de FII, FIAGRO-FII, CRI, CRA, debentures e IPO.
- Ranking de FIIs por valor de mercado, dividend yield, liquidez e menores P/VP validos.
- Visao macro com Selic, CDI, IPCA, IGP-M, IFIX, IMOB e Ibovespa.
- Detalhe de ativo/oferta com dados CVM, dados de mercado e documentos SRE.
- Duelo de ativos usando modelos via OpenRouter.
- Chat Groq com contexto de ofertas, FIIs, informes CVM, dados macro e bases hidratadas localmente.
- Cache local de documentos e PDFs de ofertas em `data/sre_offers`.

## Requisitos

- Python 3.11 ou superior.
- Acesso a internet para carregar fontes publicas e APIs externas.
- Chave Groq para o Chat Groq e relatorios com Groq.
- Chave OpenRouter opcional para o Duelo Ativos.
- Credenciais ANBIMA opcionais para auditoria/apoio.

## Instalacao

No Windows PowerShell, a partir da pasta do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Crie o arquivo local de variaveis de ambiente:

```powershell
Copy-Item .env.example .env
```

Depois edite o `.env` e preencha as chaves que voce for usar:

```env
GROQ_API_KEY=
OPENROUTER_API_KEY=
ANBIMA_CLIENT_ID=
ANBIMA_CLIENT_SECRET=
```

## Como Rodar

Com o ambiente virtual ativado:

```powershell
streamlit run app.py
```

O Streamlit normalmente abre em:

```text
http://localhost:8501
```

## Como Funciona

1. O Streamlit inicia em `app.py`.
2. As funcoes `load_*` carregam dados externos com cache de sessao do Streamlit.
3. As fontes em `fii_analytics/sources` buscam CVM, Fundamentus, BCB/Yahoo Finance, CVM SRE e ANBIMA.
4. A camada `fii_analytics/analysis` trata indicadores, prompts, LangChain, relatorios Groq, chat e debate entre modelos.
5. A camada `fii_analytics/storage` guarda caches locais de documentos SRE e manifestos.
6. O chat pode hidratar dados em `data/chat_hydration` para consultar bases locais sem exigir que o usuario escolha uma pasta.
7. Os dados baixados em runtime ficam fora do git e podem ser recriados pela propria aplicacao.

## Hidratar Dados Locais

Pelo app:

- Abra `Chat Groq`.
- Expanda `Dados usados pelo chat`.
- Clique em `Hidratar dados do chat`.

Isso grava CSVs em:

```text
data/chat_hydration/
```

Para hidratar documentos SRE via script:

```powershell
python scripts/hydrate_sre_cache.py --limit 50 --days 60 --products ALL
```

Para baixar PDFs tambem:

```powershell
python scripts/hydrate_sre_cache.py --limit 20 --days 60 --products FII CRI --download-pdfs
```

Os arquivos sao salvos em:

```text
data/sre_offers/
```

## Testes

```powershell
python -m pytest -q
```

## Estrutura de Pastas

```text
.
+-- app.py                         # Aplicacao Streamlit principal
+-- fii_analytics/
|   +-- analysis/                  # Indicadores, prompts, LangChain, Groq, debate e chat
|   +-- sources/                   # Clientes de dados externos
|   +-- storage/                   # Cache local de documentos SRE
|   +-- config.py                  # Configuracoes e leitura do .env
|   +-- logging_config.py          # Configuracao de logs
+-- scripts/
|   +-- hydrate_sre_cache.py       # Hidrata documentos/ofertas SRE localmente
+-- tests/                         # Testes automatizados
+-- docs/                          # Documentacao tecnica e fluxo do projeto
+-- data/
|   +-- chat_hydration/            # CSVs gerados pelo botao de hidratacao do chat
|   +-- sre_offers/                # Manifestos e PDFs SRE gerados em runtime
+-- logs/                          # Logs locais da aplicacao
+-- requirements.txt               # Dependencias Python
+-- pytest.ini                     # Configuracao de testes
+-- .env.example                   # Modelo de variaveis de ambiente
+-- .gitignore                     # Arquivos ignorados no Git
```

