# 🌦️ Previsão do Tempo com IA: Grafos e Visão Computacional

![Status do Projeto](https://img.shields.io/badge/status-concluído-green)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Um projeto completo de Data Science que prevê a temperatura de amanhã utilizando uma abordagem híbrida. O modelo combina dados de séries temporais, a análise de padrões atmosféricos em imagens de satélite e **dinâmicas atmosféricas** (vento e pressão) para capturar mudanças climáticas de forma mais realista.

---

### 🎥 Demonstração

![GIF do Dashboard](https://i.imgur.com/fRC0xat.png)

---

## 📖 Índice

*   [Sobre o Projeto](#-sobre-o-projeto)
*   [Arquitetura do Projeto](#-arquitetura-do-projeto)
*   [Principais Tecnologias](#-principais-tecnologias)
*   [Como Executar](#-como-executar)
    *   [Pré-requisitos](#pré-requisitos)
    *   [Instalação](#instalação)
    *   [Configuração das APIs](#configuração-das-apis)
*   [Ordem de Execução dos Scripts](#-ordem-de-execução-dos-scripts)
*   [Resultados do Modelo](#-resultados-do-modelo)
*   [Melhorias Futuras](#-melhorias-futuras)
*   [Licença](#-licença)
*   [Contato](#-contato)

---

## 🎯 Sobre o Projeto

Modelos tradicionais de previsão do tempo frequentemente se baseiam apenas na temperatura passada. Este projeto avança essa abordagem ao incorporar as **causas** das mudanças de tempo.

A hipótese central é que, ao combinar **dados locais** (temperatura), **vetores de mudança** (velocidade do vento e pressão atmosférica) e **padrões visuais de larga escala** (extraídos de imagens de satélite com Redes Neurais Convolucionais), podemos criar um modelo preditivo que não apenas segue tendências, mas também antecipa mudanças abruptas com maior precisão.

---

## 🏗️ Arquitetura do Projeto

O fluxo de dados do projeto é dividido em 5 módulos sequenciais:

1.  **Coleta e Limpeza de Dados (`coleta_dados.py`):**
    *   Dados históricos (temperatura, vento, pressão) são coletados da API da **Meteostat**.
    *   Um passo de limpeza (`fillna`) é aplicado para garantir a consistência dos dados de séries temporais.

2.  **Busca e Sincronização de Imagens (`sincroniza_tudo.py`):**
    *   Para cada registro diário, uma imagem de satélite correspondente é baixada da API **Sentinel Hub**.
    *   É gerado um `dataset_mestre.csv` que une os dados climáticos ao caminho da imagem.

3.  **Processamento e Extração de Features (`processa_dados.py`):**
    *   **Visão Computacional:** Uma CNN pré-treinada (**ResNet50**) analisa cada imagem e extrai um "vetor de características" que representa numericamente os padrões visuais.
    *   Os dados são salvos em um arquivo `dados_processados.pkl`.

4.  **Treinamento do Modelo (`treina_modelo.py`):**
    *   Uma **Rede Neural** é construída com **PyTorch**.
    *   O modelo aprende a prever a temperatura de amanhã usando um vetor de entrada combinado: `[temp_hoje, vento, pressão] + [vetor_da_imagem]`.
    *   O modelo treinado é salvo como `modelo_clima.pth`.

5.  **Dashboard Interativo (`app.py`):**
    *   Uma aplicação web com **Streamlit** carrega o modelo treinado.
    *   A aplicação busca dados em **tempo real** (temperatura, vento, pressão) da API **OpenWeatherMap** e usa o modelo para gerar previsões sob demanda.

---

## 🛠️ Principais Tecnologias

*   **Linguagem:** Python
*   **Análise de Dados:** Pandas, NumPy
*   **Deep Learning:** PyTorch, PyTorch Geometric
*   **Visão Computacional:** Torchvision, OpenCV, Pillow
*   **APIs e Coleta de Dados:** Meteostat, SentinelHub-py, Requests
*   **Dashboard:** Streamlit
*   **Machine Learning:** Scikit-learn

---

## 🚀 Como Executar

Siga os passos abaixo para configurar e executar o projeto em seu ambiente local.

### Pré-requisitos

*   Python 3.9 ou superior
*   Git

### Instalação

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/alex3ai/[SEU-REPOSITORIO-AQUI].git
    cd [SEU-REPOSITORIO-AQUI]
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    # Para Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Para macOS/Linux
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Instale as dependências a partir do arquivo `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuração das APIs

1.  **Sentinel Hub:**
    *   Crie uma conta no [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/).
    *   No seu Dashboard > "User Settings", crie um "OAuth Client".
    *   Abra o arquivo `busca_imagens.py` e insira seu `Client ID` e `Client Secret`.

2.  **OpenWeatherMap:**
    *   Crie uma conta no [OpenWeatherMap API](https://openweathermap.org/api).
    *   Abra o arquivo `app.py` e insira sua chave de API na variável `api_key`.

---

## ▶️ Ordem de Execução dos Scripts

Para treinar o modelo do zero, os scripts devem ser executados na seguinte ordem:

1.  **Coletar dados climáticos históricos:** `python coleta_dados.py`
2.  **Baixar e sincronizar as imagens (pode levar um tempo):** `python sincroniza_tudo.py`
3.  **Processar os dados e extrair features das imagens:** `python processa_dados.py`
4.  **Treinar o modelo:** `python treina_modelo.py`
5.  **Iniciar o Dashboard interativo:** `streamlit run app.py`

---

## 📊 Resultados do Modelo

O modelo foi avaliado em um conjunto de teste (20% dos dados). O **Erro Médio Absoluto (MAE)**, que mede a diferença média entre o valor real e o previsto, foi a métrica escolhida por sua fácil interpretação.

*   **Erro Médio Absoluto (MAE) no Teste:**
   
    **1.45°C** para mais ou para menos.

#### Exemplos de Previsão (do conjunto de teste):

  - Dia 1 -> Real: 17.60°C | Previsão: 16.54°C
  - Dia 2 -> Real: 22.70°C | Previsão: 21.49°C
  - Dia 3 -> Real: 16.60°C | Previsão: 16.54°C
  - Dia 4 -> Real: 21.60°C | Previsão: 20.08°C
  - Dia 5 -> Real: 22.50°C | Previsão: 19.99°C

---

## 🔮 Melhorias Futuras

*   **Modelo de Grafo Multi-Estação:** Expandir a coleta de dados para múltiplas estações e construir um grafo real para que o modelo aprenda a influência climática entre regiões vizinhas.
*   **Imagens de Satélite Dinâmicas no App:** Implementar a busca de imagens em tempo real no dashboard para uma previsão totalmente dinâmica.
*   **Deploy do Modelo:** Empacotar o projeto em um container Docker e fazer o deploy em uma plataforma de nuvem (AWS, GCP, Heroku).
*   **Sistema de Logging:** Implementar um sistema de logging para monitorar requisições de API e previsões do modelo em um ambiente de produção.

---

## 📄 Licença

Este projeto está sob a licença MIT.

---

## 👤 Contato

**Alex Mendes**

*   **GitHub:** [@alex3ai](https://github.com/alex3ai)
*   **LinkedIn:** [Alex Mendes](https://www.linkedin.com/in/alex-mendes-80244b292/)

Sinta-se à vontade para entrar em contato!