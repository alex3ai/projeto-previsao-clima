# 🤖 Projeto Clima: Previsão de Tempo com IA Híbrida

Este projeto é uma aplicação de Data Science para previsão do tempo, servindo como um estudo e componente de portfólio. O objetivo principal é desenvolver um modelo preditivo que utiliza uma **abordagem híbrida**, combinando Redes Neurais Convolucionais (CNNs) para analisar imagens de satélite e Redes Neurais de Grafos (GNNs) para interpretar dados de estações meteorológicas.

A aplicação é apresentada através de um dashboard interativo construído com Streamlit.

## ✨ Features

* **Dashboard Interativo:** Uma interface web para consultar a previsão do tempo.
* **Consulta por Cidade:** Permite ao usuário digitar o nome de uma cidade para obter a previsão.
* **Modelo Híbrido (Em desenvolvimento):** Combina dados visuais (satélite) e tabulares (estações) para uma previsão mais precisa.
* **Visualização de Dados:** Exibe a temperatura prevista e a imagem de satélite correspondente.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.10+
* **Dashboard:** Streamlit
* **Processamento de Dados:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, TensorFlow / PyTorch (para CNN/GNN)
* **Coleta de Dados:** APIs (ex: OpenWeatherMap), Requests

---

## 🚀 Como Executar o Projeto

**Pré-requisitos:** Python 3.10+ e `pip` instalados.

**1. Clone o repositório:**
```bash
git clone [https://github.com/alex3ai/projeto-previsao-clima.git](https://github.com/alex3ai/projeto-previsao-clima.git)
cd projeto-previsao-clima
````

**2. Crie e ative um ambiente virtual:**

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Instale as dependências:**

```bash
pip install -r requirements.txt
```

**4. Execute a aplicação Streamlit:**

```bash
streamlit run app.py
```

Acesse `http://localhost:8501` no seu navegador.

-----

## 📂 Estrutura do Projeto

```
projeto_clima/
│
├── .gitignore
├── app.py                # Script principal do dashboard Streamlit
├── README.md             # Documentação do projeto
├── requirements.txt      # Lista de dependências Python
│
├── coleta_dados.py       # Scripts para coleta de dados de APIs
├── processa_dados.py     # Scripts para limpeza e engenharia de features
├── treina_modelo.py      # Script para treinamento do modelo de ML/DL
│
├── images/               # Imagens estáticas para o app
└── notebooks/            # (Opcional) Jupyter notebooks para exploração
```

## 👨‍💻 Autor: **Alex Mendes**

  * **GitHub:** [@alex3ai](https://www.google.com/search?q=https://github.com/alex3ai)
  * **LinkedIn:** ([Adicione seu link aqui](https://www.linkedin.com/in/alex-mendes-80244b292/))