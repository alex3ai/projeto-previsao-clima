import streamlit as st
import pandas as pd
import torch
import requests
from PIL import Image
from io import BytesIO
from torch_geometric.data import Data

# Importar a arquitetura do modelo e funções de processamento
from treina_modelo import GNN_Clima
from processa_dados import extract_features, preprocess

# --- CARREGAR MODELO (COM CACHE) ---
# Usamos @st.cache_resource para carregar o modelo apenas uma vez
@st.cache_resource
def carregar_modelo():
    NUM_FEATURES = 2049
    model = GNN_Clima(num_node_features=NUM_FEATURES, hidden_channels=64)
    
    # Tenta carregar os pesos.
    try:
        model.load_state_dict(torch.load('modelo_clima.pth'))
    except FileNotFoundError:
        st.error("Arquivo 'modelo_clima.pth' não encontrado. Certifique-se de que ele está no mesmo diretório.")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None
        
    model.eval()  # Coloca o modelo em modo de avaliação
    return model

# Carrega o modelo usando a função em cache
model = carregar_modelo()

# Título do Dashboard
st.title('Previsão do Tempo com IA: Fusão de Dados de API e Visão de Satélite')

# --- ENTRADA DE DADOS DO USUÁRIO ---
st.header("Faça uma Previsão")
# Usaremos a API da OpenWeatherMap para dados em tempo real
api_key = "b42f9cc0346d5f0956160078b93dfdea"
city = st.text_input("Digite o nome da cidade (ex: Sao Paulo):", "Sao Paulo")

# Só executa se o modelo foi carregado com sucesso
if model is not None and st.button("Prever Temperatura de Amanhã"):
    # 1. Obter dados atuais da API
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url).json()
        
        if response['cod'] == 200:
            temp_atual = response['main']['temp']
            st.write(f"Temperatura atual em {city}: **{temp_atual:.2f}°C**")

            # --- CORREÇÃO: CARREGAR IMAGEM LOCALMENTE ---
            st.info("Nota: Usando uma imagem de satélite local para a demonstração.")
            
            try:
                # Tenta abrir o arquivo local que você salvou
                img = Image.open('exemplo_satelite.jpg').convert('RGB')

                # 3. Processar a imagem para extrair features
                #    (Passamos o objeto 'img' direto, otimizado)
                image_features = extract_features(img)

                # 4. Preparar os dados para o modelo no formato de grafo
                node_features_list = [temp_atual] + list(image_features)
                x = torch.tensor([node_features_list], dtype=torch.float)
                
                # Para um único nó, o edge_index é vazio
                edge_index = torch.tensor([[], []], dtype=torch.long)
                
                # Usa a classe Data que importamos
                data_input = Data(x=x, edge_index=edge_index)

                # 5. Fazer a predição
                with torch.no_grad():
                    previsao = model(data_input)
                    temp_prevista = previsao.item()
                
                st.success(f"**Previsão de temperatura para amanhã: {temp_prevista:.2f}°C**")
                st.image(img, caption="Imagem de Satélite Local Utilizada")
            
            except FileNotFoundError:
                st.error("Erro: Arquivo 'exemplo_satelite.jpg' não encontrado.")
                st.warning("Certifique-se de que você baixou a imagem e a salvou na mesma pasta do app.py.")
            except Exception as e:
                st.error(f"Erro ao carregar ou processar a imagem local: {e}")

        else:
            st.error(f"Não foi possível encontrar a cidade: {response.get('message', 'Erro desconhecido')}")
    
    except requests.exceptions.RequestException as e:
        st.error(f"Erro de conexão com a API: {e}")
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado: {e}")

elif model is None:
    st.warning("O modelo não pôde ser carregado. A aplicação não pode fazer previsões.")