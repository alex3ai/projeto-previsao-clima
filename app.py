import streamlit as st
import pandas as pd
import torch
import requests
from PIL import Image
from io import BytesIO
from torch_geometric.data import Data
from treina_modelo import GNN_Clima
from processa_dados import extract_features, preprocess

# --- MUDANÇA: ATUALIZAR O NÚMERO DE FEATURES ---
# temp, vento, pressão + vetor da imagem
NUM_FEATURES = 3 + 2048
# -----------------------------------------------

@st.cache_resource
def carregar_modelo():
    model = GNN_Clima(num_node_features=NUM_FEATURES, hidden_channels=64)
    try:
        model.load_state_dict(torch.load('modelo_clima.pth'))
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None
    model.eval()
    return model

model = carregar_modelo()

st.title('Previsão do Tempo com IA (Versão Aprimorada)')

st.header("Faça uma Previsão")
api_key = "b42f9cc0346d5f0956160078b93dfdea"
city = st.text_input("Digite o nome da cidade (ex: Sao Paulo):", "Sao Paulo")

if model is not None and st.button("Prever Temperatura de Amanhã"):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url).json()
        
        if response['cod'] == 200:
            # --- MUDANÇA: COLETAR AS NOVAS FEATURES DA API ---
            temp_atual = response['main']['temp']
            vento_vel = response['wind']['speed'] # m/s
            pressao = response['main']['pressure'] # hPa

            st.write(f"**Dados Atuais em {city}:**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Temperatura", f"{temp_atual:.1f}°C")
            col2.metric("Vento", f"{vento_vel:.1f} m/s")
            col3.metric("Pressão", f"{pressao} hPa")
            # ---------------------------------------------------

            try:
                img = Image.open('exemplo_satelite.jpg').convert('RGB')
                image_features = extract_features(img)

                # --- MUDANÇA CRÍTICA: MONTAR O VETOR NA ORDEM CORRETA ---
                weather_features = [temp_atual, vento_vel, pressao]
                node_features_list = weather_features + list(image_features)
                # ---------------------------------------------------------
                
                x = torch.tensor([node_features_list], dtype=torch.float)
                edge_index = torch.tensor([[], []], dtype=torch.long)
                data_input = Data(x=x, edge_index=edge_index)

                with torch.no_grad():
                    previsao = model(data_input)
                    temp_prevista = previsao.item()
                
                st.success(f"**Previsão de temperatura para amanhã: {temp_prevista:.2f}°C**")
                st.image(img, caption="Imagem de Satélite Local Utilizada")
            
            except Exception as e:
                st.error(f"Erro ao processar a imagem ou fazer a previsão: {e}")

        else:
            st.error(f"Não foi possível encontrar a cidade: {response.get('message', 'Erro desconhecido')}")
    
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado: {e}")