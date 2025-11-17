import pandas as pd
import torch
from torchvision import models, transforms
from PIL import Image
import networkx as nx
import streamlit as st  # <-- IMPORTADO PARA O CACHE

# VISÃO COMPUTACIONAL

#Mover o carregamento do modelo para uma função com cache
@st.cache_resource
def carregar_modelo_resnet():
    """Carrega o modelo ResNet50 e o coloca em cache."""
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = torch.nn.Sequential(*(list(resnet.children())[:-1])) # Remove a última camada
    resnet.eval() # Coloca o modelo em modo de avaliação
    return resnet

# 2. Definir as transformações da imagem
# (Isso é leve, pode ficar aqui)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# A função deve usar o cache e aceitar objetos ---
def extract_features(image_input):
    """Carrega uma imagem (caminho ou objeto PIL), processa e extrai um vetor."""
    
    # Carrega o modelo do cache (rápido após a 1ª vez)
    resnet = carregar_modelo_resnet()
    
    try:
        # Verifica se o input é um caminho (str) ou um objeto Imagem
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        else:
            img = image_input # Assume que é um objeto PIL
            
        img_t = preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0)
        with torch.no_grad():
            features = resnet(batch_t)
        # Converte o tensor de saída em um vetor flat
        return features.squeeze().numpy()
    except Exception as e:
        print(f"Erro ao processar a imagem {image_input}: {e}")
        return None

#  script de processamento
if __name__ == "__main__":
    # Carrega o dataset mestre
    df = pd.read_csv('dataset_mestre.csv')

    # Aplica a extração de características em todas as imagens
    print("Extraindo características das imagens de satélite...")
    # A função 'extract_features' agora funciona aqui (passando o caminho)
    df['image_features'] = df['image_path'].apply(extract_features)
    df = df.dropna(subset=['image_features'])

    # GRAFOS (Comentários - sem mudança)
    # ...

    if df.empty:
        print("\n!!! ATENÇÃO: O DataFrame processado está vazio. !!!")
        print("A função 'extract_features' falhou para todas as imagens.")
        print("O arquivo 'dados_processados.pkl' NÃO foi criado/atualizado.")
    else:
        df.to_pickle('dados_processados.pkl')
        print(f"\nDados processados salvos com {len(df)} linhas em 'dados_processados.pkl'")