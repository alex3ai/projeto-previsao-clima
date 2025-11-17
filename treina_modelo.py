import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.utils.data import random_split
import numpy as np
from sklearn.metrics import mean_absolute_error

# Arquitetura da GNN (sem mudanças)
class GNN_Clima(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GNN_Clima, self).__init__()
        self.fc1 = torch.nn.Linear(num_node_features, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x = data.x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # 1. Carregar os dados processados (que já contêm as novas colunas)
    df = pd.read_pickle('dados_processados.pkl')

    # --- NOVO: DEFINIR AS FEATURES DE ENTRADA ---
    # Estas são as colunas que nosso modelo usará para aprender
    feature_cols = ['tavg', 'wspd', 'pres'] # Adicionamos vento e pressão
    # --------------------------------------------

    # 2. Preparar os dados para o modelo
    df['tavg_amanha'] = df['tavg'].shift(-1)
    
    # Limpa Nulos em qualquer uma das colunas que vamos usar
    df.dropna(subset=feature_cols + ['image_features', 'tavg_amanha'], inplace=True)

    if df.empty:
        print("ERRO DE DADOS: O DataFrame ficou vazio após a limpeza.")
        exit()

    # --- Construção do Grafo ---
    edge_index = torch.tensor([], dtype=torch.long).t().contiguous()
    graph_snapshots = []
    
    for index, row in df.iterrows():
        # --- MUDANÇA PRINCIPAL: CONSTRUIR O VETOR DE FEATURES ---
        # Pega os valores das colunas de features (tavg, wspd, pres)
        weather_features = list(row[feature_cols].values.astype(np.float32))
        img_features = list(row['image_features'].astype(np.float32))
        
        # Combina tudo: [temp, vento, pressão] + [vetor da imagem]
        node_features_list = weather_features + img_features
        # -------------------------------------------------------------
        
        x = torch.tensor([node_features_list], dtype=torch.float)
        y = torch.tensor([[float(row['tavg_amanha'])]], dtype=torch.float)
        graph_snapshots.append(Data(x=x, edge_index=edge_index, y=y))

    # O resto do script permanece praticamente o mesmo...
    # (O código abaixo já é robusto o suficiente para lidar com as mudanças)

    if not graph_snapshots:
        print("Erro: A lista graph_snapshots está vazia.")
        exit()
        
    train_size = int(0.8 * len(graph_snapshots))
    test_size = len(graph_snapshots) - train_size
    # ... (código de divisão de treino/teste sem mudanças)
    train_dataset, test_dataset = random_split(graph_snapshots, [train_size, test_size])

    # --- MUDANÇA SUTIL: O NÚMERO DE FEATURES É AGORA DINÂMICO ---
    num_features = graph_snapshots[0].num_node_features
    print(f"\nO modelo será treinado com {num_features} features de entrada por dia.")
    # -------------------------------------------------------------

    model = GNN_Clima(num_node_features=num_features, hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    # Loop de Treinamento e Avaliação
    print("Iniciando o treinamento do modelo...")
    model.train() # Coloca o modelo em modo de treinamento

    for epoch in range(100): # 100 épocas de treino
        total_loss = 0
        for data in train_dataset:
            optimizer.zero_grad()       # Limpa os gradientes
            
            output = model(data)        # Faz a previsão
            loss = criterion(output, data.y) # Calcula o erro (loss)
            
            loss.backward()             # Calcula os gradientes (backpropagation)
            optimizer.step()            # Atualiza os pesos do modelo
            
            total_loss += loss.item()
        
        # Imprime o erro médio da época
        if (epoch + 1) % 10 == 0:
            print(f"Época {epoch+1:03d} | Erro Médio (Loss): {total_loss / len(train_dataset):.4f}")

    print("Treinamento concluído.")
    
    # Salva o modelo TREINADO
    torch.save(model.state_dict(), 'modelo_clima.pth')
    print(f"\nModelo treinado e salvo em 'modelo_clima.pth'")

    # --- AVALIAÇÃO DO MODELO NO CONJUNTO DE TESTE ---
    
    print("\nIniciando avaliação no conjunto de teste...")
    model.eval()  # Coloca o modelo em modo de avaliação

    y_true_teste = []  # Lista para as temperaturas reais
    y_pred_teste = []  # Lista para as previsões do modelo

    with torch.no_grad():  # Desliga o cálculo de gradientes
        for data in test_dataset:
            previsao = model(data)
            
            # Armazena a previsão (converte tensor para um número)
            y_pred_teste.append(previsao.item()) 
            
            # Armazena o valor real (converte tensor para um número)
            y_true_teste.append(data.y.item())

    # Calcular o MAE
    mae = mean_absolute_error(y_true_teste, y_pred_teste)

    print("-" * 40)
    print(f"AVALIAÇÃO DO MODELO CONCLUÍDA")
    print(f"Erro Médio Absoluto (MAE) no teste: {mae:.2f}°C")
    print(f"-> Isso significa que, em média, o modelo erra")
    print(f"   suas previsões em {mae:.2f}°C (para mais ou para menos).")
    print("-" * 40)

    # Mostrar alguns exemplos de previsão vs. real
    print("\nExemplos de Previsão (do conjunto de teste):")
    for i in range(min(5, len(y_true_teste))): # Mostra os 5 primeiros
        print(f"  - Dia {i+1} -> Real: {y_true_teste[i]:.2f}°C | Previsão: {y_pred_teste[i]:.2f}°C")