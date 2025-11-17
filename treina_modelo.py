"""Vamos construir e treinar uma Rede Neural de Grafo (GNN) que aprende a prever a temperatura 
de amanhã (tavg) usando os dados de hoje, incluindo o vetor da imagem."""

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.utils.data import random_split
import numpy as np
from sklearn.metrics import mean_absolute_error  # <-- NOVO IMPORT

# (Se você não tiver o scikit-learn, abra o terminal e rode: pip install scikit-learn)

# 4. Definir a arquitetura da GNN
class GNN_Clima(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GNN_Clima, self).__init__()
        self.fc1 = torch.nn.Linear(num_node_features, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, 1) # Saída é 1 valor: a temperatura

    def forward(self, data):
        x = data.x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# --- INÍCIO DO BLOCO DE EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":

    # 1. Carregar os dados processados
    df = pd.read_pickle('dados_processados.pkl')

    # 2. Preparar os dados para o modelo
    df['tavg_amanha'] = df['tavg'].shift(-1)
    df = df.dropna(subset=['tavg', 'image_features', 'tavg_amanha'])

    if df.empty:
        print("ERRO DE DADOS: O DataFrame ficou vazio...")
        exit() 

    # --- Construção do Grafo ---
    edge_index = torch.tensor([], dtype=torch.long).t().contiguous()
    graph_snapshots = []
    
    for index, row in df.iterrows():
        temp_hoje = float(row['tavg'])
        img_features = list(row['image_features'].astype(np.float32)) 
        node_features_list = [temp_hoje] + img_features
        x = torch.tensor([node_features_list], dtype=torch.float)
        y = torch.tensor([float(row['tavg_amanha'])], dtype=torch.float)
        graph_snapshots.append(Data(x=x, edge_index=edge_index, y=y))

    # 3. Dividir em treino e teste
    if not graph_snapshots:
        print("Erro: A lista graph_snapshots está vazia.")
        exit()
        
    train_size = int(0.8 * len(graph_snapshots))
    test_size = len(graph_snapshots) - train_size
    if test_size == 0 and train_size > 0:
        train_size = train_size - 1
        test_size = 1
    if train_size == 0 or test_size == 0:
        print(f"Erro: Dados insuficientes para dividir em treino/teste.")
        exit()

    train_dataset, test_dataset = random_split(graph_snapshots, [train_size, test_size])

    # 5. Loop de Treinamento
    num_features = graph_snapshots[0].num_node_features
    model = GNN_Clima(num_node_features=num_features, hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss() 

    print("Iniciando o treinamento do modelo...")
    for epoch in range(100):
        model.train()
        total_loss = 0
        for data in train_dataset:
            optimizer.zero_grad()
            out = model(data)
            target = data.y.view_as(out) 
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataset)
        if epoch % 10 == 0 or epoch == 99:
            print(f'Epoch {epoch:03d}, Loss: {avg_loss:.4f}')

    # Salvar o modelo treinado
    torch.save(model.state_dict(), 'modelo_clima.pth')
    print("\nModelo treinado e salvo em 'modelo_clima.pth'")


    # AVALIAÇÃO DO MODELO NO CONJUNTO DE TESTE
    
    print("\nIniciando avaliação no conjunto de teste...")
    model.eval()  # Coloca o modelo em modo de avaliação (desliga dropout, etc.)

    y_true_teste = []  # Lista para as temperaturas reais
    y_pred_teste = []  # Lista para as previsões do modelo

    with torch.no_grad():  # Desliga o cálculo de gradientes (economiza memória)
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