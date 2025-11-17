"""Vamos ler nosso CSV de clima e, para cada dia, buscar a imagem correspondente."""

import pandas as pd
from tqdm import tqdm # Para ter uma barra de progresso
# Importa a função que criamos no arquivo anterior
from busca_imagens import get_satellite_image 

# Supondo que a estação de SP está nessas coordenadas
LATITUDE_ESTACAO = -23.5505
LONGITUDE_ESTACAO = -46.6333

df = pd.read_csv('dados_climaticos_sp.csv', parse_dates=['time'])
df = df.dropna(subset=['tavg']) # Remove dias sem dados de temperatura

image_paths = []
# tqdm cria uma barra de progresso para o loop
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    date_str = row['time'].strftime('%Y-%m-%d')
    path = get_satellite_image(LATITUDE_ESTACAO, LONGITUDE_ESTACAO, date_str)
    image_paths.append(path)

df['image_path'] = image_paths
df_final = df.dropna(subset=['image_path']) # Remove linhas onde não conseguimos baixar a imagem

df_final.to_csv('dataset_mestre.csv', index=False)
print("\nDataset mestre criado com sucesso: 'dataset_mestre.csv'")

# VERIFICAÇÃO !!

if df_final.empty:
    print("\n!!! ATENÇÃO: O DataFrame 'df_final' está vazio. !!!")
    print("Nenhuma imagem foi baixada com sucesso ou 'dados_climaticos_sp.csv' não tinha dados válidos em 'tavg'.")
    print("O arquivo 'dataset_mestre.csv' NÃO foi criado/atualizado.")
else:
    df_final.to_csv('dataset_mestre.csv', index=False)
    print(f"\nDataset mestre criado com sucesso com {len(df_final)} linhas: 'dataset_mestre.csv'")

"""Agora temos um arquivo dataset_mestre.csv e uma pasta images/. O CSV contém os dados do clima e uma 
coluna que aponta para a imagem de satélite daquele dia exato. Esta é nossa fonte de verdade para o resto 
do projeto."""