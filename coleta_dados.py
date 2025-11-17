import pandas as pd
from datetime import datetime
from meteostat import Point, Daily

# 1. Defina o período de interesse
start = datetime(2022, 1, 1)
end = datetime(2023, 12, 31)

# 2. Defina um ponto central para a busca de estações (Ex: Cidade de São Paulo)
sao_paulo_centro = Point(-23.5505, -46.6333)

# 3. Busque os dados diários para a estação mais próxima
data = Daily(sao_paulo_centro, start, end)
data = data.fetch()

# 4. LIMPEZA DOS DADOS (NOVO PASSO CRÍTICO!)
# Seleciona as colunas de interesse. A umidade (rhum) pode não estar sempre disponível.
colunas_interesse = ['tavg', 'prcp', 'wspd', 'pres']
for col in colunas_interesse:
    if col not in data.columns:
        # Se uma coluna não existir, a criamos com Nulos para o ffill funcionar
        data[col] = pd.NA

# Preenche valores ausentes com o último valor válido. Essencial para consistência.
data[colunas_interesse] = data[colunas_interesse].fillna(method='ffill')
# Se ainda houver nulos no início, preenche com o próximo valor válido
data[colunas_interesse] = data[colunas_interesse].fillna(method='bfill')


# 5. Explore e salve os dados
print("Dados coletados e limpos:")
print(data.head())
print("\nInformações sobre os dados:")
data.info() # Verificamos se há nulos restantes

data.to_csv('dados_climaticos_sp.csv')
print("\nDados climáticos aprimorados salvos em 'dados_climaticos_sp.csv'")