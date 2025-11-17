import pandas as pd
from datetime import datetime
from meteostat import Point, Daily

# 1. Defina o período de interesse
start = datetime(2022, 1, 1)
end = datetime(2023, 12, 31)

# 2. Defina um ponto central para a busca de estações (Ex: Cidade de São Paulo)
# Latitude: -23.5505, Longitude: -46.6333
sao_paulo_centro = Point(-23.5505, -46.6333)

# 3. Busque os dados diários para a estação mais próxima
# A biblioteca vai encontrar a estação mais próxima do ponto fornecido
data = Daily(sao_paulo_centro, start, end)
data = data.fetch()

# 4. Explore e salve os dados
print("Dados coletados:")
print(data.head())

data.to_csv('dados_climaticos_sp.csv')
print("\nDados climáticos salvos em 'dados_climaticos_sp.csv'")

"""*   Usamos a `meteostat` para buscar 2 anos de dados climáticos diários 
(temperatura média `tavg`, precipitação `prcp`, etc.) da estação mais próxima ao 
centro de São Paulo e salvamos em um CSV. Para um projeto ainda mais robusto, 
repetiriamos isso para várias estações em um raio."""