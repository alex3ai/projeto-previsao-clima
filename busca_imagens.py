"""Buscamos as Imagens de Satélite Sincronizadas (Sentinel Hub)
Esta é a parte mais desafiadora. Vamos criar uma função para buscar 
a imagem para um local e data específicos."""

import os
import pandas as pd
from sentinelhub import SHConfig, SentinelHubRequest, BBox, CRS, MimeType, DataCollection
import numpy as np
from PIL import Image

# CONFIGURAÇÃO INICIAL (FAREMOS ISSO APENAS UMA VEZ) !!!!!!!!!!!!!
#IDS_CLIENTS em https://insights.planet.com/ <<<< crie um novo cliente AuthID acessando as configurações
config = SHConfig()
# Insira suas credenciais aqui
config.sh_client_id = 'f444bb9a-b90f-4d16-8c63-0836321dbbab'
config.sh_client_secret = 'OTZChoKICYy1Dpch9VUFJLnefdWZHOK8'
config.save()
# ---------------------------------------------------------

def get_satellite_image(lat, lon, date_str, output_folder='images'):
    """Busca e salva uma imagem de satélite para uma dada lat, lon e data."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. Define a "caixa" (bounding box) ao redor do ponto de interesse
    bbox_size = 0.05 # Define o tamanho da área da imagem
    bbox = BBox(bbox=[lon - bbox_size, lat - bbox_size, lon + bbox_size, lat + bbox_size], crs=CRS.WGS84)

    # 2. Define o intervalo de tempo (o dia inteiro da data fornecida)
    time_interval = (f'{date_str}T00:00:00Z', f'{date_str}T23:59:59Z')

    # 3. Define a requisição para a API (pedindo uma imagem de cor real)
    request = SentinelHubRequest(
        evalscript="""
            //VERSION=3
            function setup() {
                return {
                    input: ["B04", "B03", "B02"],
                    output: { bands: 3 }
                };
            }
            function evaluatePixel(sample) {
                return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
            }
        """,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=time_interval,
            ),
        ],
        responses=[SentinelHubRequest.output_response('default', MimeType.PNG)],
        bbox=bbox,
        size=(256, 256), # Tamanho da imagem em pixels
        config=config
    )

    try:
        # 4. Executa a requisição e obtém os dados da imagem
        image_data = request.get_data()[0]
        image = Image.fromarray(image_data)
        
        # 5. Salva a imagem
        image_path = os.path.join(output_folder, f'img_{date_str}_{lat}_{lon}.png')
        image.save(image_path)
        print(f"Imagem salva em: {image_path}")
        return image_path
    except Exception as e:
        print(f"Não foi possível obter a imagem para {date_str}. Erro: {e}")
        return None

# Exemplo de uso
# (Vamos integrar isso no próximo passo)
# lat, lon = -23.55, -46.63
# date = '2022-10-15'
# get_satellite_image(lat, lon, date)