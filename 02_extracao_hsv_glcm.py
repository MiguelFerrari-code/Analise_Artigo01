import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

# Caminho para a pasta onde os recortes foram salvos na etapa anterior
PASTA_DATASET = 'dataset_recortado/'

# Lista que vai guardar as linhas da nossa planilha (DataFrame)
dados_tcc = []

print("A iniciar a extração rigorosa de matrizes HSV, LAB e GLCM...")

# Percorre as pastas de cada doença
for nome_doenca in os.listdir(PASTA_DATASET):
    caminho_doenca = os.path.join(PASTA_DATASET, nome_doenca)
    
    if not os.path.isdir(caminho_doenca):
        continue
        
    print(f"A processar lesões da classe: {nome_doenca}...")
    
    for nome_imagem in os.listdir(caminho_doenca):
        caminho_img = os.path.join(caminho_doenca, nome_imagem)
        imagem_rgb = cv2.imread(caminho_img)
        
        if imagem_rgb is None:
            continue
            
        # EXTRAÇÃO 1: Espaço de Cor HSV
        imagem_hsv = cv2.cvtColor(imagem_rgb, cv2.COLOR_BGR2HSV)
        h_media = np.mean(imagem_hsv[:, :, 0])
        s_media = np.mean(imagem_hsv[:, :, 1])
        v_media = np.mean(imagem_hsv[:, :, 2])

        # EXTRAÇÃO 2: Espaço de Cor LAB (Conforme Quadro 2 do Pré-TCC)
        imagem_lab = cv2.cvtColor(imagem_rgb, cv2.COLOR_BGR2LAB)
        l_media = np.mean(imagem_lab[:, :, 0])
        a_media = np.mean(imagem_lab[:, :, 1])
        b_media = np.mean(imagem_lab[:, :, 2])
        
        # EXTRAÇÃO 3: Textura GLCM
        # A GLCM exige conversão para tons de cinza
        imagem_cinza = cv2.cvtColor(imagem_rgb, cv2.COLOR_BGR2GRAY)
        
        # Calculando a matriz GLCM (distância=1 pixel, ângulo=0 radianos)
        glcm = graycomatrix(imagem_cinza, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        
        # Extraindo as propriedades (Features) exigidas no método
        contraste = graycoprops(glcm, 'contrast')[0, 0]
        homogeneidade = graycoprops(glcm, 'homogeneity')[0, 0]
        energia = graycoprops(glcm, 'energy')[0, 0]
        correlacao = graycoprops(glcm, 'correlation')[0, 0]
        
        # Adiciona a linha (registo) na nossa lista de dados
        dados_tcc.append({
            'Nome_Arquivo': nome_imagem,
            'Classe': nome_doenca,
            'HSV_H': h_media,
            'HSV_S': s_media,
            'HSV_V': v_media,
            'LAB_L': l_media,
            'LAB_A': a_media,
            'LAB_B': b_media,
            'GLCM_Contraste': contraste,
            'GLCM_Homogeneidade': homogeneidade,
            'GLCM_Energia': energia,
            'GLCM_Correlacao': correlacao
        })

# Cria um DataFrame do Pandas com os dados extraídos
df = pd.DataFrame(dados_tcc)

# Salva em um ficheiro CSV (Base Tabular Sólida - Saída do Artigo 1)
NOME_CSV = 'base_tabular_tcc.csv'
df.to_csv(NOME_CSV, index=False)

print(f"\nExtração concluída com sucesso!")
print(f"Total de lesões mapeadas matematicamente: {len(df)}")
print(f"Ficheiro gerado e pronto para o PCA: {NOME_CSV}")