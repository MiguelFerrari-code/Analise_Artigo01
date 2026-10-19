"""
ETAPA 02 - Extração de atributos de cor (HSV, LAB) e textura (GLCM)

Para cada lesão recortada na Etapa 01, calcula os atributos SOMENTE sobre os
pixels da máscara da lesão (tecido sadio e fundo são ignorados).

Atributos (18 no total):
  Cor (12): média e desvio-padrão de H, S, V (HSV) e de L*, a*, b* (LAB);
            para o matiz H usam-se média e desvio circulares.
  Textura (6): Contraste, Dissimilaridade, Homogeneidade, Energia,
               Correlação e Entropia, extraídos da GLCM (Haralick et al., 1973).

Parâmetros da GLCM:
  - imagem em tons de cinza quantizada em NIVEIS_CINZA níveis
    (256 níveis geram matrizes esparsas demais para recortes de ~150 px);
  - distâncias 1 e 2 px; ângulos 0°, 45°, 90° e 135°;
  - matriz simétrica; cada atributo é a média sobre as 8 combinações
    (média angular torna o descritor aproximadamente invariante à rotação);
  - apenas pares de pixels em que AMBOS pertencem à lesão são contados.

Para medir o efeito da segmentação, o script também gera uma base com os
mesmos atributos calculados sobre a caixa inteira (sem máscara).

Saídas:
  base_tabular_tcc.csv                (com máscara - base principal)
  base_tabular_sem_mascara.csv        (caixa inteira - usada na comparação)
"""
import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

PASTA_DATASET = "dataset_recortado"
REGISTRO = os.path.join("resultados", "tabelas", "registro_recortes.csv")

NIVEIS_CINZA = 32
DISTANCIAS = [1, 2]
ANGULOS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
PROPRIEDADES = {
    "contrast": "GLCM_Contraste",
    "dissimilarity": "GLCM_Dissimilaridade",
    "homogeneity": "GLCM_Homogeneidade",
    "energy": "GLCM_Energia",
    "correlation": "GLCM_Correlacao",
}


def atributos_cor(imagem_bgr, mascara):
    """Média e desvio-padrão dos canais HSV e LAB dentro da máscara."""
    hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2LAB)
    m = mascara > 0
    r = {}
    for espaco, img, canais in (("HSV", hsv, "HSV"), ("LAB", lab, "LAB")):
        for i, canal in enumerate(canais):
            valores = img[..., i][m].astype(float)
            if espaco == "HSV" and canal == "H":
                media, desvio = estatistica_circular_matiz(valores)
            else:
                media, desvio = valores.mean(), valores.std()
            r[f"{espaco}_{canal}_media"] = media
            r[f"{espaco}_{canal}_desvio"] = desvio
    return r


def estatistica_circular_matiz(h):
    """Média e desvio circulares do matiz (OpenCV: 0-179 representa 0-358°).
    O matiz é um ângulo: tons vermelhos/marrons ficam perto de 0 e de 179 ao
    mesmo tempo, e a média aritmética comum distorceria esses valores."""
    ang = h * (2 * np.pi / 180.0)
    c, s = np.cos(ang).mean(), np.sin(ang).mean()
    media = (np.arctan2(s, c) % (2 * np.pi)) * 180.0 / (2 * np.pi)
    R = np.clip(np.hypot(c, s), 1e-12, 1.0)
    desvio = np.sqrt(-2 * np.log(R)) * 180.0 / (2 * np.pi)
    return media, desvio


def atributos_textura(imagem_bgr, mascara):
    """Atributos de Haralick a partir da GLCM restrita aos pixels da lesão."""
    cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    # Quantização: níveis 1..NIVEIS_CINZA para a lesão; 0 marca "fora da máscara"
    q = (cinza.astype(np.uint16) * NIVEIS_CINZA // 256 + 1).astype(np.uint8)
    q[mascara == 0] = 0

    glcm = graycomatrix(q, DISTANCIAS, ANGULOS, levels=NIVEIS_CINZA + 1,
                        symmetric=True, normed=False)
    # Descarta linha/coluna 0: pares que envolvem pixels fora da lesão
    glcm = glcm[1:, 1:, :, :].astype(float)
    soma = glcm.sum(axis=(0, 1), keepdims=True)
    soma[soma == 0] = 1
    glcm /= soma

    r = {}
    for prop, nome in PROPRIEDADES.items():
        valores = graycoprops(glcm, prop)
        r[nome] = float(np.nanmean(valores)) if np.isfinite(valores).any() else 0.0
    # Entropia média sobre as 8 matrizes (distância x ângulo)
    entropias = [-(g[g > 0] * np.log2(g[g > 0])).sum()
                 for g in np.moveaxis(glcm.reshape(NIVEIS_CINZA, NIVEIS_CINZA, -1), -1, 0)]
    r["GLCM_Entropia"] = float(np.mean(entropias))
    return r


def main():
    registro = pd.read_csv(REGISTRO)
    registro = registro[~registro["Excluida"]]
    print(f"Extraindo atributos de {len(registro)} lesões...")

    com_mascara, sem_mascara = [], []
    for i, linha in enumerate(registro.itertuples(index=False), 1):
        pasta = os.path.join(PASTA_DATASET, linha.Classe)
        imagem = cv2.imread(os.path.join(pasta, linha.ID_Lesao + ".jpg"))
        mascara = cv2.imread(os.path.join(pasta, linha.ID_Lesao + "_mask.png"), cv2.IMREAD_GRAYSCALE)
        if imagem is None or mascara is None:
            continue
        ident = {"ID_Lesao": linha.ID_Lesao, "ID_Folha": linha.ID_Folha, "Classe": linha.Classe}

        com_mascara.append({**ident, **atributos_cor(imagem, mascara),
                            **atributos_textura(imagem, mascara)})
        caixa = np.full(mascara.shape, 255, np.uint8)
        sem_mascara.append({**ident, **atributos_cor(imagem, caixa),
                            **atributos_textura(imagem, caixa)})
        if i % 300 == 0:
            print(f"  {i} lesões processadas")

    df = pd.DataFrame(com_mascara)
    df.to_csv("base_tabular_tcc.csv", index=False)
    pd.DataFrame(sem_mascara).to_csv("base_tabular_sem_mascara.csv", index=False)

    n_attr = df.shape[1] - 3
    print(f"\nBase principal: {len(df)} lesões x {n_attr} atributos -> base_tabular_tcc.csv")
    print("Base de comparação (sem máscara) -> base_tabular_sem_mascara.csv")
    print(df["Classe"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
