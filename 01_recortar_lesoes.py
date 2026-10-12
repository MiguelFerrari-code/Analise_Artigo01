import os
import cv2

# Variável para definir qual a partição a processar ('train', 'test' ou 'valid')
TIPO = 'valid'

PASTA_LABELS = f'BRACOL-ORIGINAL-DETECT/{TIPO}/labels/'
PASTA_IMAGES = f'BRACOL-ORIGINAL-DETECT/{TIPO}/images/'
PASTA_DESTINO = 'dataset_recortado/'

# Classes mapeadas exatamente como no data.yaml
CLASSES = {
    0: 'Cercospora', 
    1: 'Miner', 
    2: 'Phoma', 
    3: 'Rust'
}

# Criação automática das subpastas para cada doença
for nome_classe in CLASSES.values():
    os.makedirs(os.path.join(PASTA_DESTINO, nome_classe), exist_ok=True)

print(f"A iniciar o recorte das lesões na partição '{TIPO}'...")
contador = 0

if not os.path.exists(PASTA_LABELS) or not os.path.exists(PASTA_IMAGES):
    print(f"Erro: Verifique se as pastas '{PASTA_LABELS}' e '{PASTA_IMAGES}' existem.")
else:
    # Varredura dos ficheiros de anotação
    for arquivo in os.listdir(PASTA_LABELS):
        if arquivo.endswith('.txt') and arquivo not in ["classes.txt"]:
            
            # Busca a imagem correspondente (.jpg ou .png) na pasta images
            nome_imagem = arquivo.replace('.txt', '.jpg')
            caminho_imagem = os.path.join(PASTA_IMAGES, nome_imagem)
            
            if not os.path.exists(caminho_imagem):
                nome_imagem = arquivo.replace('.txt', '.png')
                caminho_imagem = os.path.join(PASTA_IMAGES, nome_imagem)
                
            if not os.path.exists(caminho_imagem):
                continue 

            imagem = cv2.imread(caminho_imagem)
            if imagem is None:
                continue
                
            h_img, w_img, _ = imagem.shape

            # Leitura das coordenadas no formato YOLOv8
            with open(os.path.join(PASTA_LABELS, arquivo), 'r') as f:
                linhas = f.readlines()

            for idx, linha in enumerate(linhas):
                dados = linha.strip().split()
                if len(dados) >= 5:
                    id_classe = int(dados[0])
                    x_center, y_center, w_norm, h_norm = map(float, dados[1:5])

                    # Conversão das proporções do YOLO para píxeis absolutos
                    w_box = int(w_norm * w_img)
                    h_box = int(h_norm * h_img)
                    x_min = int((x_center * w_img) - (w_box / 2))
                    y_min = int((y_center * h_img) - (h_box / 2))

                    # Travas de limite para o recorte não vazar da imagem
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(w_img, x_min + w_box), min(h_img, y_min + h_box)

                    recorte = imagem[y_min:y_max, x_min:x_max]

                    # Salvamento do recorte isolado
                    if recorte.size > 0 and id_classe in CLASSES:
                        nome_classe = CLASSES[id_classe]
                        nome_salvamento = f"{nome_imagem.split('.')[0]}_{idx}.jpg"
                        cv2.imwrite(os.path.join(PASTA_DESTINO, nome_classe, nome_salvamento), recorte)
                        contador += 1

print(f"Recorte finalizado! {contador} lesões limpas extraídas com sucesso.")