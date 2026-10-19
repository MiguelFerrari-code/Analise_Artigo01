"""
ETAPA 01 - Recorte e segmentação das lesões (BRACOL)

O que este script faz:
  1. Percorre as três partições do Roboflow (train, valid, test) e une tudo
     em uma única base, pois a divisão treino/teste será refeita no Artigo 2
     agrupando por folha (imagem de origem).
  2. Converte cada anotação YOLO (x_c, y_c, w, h normalizados) em pixels e
     recorta a caixa delimitadora (bounding box) da lesão.
  3. Exclui caixas degeneradas (lado menor que TAMANHO_MINIMO pixels).
  4. Segmenta a lesão DENTRO da caixa, removendo o fundo branco (limiar em HSV) e o tecido foliar sadio
     (limiar de Otsu no canal a* do LAB), seguido de morfologia matemática.
  5. Salva o recorte (.jpg), a máscara binária (.png) e um registro (CSV)
     com a origem de cada lesão e os motivos de exclusão.
  6. Gera uma figura com exemplos de recorte e máscara por classe.

Saídas:
  dataset_recortado/<Classe>/<id>.jpg          recorte da caixa
  dataset_recortado/<Classe>/<id>_mask.png     máscara da lesão (255 = lesão)
  resultados/tabelas/registro_recortes.csv
  resultados/figuras/fig_exemplos_segmentacao.png
"""
import os
import shutil
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PASTA_BASE = "BRACOL-ORIGINAL-DETECT"
PARTICOES = ["train", "valid", "test"]
PASTA_DESTINO = "dataset_recortado"
PASTA_FIG = os.path.join("resultados", "figuras")
PASTA_TAB = os.path.join("resultados", "tabelas")

CLASSES = {0: "Cercospora", 1: "Miner", 2: "Phoma", 3: "Rust"}

# Critérios de exclusão
TAMANHO_MINIMO = 16        # px: largura e altura mínimas da caixa
PIXELS_LESAO_MINIMO = 50   # px: abaixo disso a máscara é considerada falha

# Limiares (escala OpenCV: H 0-179; S, V, L, a*, b* 0-255; a* = 128 é neutro)
FUNDO_S_MAX, FUNDO_V_MIN = 45, 100   # fundo branco/cinza, inclusive sombreado (baixa saturação)
A_MIN, A_MAX = 116, 126              # faixa permitida para o limiar de Otsu em a*

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def segmentar_lesao(recorte_bgr):
    """Retorna a máscara (uint8, 0/255) da lesão dentro da caixa e um
    indicador de fallback (True se a segmentação precisou ser relaxada).

    Critério: o tecido sadio do cafeeiro é verde (a* < 128 no LAB do OpenCV),
    enquanto necroses e pústulas tendem ao vermelho/amarelo (a* maior).
    O limiar em a* é escolhido por Otsu em cada recorte e limitado ao
    intervalo [A_MIN, A_MAX] para não dividir lesões que ocupam a caixa toda.
    """
    hsv = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2LAB)
    s, v = hsv[..., 1], hsv[..., 2]
    a = lab[..., 1]

    fundo = (s <= FUNDO_S_MAX) & (v >= FUNDO_V_MIN)
    pixels_folha = a[~fundo]
    if pixels_folha.size > 0:
        limiar, _ = cv2.threshold(pixels_folha.reshape(-1, 1), 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        limiar = A_MIN
    limiar = float(np.clip(limiar, A_MIN, A_MAX))

    lesao = (a > limiar) & ~fundo
    mascara = lesao.astype(np.uint8) * 255
    # Abertura remove ruído isolado; fechamento preenche pequenos buracos
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, KERNEL)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, KERNEL)

    fallback = False
    if (mascara > 0).sum() < PIXELS_LESAO_MINIMO:
        # Lesão muito pequena/pouco contrastada: mantém tudo que não é fundo
        mascara = (~fundo).astype(np.uint8) * 255
        fallback = True
    return mascara, fallback


def main():
    # Limpa execuções anteriores para não misturar recortes antigos
    if os.path.exists(PASTA_DESTINO):
        shutil.rmtree(PASTA_DESTINO)
    for nome in CLASSES.values():
        os.makedirs(os.path.join(PASTA_DESTINO, nome), exist_ok=True)
    os.makedirs(PASTA_FIG, exist_ok=True)
    os.makedirs(PASTA_TAB, exist_ok=True)

    registros = []
    for particao in PARTICOES:
        pasta_labels = os.path.join(PASTA_BASE, particao, "labels")
        pasta_images = os.path.join(PASTA_BASE, particao, "images")
        if not os.path.isdir(pasta_labels):
            print(f"Aviso: partição '{particao}' não encontrada, ignorando.")
            continue
        print(f"Processando partição '{particao}'...")

        for arquivo in sorted(os.listdir(pasta_labels)):
            if not arquivo.endswith(".txt"):
                continue
            base = arquivo[:-4]
            caminho_img = None
            for ext in (".jpg", ".png", ".jpeg"):
                c = os.path.join(pasta_images, base + ext)
                if os.path.exists(c):
                    caminho_img = c
                    break
            if caminho_img is None:
                continue
            imagem = cv2.imread(caminho_img)
            if imagem is None:
                continue
            h_img, w_img = imagem.shape[:2]

            # ID da folha = nome original da imagem no BRACOL (ex.: "1016_jpg")
            id_folha = base.split(".")[0]

            with open(os.path.join(pasta_labels, arquivo)) as f:
                linhas = [l.split() for l in f if l.strip()]

            for idx, dados in enumerate(linhas):
                if len(dados) < 5:
                    continue
                id_classe = int(dados[0])
                if id_classe not in CLASSES:
                    continue
                xc, yc, wn, hn = map(float, dados[1:5])
                w_box, h_box = int(round(wn * w_img)), int(round(hn * h_img))
                x_min = max(0, int(round(xc * w_img - w_box / 2)))
                y_min = max(0, int(round(yc * h_img - h_box / 2)))
                x_max = min(w_img, x_min + w_box)
                y_max = min(h_img, y_min + h_box)

                reg = {
                    "ID_Lesao": f"{id_folha}_{idx}",
                    "ID_Folha": id_folha,
                    "Particao_Roboflow": particao,
                    "Classe": CLASSES[id_classe],
                    "Largura_px": x_max - x_min,
                    "Altura_px": y_max - y_min,
                    "Fallback_Mascara": False,
                    "Excluida": False,
                    "Motivo_Exclusao": "",
                }
                if reg["Largura_px"] < TAMANHO_MINIMO or reg["Altura_px"] < TAMANHO_MINIMO:
                    reg["Excluida"] = True
                    reg["Motivo_Exclusao"] = f"caixa menor que {TAMANHO_MINIMO}px"
                    registros.append(reg)
                    continue

                recorte = imagem[y_min:y_max, x_min:x_max]
                mascara, fallback = segmentar_lesao(recorte)
                reg["Fallback_Mascara"] = fallback
                reg["Fracao_Lesao"] = round(float((mascara > 0).mean()), 4)

                destino = os.path.join(PASTA_DESTINO, CLASSES[id_classe])
                cv2.imwrite(os.path.join(destino, reg["ID_Lesao"] + ".jpg"), recorte)
                cv2.imwrite(os.path.join(destino, reg["ID_Lesao"] + "_mask.png"), mascara)
                registros.append(reg)

    df = pd.DataFrame(registros)
    df.to_csv(os.path.join(PASTA_TAB, "registro_recortes.csv"), index=False)

    validas = df[~df["Excluida"]]
    print("\nResumo")
    print(f"  Imagens (folhas) de origem: {df['ID_Folha'].nunique()}")
    print(f"  Anotações lidas:            {len(df)}")
    print(f"  Excluídas (caixa pequena):  {df['Excluida'].sum()}")
    print(f"  Lesões válidas:             {len(validas)}")
    print(f"  Máscaras com fallback:      {validas['Fallback_Mascara'].sum()}")
    print("  Lesões válidas por classe:")
    print(validas["Classe"].value_counts().sort_index().to_string())

    gerar_figura_exemplos(validas)


def gerar_figura_exemplos(validas, n_por_classe=3, semente=42):
    """Figura para a Metodologia: recorte original e lesão segmentada."""
    rng = np.random.default_rng(semente)
    classes = list(CLASSES.values())
    fig, axes = plt.subplots(len(classes), n_por_classe * 2,
                             figsize=(n_por_classe * 4, len(classes) * 2.2))
    for i, classe in enumerate(classes):
        cand = validas[(validas["Classe"] == classe)
                       & (validas["Largura_px"] >= 80) & (validas["Altura_px"] >= 80)]
        escolhidas = cand.sample(n=min(n_por_classe, len(cand)),
                                 random_state=int(rng.integers(1e6)))
        for j, (_, r) in enumerate(escolhidas.iterrows()):
            pasta = os.path.join(PASTA_DESTINO, classe)
            rec = cv2.cvtColor(cv2.imread(os.path.join(pasta, r["ID_Lesao"] + ".jpg")), cv2.COLOR_BGR2RGB)
            msk = cv2.imread(os.path.join(pasta, r["ID_Lesao"] + "_mask.png"), 0)
            seg = rec.copy()
            seg[msk == 0] = 255
            axes[i, 2 * j].imshow(rec)
            axes[i, 2 * j + 1].imshow(seg)
            if i == 0:
                axes[i, 2 * j].set_title("Recorte", fontsize=9)
                axes[i, 2 * j + 1].set_title("Lesão segmentada", fontsize=9)
        axes[i, 0].set_ylabel(classe, fontsize=10)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    caminho = os.path.join(PASTA_FIG, "fig_exemplos_segmentacao.png")
    plt.savefig(caminho, dpi=300)
    plt.close()
    print(f"\nFigura salva em: {caminho}")


if __name__ == "__main__":
    main()
