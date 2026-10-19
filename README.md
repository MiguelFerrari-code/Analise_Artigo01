# Artigo 01 — Extração de atributos de cor e textura de lesões foliares do cafeeiro

Pipeline de pré-processamento, extração de atributos e análise multivariada sobre o
dataset BRACOL (versão de detecção publicada no Roboflow Universe, licença CC BY 4.0).

## Ambiente

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

## Ordem de execução

| Etapa | Script | O que faz | Principais saídas |
|---|---|---|---|
| 01 | `01_recortar_lesoes.py` | Recorta as caixas YOLO das 3 partições, exclui caixas < 16 px e segmenta a lesão (fundo por HSV, tecido sadio por Otsu em a*) | `dataset_recortado/`, `registro_recortes.csv`, `fig_exemplos_segmentacao.png` |
| 02 | `02_extracao_hsv_glcm.py` | 12 atributos de cor (média/desvio HSV e LAB) + 6 de textura GLCM, apenas dentro da máscara | `base_tabular_tcc.csv`, `base_tabular_sem_mascara.csv` |
| 03 | `03_analise_pca_lda.py` | PCA, LDA e separabilidade linear (LDA em GroupKFold por folha) | `fig_pca_lda.png`, `fig_variancia_pca.png`, `fig_matriz_confusao_lda.png`, `separabilidade_lda.csv` |
| 04 | `04_analise_exploratoria.py` | Correlação de Spearman, boxplots e descritivas por classe | `fig_heatmap_correlacao.png`, `fig_boxplots_atributos.png` |
| 05 | `05_testes_estatisticos.py` | Shapiro-Wilk, Kruskal-Wallis (η²_H) e Mann-Whitney com Holm (δ de Cliff) | `kruskal_wallis.csv`, `posteste_mann_whitney.csv` |

```bash
python 01_recortar_lesoes.py
python 02_extracao_hsv_glcm.py
python 03_analise_pca_lda.py
python 04_analise_exploratoria.py
python 05_testes_estatisticos.py
```

Figuras ficam em `resultados/figuras/` e tabelas em `resultados/tabelas/`.

## Observações metodológicas

- As partições `train`, `valid` e `test` do Roboflow são unidas; a divisão para
  avaliação é refeita por **folha** (`ID_Folha`), para que lesões da mesma folha
  não apareçam simultaneamente em treino e teste.
- Escalas do OpenCV: H em 0–179 (metade dos graus); S, V, L*, a*, b* em 0–255,
  com a* = b* = 128 correspondendo ao neutro.
- O matiz H é tratado como variável circular (média e desvio circulares).
