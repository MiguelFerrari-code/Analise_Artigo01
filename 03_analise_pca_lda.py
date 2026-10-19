"""
ETAPA 03 - Análise multivariada: PCA e LDA

1. Padroniza os 18 atributos (z-score).
2. PCA (não supervisionada): variância explicada, nº de componentes para 90%
   e cargas (loadings) dos dois primeiros componentes.
3. LDA (supervisionada): projeção nos discriminantes e razão da variância
   entre classes captada por cada discriminante.
4. Separabilidade linear: o LDA é usado como classificador linear de
   referência em validação cruzada de 5 partições AGRUPADA POR FOLHA
   (GroupKFold): lesões da mesma folha nunca ficam ao mesmo tempo em treino
   e teste. Métricas: acurácia, acurácia balanceada e F1-macro.
5. Comparação: base sem máscara x com máscara, e só cor x só textura x
   cor + textura (mostra a contribuição de cada etapa do método).

Observação: o scaler é ajustado dentro de cada partição de treino
(Pipeline), evitando vazamento de informação do conjunto de teste.

Saídas (resultados/):
  figuras/fig_pca_lda.png
  figuras/fig_variancia_pca.png
  figuras/fig_matriz_confusao_lda.png
  tabelas/pca_variancia.csv, pca_cargas.csv, lda_variancia.csv
  tabelas/separabilidade_lda.csv, matriz_confusao_lda.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix

PASTA_FIG = os.path.join("resultados", "figuras")
PASTA_TAB = os.path.join("resultados", "tabelas")
os.makedirs(PASTA_FIG, exist_ok=True)
os.makedirs(PASTA_TAB, exist_ok=True)

CLASSES = ["Cercospora", "Miner", "Phoma", "Rust"]
CORES = {"Cercospora": "#e74c3c", "Miner": "#3498db", "Phoma": "#2ecc71", "Rust": "#f1c40f"}
N_PARTICOES = 5

df = pd.read_csv("base_tabular_tcc.csv")
ATRIBUTOS = [c for c in df.columns if c not in ("ID_Lesao", "ID_Folha", "Classe")]
ATR_COR = [c for c in ATRIBUTOS if c.startswith(("HSV_", "LAB_"))]
ATR_TEXTURA = [c for c in ATRIBUTOS if c.startswith("GLCM_")]
y = df["Classe"].values
grupos = df["ID_Folha"].values
print(f"{len(df)} lesões, {len(ATRIBUTOS)} atributos, {df['ID_Folha'].nunique()} folhas")

X = StandardScaler().fit_transform(df[ATRIBUTOS])

# ---------------- PCA ----------------
pca = PCA().fit(X)
var = pca.explained_variance_ratio_
acum = np.cumsum(var)
n90 = int(np.searchsorted(acum, 0.90) + 1)
pd.DataFrame({"Componente": [f"PC{i+1}" for i in range(len(var))],
              "Variancia_%": (var * 100).round(2),
              "Acumulada_%": (acum * 100).round(2)}).to_csv(
    os.path.join(PASTA_TAB, "pca_variancia.csv"), index=False)
cargas = pd.DataFrame(pca.components_[:2].T, index=ATRIBUTOS, columns=["PC1", "PC2"]).round(3)
cargas.to_csv(os.path.join(PASTA_TAB, "pca_cargas.csv"))
print(f"\nPCA: PC1 = {var[0]*100:.1f}%, PC2 = {var[1]*100:.1f}%, "
      f"PC1+PC2 = {acum[1]*100:.1f}%; {n90} componentes explicam >= 90%")
print("Maiores cargas em PC1:", ", ".join(cargas["PC1"].abs().sort_values(ascending=False).index[:4]))
print("Maiores cargas em PC2:", ", ".join(cargas["PC2"].abs().sort_values(ascending=False).index[:4]))

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(range(1, len(var) + 1), var * 100, color="#7f8c8d", label="Individual")
ax.plot(range(1, len(var) + 1), acum * 100, "o-", color="#2c3e50", label="Acumulada")
ax.axhline(90, ls="--", color="#c0392b", lw=1)
ax.set_xlabel("Componente principal")
ax.set_ylabel("Variância explicada (%)")
ax.set_xticks(range(1, len(var) + 1))
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PASTA_FIG, "fig_variancia_pca.png"), dpi=300)
plt.close()

# ---------------- LDA (projeção) ----------------
lda = LDA().fit(X, y)
Z_lda = lda.transform(X)
razao = lda.explained_variance_ratio_
pd.DataFrame({"Discriminante": [f"LD{i+1}" for i in range(len(razao))],
              "Razao_variancia_entre_classes_%": (razao * 100).round(2)}).to_csv(
    os.path.join(PASTA_TAB, "lda_variancia.csv"), index=False)
print(f"LDA: LD1 = {razao[0]*100:.1f}%, LD2 = {razao[1]*100:.1f}%, LD3 = {razao[2]*100:.1f}% "
      "(razão da variância entre classes)")

Z_pca = pca.transform(X)[:, :2]
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, Z, titulo, xl, yl in (
        (axes[0], Z_pca, "(a) PCA", f"PC1 ({var[0]*100:.1f}%)", f"PC2 ({var[1]*100:.1f}%)"),
        (axes[1], Z_lda[:, :2], "(b) LDA", f"LD1 ({razao[0]*100:.1f}%)", f"LD2 ({razao[1]*100:.1f}%)")):
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=y, hue_order=CLASSES, palette=CORES,
                    ax=ax, alpha=0.6, s=18, edgecolor=None)
    ax.set_title(titulo)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.legend(title="Classe")
plt.tight_layout()
plt.savefig(os.path.join(PASTA_FIG, "fig_pca_lda.png"), dpi=300)
plt.close()

# ---------------- Separabilidade linear (LDA como classificador) ----------------
cv = GroupKFold(n_splits=N_PARTICOES)
metricas = {"accuracy": "Acuracia", "balanced_accuracy": "Acuracia_balanceada", "f1_macro": "F1_macro"}


def avaliar(base, colunas, nome):
    modelo = make_pipeline(StandardScaler(), LDA())
    res = cross_validate(modelo, base[colunas], base["Classe"], groups=base["ID_Folha"],
                         cv=cv, scoring=list(metricas))
    linha = {"Configuracao": nome, "N_atributos": len(colunas)}
    for k, rot in metricas.items():
        linha[rot] = f"{res['test_' + k].mean():.3f} ± {res['test_' + k].std():.3f}"
    return linha


df_sem = pd.read_csv("base_tabular_sem_mascara.csv").set_index("ID_Lesao").loc[df["ID_Lesao"]].reset_index()
cols_medias = [c for c in ATR_COR if c.endswith("_media")]
resultados = [
    avaliar(df_sem, cols_medias, "Sem máscara - médias de cor"),
    avaliar(df_sem, ATRIBUTOS, "Sem máscara - todos os atributos"),
    avaliar(df, cols_medias, "Com máscara - médias de cor"),
    avaliar(df, ATR_COR, "Com máscara - cor (média + desvio)"),
    avaliar(df, ATR_TEXTURA, "Com máscara - textura GLCM"),
    avaliar(df, ATRIBUTOS, "Com máscara - cor + textura (proposto)"),
]
tab = pd.DataFrame(resultados)
tab.to_csv(os.path.join(PASTA_TAB, "separabilidade_lda.csv"), index=False)
print(f"\nLDA como classificador linear (GroupKFold, {N_PARTICOES} partições por folha):")
print(tab.to_string(index=False))

# Matriz de confusão da configuração proposta
pred = cross_val_predict(make_pipeline(StandardScaler(), LDA()), df[ATRIBUTOS], y,
                         groups=grupos, cv=cv)
cm = confusion_matrix(y, pred, labels=CLASSES)
pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(os.path.join(PASTA_TAB, "matriz_confusao_lda.csv"))
cm_norm = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASSES,
            yticklabels=CLASSES, vmin=0, vmax=1, ax=ax)
ax.set_xlabel("Classe prevista")
ax.set_ylabel("Classe real")
plt.tight_layout()
plt.savefig(os.path.join(PASTA_FIG, "fig_matriz_confusao_lda.png"), dpi=300)
plt.close()
print("\nRecall por classe (LDA, proposto):",
      ", ".join(f"{c} {cm_norm[i, i]:.2f}" for i, c in enumerate(CLASSES)))
print(f"\nFiguras em {PASTA_FIG} e tabelas em {PASTA_TAB}")
