"""
ETAPA 04 - Análise exploratória

1. Matriz de correlação de Spearman entre os 18 atributos (robusta a
   assimetria e outliers) e lista dos pares fortemente correlacionados
   (|rho| >= 0,90), que indicam redundância de informação.
2. Boxplots dos atributos com maior poder de discriminação entre as classes
   (ordenados pela estatística H de Kruskal-Wallis; ver Etapa 05).
3. Tabela de estatísticas descritivas (mediana e intervalo interquartil) por classe.

Saídas (resultados/):
  figuras/fig_heatmap_correlacao.png
  figuras/fig_boxplots_atributos.png
  tabelas/pares_correlacionados.csv
  tabelas/descritivas_por_classe.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal

PASTA_FIG = os.path.join("resultados", "figuras")
PASTA_TAB = os.path.join("resultados", "tabelas")
os.makedirs(PASTA_FIG, exist_ok=True)
os.makedirs(PASTA_TAB, exist_ok=True)

CLASSES = ["Cercospora", "Miner", "Phoma", "Rust"]
CORES = {"Cercospora": "#e74c3c", "Miner": "#3498db", "Phoma": "#2ecc71", "Rust": "#f1c40f"}
LIMIAR_REDUNDANCIA = 0.90
N_BOXPLOTS = 6

df = pd.read_csv("base_tabular_tcc.csv")
ATRIBUTOS = [c for c in df.columns if c not in ("ID_Lesao", "ID_Folha", "Classe")]

# ---------------- Correlação ----------------
corr = df[ATRIBUTOS].corr(method="spearman")
plt.figure(figsize=(13, 11))
mascara_sup = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mascara_sup, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, linewidths=0.4, annot_kws={"size": 7},
            cbar_kws={"label": "rho de Spearman"})
plt.tight_layout()
plt.savefig(os.path.join(PASTA_FIG, "fig_heatmap_correlacao.png"), dpi=300)
plt.close()

pares = []
for i, a in enumerate(ATRIBUTOS):
    for b in ATRIBUTOS[i + 1:]:
        if abs(corr.loc[a, b]) >= LIMIAR_REDUNDANCIA:
            pares.append({"Atributo_1": a, "Atributo_2": b, "rho": round(corr.loc[a, b], 3)})
pares = pd.DataFrame(pares, columns=["Atributo_1", "Atributo_2", "rho"])
if not pares.empty:
    pares = pares.sort_values("rho", key=abs, ascending=False)
pares.to_csv(os.path.join(PASTA_TAB, "pares_correlacionados.csv"), index=False)
print(f"Pares com |rho| >= {LIMIAR_REDUNDANCIA}:")
print(pares.to_string(index=False) if not pares.empty else "  nenhum")

# ---------------- Boxplots ----------------
H = {a: kruskal(*[df.loc[df["Classe"] == c, a] for c in CLASSES]).statistic for a in ATRIBUTOS}
destaque = sorted(H, key=H.get, reverse=True)[:N_BOXPLOTS]
print("\nAtributos nos boxplots (maior H de Kruskal-Wallis):", ", ".join(destaque))

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, atr in zip(axes.flatten(), destaque):
    sns.boxplot(data=df, x="Classe", y=atr, hue="Classe", order=CLASSES, hue_order=CLASSES,
                palette=CORES, legend=False, ax=ax, fliersize=2)
    ax.set_title(atr)
    ax.set_xlabel("")
    ax.set_ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(PASTA_FIG, "fig_boxplots_atributos.png"), dpi=300)
plt.close()

# ---------------- Descritivas ----------------
linhas = []
for atr in ATRIBUTOS:
    linha = {"Atributo": atr}
    for c in CLASSES:
        v = df.loc[df["Classe"] == c, atr]
        q1, med, q3 = v.quantile([0.25, 0.5, 0.75])
        linha[c] = f"{med:.2f} ({q1:.2f}-{q3:.2f})"
    linhas.append(linha)
pd.DataFrame(linhas).to_csv(os.path.join(PASTA_TAB, "descritivas_por_classe.csv"), index=False)
print(f"\nTabela de medianas (IIQ) salva em {PASTA_TAB}/descritivas_por_classe.csv")
