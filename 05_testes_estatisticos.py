"""
ETAPA 05 - Testes estatísticos

1. Normalidade (Shapiro-Wilk em amostra de até 500 lesões por classe):
   justifica o uso de testes não paramétricos.
2. Kruskal-Wallis por atributo (H0: as 4 classes têm a mesma distribuição),
   com tamanho de efeito eta²_H = (H - k + 1) / (n - k)  (Tomczak; Tomczak, 2014).
   Interpretação usual: < 0,06 pequeno; 0,06-0,14 moderado; >= 0,14 grande.
3. Pós-teste par a par: Mann-Whitney U com correção de Holm para
   comparações múltiplas e delta de Cliff como tamanho de efeito
   (|d| < 0,147 desprezível; < 0,33 pequeno; < 0,474 médio; >= 0,474 grande).

Ressalva: lesões de uma mesma folha não são totalmente independentes; os
p-valores devem ser lidos como descritivos (há em média ~1,6 lesão por folha).

Saídas (resultados/tabelas/):
  normalidade_shapiro.csv
  kruskal_wallis.csv
  posteste_mann_whitney.csv
"""
import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, shapiro

PASTA_TAB = os.path.join("resultados", "tabelas")
os.makedirs(PASTA_TAB, exist_ok=True)
CLASSES = ["Cercospora", "Miner", "Phoma", "Rust"]
ALFA = 0.05

df = pd.read_csv("base_tabular_tcc.csv")
ATRIBUTOS = [c for c in df.columns if c not in ("ID_Lesao", "ID_Folha", "Classe")]
n, k = len(df), len(CLASSES)


def holm(pvalores):
    """Correção de Holm-Bonferroni (retorna p-valores ajustados)."""
    p = np.asarray(pvalores)
    ordem = np.argsort(p)
    m = len(p)
    ajust = np.empty(m)
    acumulado = 0.0
    for pos, idx in enumerate(ordem):
        acumulado = max(acumulado, (m - pos) * p[idx])
        ajust[idx] = min(1.0, acumulado)
    return ajust


def delta_cliff(a, b):
    """Delta de Cliff calculado a partir da estatística U de Mann-Whitney."""
    u = mannwhitneyu(a, b).statistic
    return 2 * u / (len(a) * len(b)) - 1


def magnitude_eta(e):
    return "grande" if e >= 0.14 else "moderado" if e >= 0.06 else "pequeno"


def magnitude_cliff(d):
    d = abs(d)
    return ("grande" if d >= 0.474 else "médio" if d >= 0.33
            else "pequeno" if d >= 0.147 else "desprezível")


# 1. Normalidade
rng = np.random.default_rng(42)
norm = []
for atr in ATRIBUTOS:
    linha = {"Atributo": atr}
    for c in CLASSES:
        v = df.loc[df["Classe"] == c, atr].values
        if len(v) > 500:
            v = rng.choice(v, 500, replace=False)
        linha[c] = shapiro(v).pvalue
    norm.append(linha)
norm = pd.DataFrame(norm)
norm.to_csv(os.path.join(PASTA_TAB, "normalidade_shapiro.csv"), index=False)
n_rejeita = int((norm[CLASSES] < ALFA).values.sum())
print(f"Shapiro-Wilk: normalidade rejeitada em {n_rejeita} de {norm[CLASSES].size} "
      "combinações atributo x classe -> uso de testes não paramétricos.")

# 2. Kruskal-Wallis
kw = []
for atr in ATRIBUTOS:
    H, p = kruskal(*[df.loc[df["Classe"] == c, atr] for c in CLASSES])
    eta = max(0.0, (H - k + 1) / (n - k))
    kw.append({"Atributo": atr, "H": round(H, 1), "p_valor": p,
               "eta2_H": round(eta, 3), "Efeito": magnitude_eta(eta)})
kw = pd.DataFrame(kw).sort_values("H", ascending=False)
kw["p_valor"] = kw["p_valor"].map(lambda p: f"{p:.2e}")
kw.to_csv(os.path.join(PASTA_TAB, "kruskal_wallis.csv"), index=False)
print("\nKruskal-Wallis (ordenado por H):")
print(kw.to_string(index=False))

# 3. Pós-teste par a par
pares = list(itertools.combinations(CLASSES, 2))
linhas = []
for atr in ATRIBUTOS:
    for a, b in pares:
        va = df.loc[df["Classe"] == a, atr]
        vb = df.loc[df["Classe"] == b, atr]
        d = delta_cliff(va, vb)
        linhas.append({"Atributo": atr, "Par": f"{a} x {b}",
                       "p_valor": mannwhitneyu(va, vb).pvalue,
                       "Delta_Cliff": round(d, 3), "Efeito": magnitude_cliff(d)})
post = pd.DataFrame(linhas)
post["p_ajustado_Holm"] = holm(post["p_valor"])
post["Significativo"] = post["p_ajustado_Holm"] < ALFA
post.to_csv(os.path.join(PASTA_TAB, "posteste_mann_whitney.csv"), index=False)

print("\nPós-teste: nº de atributos com efeito médio ou grande (|delta| >= 0,33) e significativo, por par:")
forte = post[post["Significativo"] & post["Efeito"].isin(["médio", "grande"])]
resumo = forte.groupby("Par")["Atributo"].agg(["count", lambda s: ", ".join(s.head(4))])
resumo.columns = ["N_atributos", "Exemplos"]
print(resumo.reindex([f"{a} x {b}" for a, b in pares]).fillna({"N_atributos": 0}).to_string())
