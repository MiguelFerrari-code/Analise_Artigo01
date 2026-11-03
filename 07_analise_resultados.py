"""
ETAPA 07 (Artigo 02) - Comparação estatística, custo computacional e interpretabilidade

Lê as saídas da Etapa 06 e produz:
  1. Tabela-resumo (média, desvio e IC 95% corrigido) por modelo.
  2. Comparação par a par com o t-teste corrigido para validação cruzada
     repetida (Nadeau; Bengio, 2003; Bouckaert; Frank, 2004), com correção de
     Holm para múltiplas comparações.
  3. Sensibilidade por classe e matrizes de confusão (LDA x melhor modelo).
  4. Modelo final: hiperparâmetros escolhidos por validação agrupada em toda a
     base; interpretabilidade por valores SHAP (TreeSHAP nativo da biblioteca).
  5. Redução de atributos: o melhor modelo com 13 atributos (um de cada par
     redundante do Artigo 01 removido), no mesmo protocolo aninhado.
  6. Custo computacional: tempo de extração dos 18 atributos por lesão, tempo
     de inferência por lesão e tamanho do modelo salvo.

Saídas: resultados_artigo02/tabelas/*.csv e resultados_artigo02/figuras/*.png
"""
import os
import io
import time
import pickle
import warnings
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
PASTA_TAB = os.path.join("resultados_artigo02", "tabelas")
PASTA_FIG = os.path.join("resultados_artigo02", "figuras")
os.makedirs(PASTA_FIG, exist_ok=True)
SEMENTE = 42

res = pd.read_csv(os.path.join(PASTA_TAB, "metricas_por_particao.csv"))
oof = pd.read_csv(os.path.join(PASTA_TAB, "predicoes_oof.csv"))
df = pd.read_csv("base_tabular_tcc.csv")
ATRIBUTOS = [c for c in df.columns if c not in ("ID_Lesao", "ID_Folha", "Classe")]
codificador = LabelEncoder().fit(df["Classe"])
CLASSES = list(codificador.classes_)
X = df[ATRIBUTOS].to_numpy(dtype=float)
y = codificador.transform(df["Classe"])
grupos = df["ID_Folha"].to_numpy()
NOMES_PT = {"Cercospora": "Cercosporiose", "Miner": "Bicho-mineiro", "Phoma": "Phoma", "Rust": "Ferrugem"}
PRINCIPAIS = ["LDA", "Random Forest", "SVM (RBF)", "XGBoost", "LightGBM"]

# ---------------------------------------------------------------- 1. resumo
n_folds = res.groupby("Modelo").size().iloc[0]
razao = (res["n_teste"] / res["n_treino"]).mean()


def ic_corrigido(v):
    """IC 95% com a variância corrigida de Nadeau e Bengio (2003)."""
    m, s2 = np.mean(v), np.var(v, ddof=1)
    ep = np.sqrt((1 / len(v) + razao) * s2)
    t = stats.t.ppf(0.975, len(v) - 1)
    return m - t * ep, m + t * ep


linhas = []
for nome, g in res.groupby("Modelo"):
    lo, hi = ic_corrigido(g["Acuracia_balanceada"])
    linhas.append({"Modelo": nome,
                   "Acuracia_balanceada": g["Acuracia_balanceada"].mean(),
                   "AB_desvio": g["Acuracia_balanceada"].std(), "AB_IC95_inf": lo, "AB_IC95_sup": hi,
                   "F1_macro": g["F1_macro"].mean(), "F1_desvio": g["F1_macro"].std(),
                   "Acuracia": g["Acuracia"].mean(), "Kappa": g["Kappa"].mean(),
                   **{f"Sens_{c}": g[f"Sensibilidade_{c}"].mean() for c in CLASSES},
                   "Tempo_treino_busca_s": g["Tempo_treino_s"].mean()})
resumo = pd.DataFrame(linhas).sort_values("Acuracia_balanceada", ascending=False)
resumo.round(4).to_csv(os.path.join(PASTA_TAB, "resumo_modelos.csv"), index=False)
print("Resumo:\n", resumo[["Modelo", "Acuracia_balanceada", "AB_desvio", "AB_IC95_inf", "AB_IC95_sup",
                           "F1_macro", "Acuracia", "Kappa"]].round(3).to_string(index=False))
melhor = resumo[resumo["Modelo"].isin(PRINCIPAIS)].iloc[0]["Modelo"]
print("Melhor modelo:", melhor)

# ------------------------------------------- 2. t-teste corrigido + Holm


def t_corrigido(a, b):
    d = np.asarray(a) - np.asarray(b)
    s2 = np.var(d, ddof=1)
    t = d.mean() / np.sqrt((1 / len(d) + razao) * s2) if s2 > 0 else np.inf
    p = 2 * stats.t.sf(abs(t), len(d) - 1)
    return d.mean(), t, p


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); aj = np.empty(m); acc = 0
    for pos, i in enumerate(o):
        acc = max(acc, (m - pos) * p[i]); aj[i] = min(1, acc)
    return aj


piv = res.pivot_table(index=["Repeticao", "Particao"], columns="Modelo", values="Acuracia_balanceada")
comp = []
for a, b in ([(m, "LDA") for m in piv.columns if m != "LDA"] +
             [(melhor, m) for m in piv.columns if m not in (melhor, "LDA")]):
    d, t, p = t_corrigido(piv[a], piv[b])
    comp.append({"Comparacao": f"{a} x {b}", "Diferenca_AB": d, "t": t, "p": p})
comp = pd.DataFrame(comp)
comp["p_Holm"] = holm(comp["p"])
comp["Significativo"] = comp["p_Holm"] < 0.05
comp.round(4).to_csv(os.path.join(PASTA_TAB, "comparacoes_t_corrigido.csv"), index=False)
print("\nComparações (t corrigido, Holm):\n", comp.round(4).to_string(index=False))

# ------------------------------------------------- 3. figuras de desempenho
CORES = {"LDA": "#7f8c8d", "Random Forest": "#27ae60", "SVM (RBF)": "#8e44ad",
         "XGBoost": "#2980b9", "LightGBM": "#e67e22",
         "XGBoost sem peso": "#a9cce3", "LightGBM sem peso": "#f5cba7"}
ordem = resumo["Modelo"].tolist()[::-1]
fig, ax = plt.subplots(figsize=(9, 4.8))
for i, m in enumerate(ordem):
    r = resumo.set_index("Modelo").loc[m]
    v = res[res.Modelo == m]["Acuracia_balanceada"]
    ax.scatter(v, np.full(len(v), i) + np.random.default_rng(i).uniform(-.12, .12, len(v)),
               s=10, color=CORES[m], alpha=.35)
    ax.errorbar(r.Acuracia_balanceada, i, xerr=[[r.Acuracia_balanceada - r.AB_IC95_inf],
                [r.AB_IC95_sup - r.Acuracia_balanceada]], fmt="o", color=CORES[m], ms=8, capsize=4, lw=2)
    ax.text(r.AB_IC95_sup + 0.004, i, f"{r.Acuracia_balanceada:.3f}", va="center", fontsize=9)
ax.set_yticks(range(len(ordem))); ax.set_yticklabels(ordem)
ax.set_xlabel("Acurácia balanceada (pontos = 15 partições; barra = IC 95% corrigido)")
ax.grid(axis="x", alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, "fig_desempenho_modelos.png"), dpi=300); plt.close()

sens = resumo[resumo.Modelo.isin(PRINCIPAIS)].set_index("Modelo")[[f"Sens_{c}" for c in CLASSES]]
sens.columns = [NOMES_PT[c] for c in CLASSES]
fig, ax = plt.subplots(figsize=(10, 4.8))
w = 0.16
for i, m in enumerate(PRINCIPAIS):
    ax.bar(np.arange(4) + (i - 2) * w, sens.loc[m], w, label=m, color=CORES[m])
ax.set_xticks(range(4)); ax.set_xticklabels(sens.columns); ax.set_ylim(0.4, 1); ax.set_ylabel("Sensibilidade média")
ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(.5, 1.12)); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, "fig_sensibilidade_classes.png"), dpi=300); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, m in zip(axes, ["LDA", melhor]):
    o = oof[(oof.Modelo == m) & (oof.Repeticao == 0)]
    cm = confusion_matrix(o.Real, o.Previsto, labels=CLASSES)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(os.path.join(PASTA_TAB, f"matriz_confusao_{m.replace(' ', '_')}.csv"))
    sns.heatmap(cm / cm.sum(1, keepdims=True), annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax,
                xticklabels=[NOMES_PT[c] for c in CLASSES], yticklabels=[NOMES_PT[c] for c in CLASSES], cbar=False)
    ax.set_title(f"({'a' if m == 'LDA' else 'b'}) {m}"); ax.set_xlabel("Classe prevista"); ax.set_ylabel("Classe real")
plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, "fig_matrizes_confusao.png"), dpi=300); plt.close()

# ----------------------------------------------- 4. modelo final + SHAP


def estimador(nome):
    if nome == "XGBoost":
        return (XGBClassifier(objective="multi:softprob", tree_method="hist", random_state=SEMENTE, n_jobs=1, verbosity=0),
                {"n_estimators": randint(100, 601), "max_depth": randint(2, 9), "learning_rate": loguniform(0.01, 0.3),
                 "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.6, 0.4),
                 "min_child_weight": randint(1, 11), "reg_lambda": loguniform(0.1, 10)}, True)
    if nome == "LightGBM":
        return (LGBMClassifier(class_weight="balanced", subsample_freq=1, random_state=SEMENTE, n_jobs=1, verbose=-1),
                {"n_estimators": randint(100, 601), "num_leaves": randint(8, 65), "learning_rate": loguniform(0.01, 0.3),
                 "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.6, 0.4),
                 "min_child_samples": randint(5, 51), "reg_lambda": loguniform(0.1, 10)}, False)
    if nome == "Random Forest":
        return (RandomForestClassifier(class_weight="balanced", random_state=SEMENTE, n_jobs=1),
                {"n_estimators": randint(200, 601), "max_depth": [None, 8, 12, 20],
                 "min_samples_leaf": randint(1, 6), "max_features": ["sqrt", 0.33, 0.5]}, False)
    if nome == "SVM (RBF)":
        return (make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced")),
                {"svc__C": loguniform(0.1, 100), "svc__gamma": loguniform(1e-3, 1)}, False)
    return make_pipeline(StandardScaler(), LDA()), None, False


def ajustar_final(nome, Xm, ym):
    est, espaco, peso = estimador(nome)
    kw = {"sample_weight": compute_sample_weight("balanced", ym)} if peso else {}
    t0 = time.perf_counter()
    if espaco is None:
        mod = clone(est).fit(Xm, ym)
    else:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
        b = RandomizedSearchCV(clone(est), espaco, n_iter=40, scoring="balanced_accuracy", cv=cv,
                               random_state=SEMENTE, n_jobs=-1).fit(Xm, ym, groups=grupos, **kw)
        mod = b.best_estimator_
    return mod, time.perf_counter() - t0


finais = {}
for m in PRINCIPAIS:
    finais[m] = ajustar_final(m, X, y)
    print(f"Modelo final {m} ajustado ({finais[m][1]:.1f}s)")

booster_nome = melhor if melhor in ("XGBoost", "LightGBM") else \
    resumo[resumo.Modelo.isin(["XGBoost", "LightGBM"])].iloc[0]["Modelo"]
mod = finais[booster_nome][0]
if booster_nome == "XGBoost":
    import xgboost as xgb
    contrib = mod.get_booster().predict(xgb.DMatrix(X, feature_names=ATRIBUTOS), pred_contribs=True)
    shap_vals = contrib[:, :, :-1]  # (n, classes, atributos); última coluna = viés
else:
    contrib = mod.predict(X, pred_contrib=True)  # (n, classes*(atributos+1))
    shap_vals = contrib.reshape(len(X), len(CLASSES), len(ATRIBUTOS) + 1)[:, :, :-1]
imp = pd.DataFrame(np.abs(shap_vals).mean(axis=0).T, index=ATRIBUTOS, columns=[NOMES_PT[c] for c in CLASSES])
imp["Total"] = imp.sum(axis=1)
imp = imp.sort_values("Total")
imp.round(4).to_csv(os.path.join(PASTA_TAB, f"importancia_shap_{booster_nome}.csv"))
fig, ax = plt.subplots(figsize=(9, 6.5))
esq = np.zeros(len(imp))
for c, cor in zip([NOMES_PT[c] for c in CLASSES], ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f"]):
    ax.barh(imp.index, imp[c], left=esq, color=cor, label=c); esq += imp[c].to_numpy()
ax.set_xlabel("Média de |valor SHAP| (contribuição para a decisão, em log-odds)")
ax.legend(title="Classe", loc="lower right"); ax.grid(axis="x", alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, "fig_importancia_shap.png"), dpi=300); plt.close()
print("\nImportância SHAP (top 8):\n", imp.sort_values("Total", ascending=False).head(8).round(3).to_string())

# ----------------------------------------------- 5. redução de atributos
REMOVIDOS = ["HSV_V_media", "HSV_V_desvio", "GLCM_Dissimilaridade", "GLCM_Contraste", "GLCM_Energia"]
REDUZIDOS = [a for a in ATRIBUTOS if a not in REMOVIDOS]
Xr = df[REDUZIDOS].to_numpy(dtype=float)
ab_red = []
for rep in range(3):
    ext = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMENTE + rep)
    for k, (tr, te) in enumerate(ext.split(Xr, y, grupos)):
        est, espaco, peso = estimador(booster_nome)
        kw = {"sample_weight": compute_sample_weight("balanced", y[tr])} if peso else {}
        inn = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEMENTE + 100 * rep + k)
        b = RandomizedSearchCV(clone(est), espaco, n_iter=20, scoring="balanced_accuracy", cv=inn,
                               random_state=SEMENTE + k, n_jobs=-1).fit(Xr[tr], y[tr], groups=grupos[tr], **kw)
        ab_red.append(balanced_accuracy_score(y[te], b.predict(Xr[te])))
ab_cheio = piv[booster_nome].to_numpy()
d, t, p = t_corrigido(ab_cheio, ab_red)
reducao = pd.DataFrame([{"Modelo": booster_nome, "AB_18_atributos": ab_cheio.mean(), "AB_13_atributos": np.mean(ab_red),
                         "Diferenca": d, "p_t_corrigido": p, "Removidos": ", ".join(REMOVIDOS)}])
reducao.round(4).to_csv(os.path.join(PASTA_TAB, "reducao_atributos.csv"), index=False)
print("\nRedução de atributos:\n", reducao.round(4).to_string(index=False))

# ----------------------------------------------- 6. custo computacional
spec = importlib.util.spec_from_file_location("extracao", "02_extracao_hsv_glcm.py")
ext_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext_mod)
import cv2
amostra = pd.read_csv(os.path.join("resultados", "tabelas", "registro_recortes.csv"))
amostra = amostra[~amostra.Excluida].sample(200, random_state=SEMENTE)
spec1 = importlib.util.spec_from_file_location("recorte", "01_recortar_lesoes.py")
rec_mod = importlib.util.module_from_spec(spec1); spec1.loader.exec_module(rec_mod)
tempos_seg, tempos_ext = [], []
for r in amostra.itertuples():
    im = cv2.imread(os.path.join("dataset_recortado", r.Classe, r.ID_Lesao + ".jpg"))
    t0 = time.perf_counter(); msk, _ = rec_mod.segmentar_lesao(im); t1 = time.perf_counter()
    ext_mod.atributos_cor(im, msk); ext_mod.atributos_textura(im, msk); t2 = time.perf_counter()
    tempos_seg.append((t1 - t0) * 1000); tempos_ext.append((t2 - t1) * 1000)
custo = []
Xone = X[:1]
for m in PRINCIPAIS:
    modelo, t_fit = finais[m]
    buf = io.BytesIO(); pickle.dump(modelo, buf)
    for _ in range(20): modelo.predict(Xone)  # aquecimento
    t0 = time.perf_counter()
    for _ in range(500): modelo.predict(Xone)
    lat = (time.perf_counter() - t0) / 500 * 1000
    t0 = time.perf_counter(); modelo.predict(X); lote = (time.perf_counter() - t0) / len(X) * 1000
    custo.append({"Modelo": m, "Tamanho_KB": len(buf.getvalue()) / 1024,
                  "Latencia_1_lesao_ms": lat, "Inferencia_em_lote_ms_por_lesao": lote,
                  "Tempo_ajuste_final_com_busca_s": t_fit})
custo = pd.DataFrame(custo)
custo.round(4).to_csv(os.path.join(PASTA_TAB, "custo_computacional.csv"), index=False)
extracao = pd.DataFrame([{"Etapa": "Segmentação", "Mediana_ms": np.median(tempos_seg), "P95_ms": np.percentile(tempos_seg, 95)},
                         {"Etapa": "Extração dos 18 atributos", "Mediana_ms": np.median(tempos_ext), "P95_ms": np.percentile(tempos_ext, 95)}])
extracao.round(3).to_csv(os.path.join(PASTA_TAB, "custo_extracao.csv"), index=False)
print("\nCusto:\n", custo.round(3).to_string(index=False)); print(extracao.round(2).to_string(index=False))

# deslocamento dos rótulos que se sobrepõem (LDA e XGBoost têm latência parecida)
DESLOC_ROTULO = {"LDA": (25, -22, "left"), "XGBoost": (-25, 22, "right")}
fig, ax = plt.subplots(figsize=(8, 5))
for r in custo.itertuples():
    ab = resumo.set_index("Modelo").loc[r.Modelo, "Acuracia_balanceada"]
    ax.scatter(r.Latencia_1_lesao_ms, ab, s=40 + r.Tamanho_KB / 5, color=CORES[r.Modelo], alpha=.8, edgecolor="k")
    dx, dy, ha = DESLOC_ROTULO.get(r.Modelo, (10, -4, "left"))
    ax.annotate(f"{r.Modelo}\n{r.Tamanho_KB:.0f} KB", (r.Latencia_1_lesao_ms, ab), xytext=(dx, dy),
                textcoords="offset points", fontsize=9, va="center", ha=ha,
                arrowprops=dict(arrowstyle="-", color="0.5", lw=.6) if r.Modelo in DESLOC_ROTULO else None)
ax.set_xscale("log"); ax.margins(x=.25, y=.12); ax.set_xlabel("Tempo de inferência de 1 lesão (ms, escala log)")
ax.set_ylabel("Acurácia balanceada média"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(PASTA_FIG, "fig_custo_desempenho.png"), dpi=300); plt.close()
print("\nFiguras em", PASTA_FIG)git add 07_analise_resultados.py