"""
ETAPA 08 (Artigo 02) - Verificação de robustez do ajuste do boosting

Repete XGBoost e LightGBM com busca aleatória de 60 combinações (em vez de 20),
nas mesmas 15 partições externas da Etapa 06, e compara com o SVM (RBF) da
Etapa 06 pelo t-teste corrigido. Objetivo: verificar se a diferença a favor do
SVM se deve a um ajuste insuficiente dos modelos de boosting.

Saída: resultados_artigo02/tabelas/busca_estendida.csv
"""
import os, importlib.util, time, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import balanced_accuracy_score
warnings.filterwarnings("ignore")
src = open("06_treino_modelos.py", encoding="utf-8").read().split("linhas, preds, params = [], [], []")[0]
ns = {}; exec(src, ns)
X, y, grupos, modelos, SEMENTE = ns["X"], ns["y"], ns["grupos"], ns["modelos"], ns["SEMENTE"]
N_BUSCA = 60
base = pd.read_csv(os.path.join("resultados_artigo02", "tabelas", "metricas_por_particao.csv"))
alvo = {n: (e, s, w) for n, e, s, w in modelos() if n in ("XGBoost", "LightGBM")}
linhas = []; t_ini = time.time()
for rep in range(3):
    ext = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMENTE + rep)
    for k, (tr, te) in enumerate(ext.split(X, y, grupos)):
        for nome, (est, esp, peso) in alvo.items():
            kw = {"sample_weight": compute_sample_weight("balanced", y[tr])} if peso else {}
            inn = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEMENTE + 100 * rep + k)
            b = RandomizedSearchCV(clone(est), esp, n_iter=N_BUSCA, scoring="balanced_accuracy", cv=inn,
                                   random_state=SEMENTE + k, n_jobs=-1).fit(X[tr], y[tr], groups=grupos[tr], **kw)
            ab = balanced_accuracy_score(y[te], b.predict(X[te]))
            linhas.append({"Modelo": nome, "Repeticao": rep, "Particao": k, "AB_60": ab})
            print(f"rep {rep} part {k} {nome} AB60={ab:.3f} [{(time.time()-t_ini)/60:.1f} min]", flush=True)
r = pd.DataFrame(linhas)
razao = (base.n_teste / base.n_treino).mean()
def t_corr(a, b):
    d = np.asarray(a) - np.asarray(b); s2 = np.var(d, ddof=1)
    t = d.mean() / np.sqrt((1 / len(d) + razao) * s2); return d.mean(), t, 2 * stats.t.sf(abs(t), len(d) - 1)
svm = base[base.Modelo == "SVM (RBF)"].sort_values(["Repeticao", "Particao"]).Acuracia_balanceada.to_numpy()
out = []
for nome in alvo:
    a60 = r[r.Modelo == nome].sort_values(["Repeticao", "Particao"]).AB_60.to_numpy()
    a20 = base[base.Modelo == nome].sort_values(["Repeticao", "Particao"]).Acuracia_balanceada.to_numpy()
    d1, _, p1 = t_corr(a60, a20); d2, _, p2 = t_corr(svm, a60)
    out.append({"Modelo": nome, "AB_20_iter": a20.mean(), "AB_60_iter": a60.mean(), "Ganho": d1, "p_ganho": p1,
                "SVM_menos_60iter": d2, "p_SVM_vs_60iter": p2})
out = pd.DataFrame(out).round(4)
out.to_csv(os.path.join("resultados_artigo02", "tabelas", "busca_estendida.csv"), index=False)
r.to_csv(os.path.join("resultados_artigo02", "tabelas", "busca_estendida_particoes.csv"), index=False)
print(out.to_string(index=False))