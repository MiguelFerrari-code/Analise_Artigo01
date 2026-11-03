"""
ETAPA 06 (Artigo 02) - Treino e comparação de classificadores

Base: base_tabular_tcc.csv (Artigo 01) - 1.858 lesões x 18 atributos.

Modelos comparados:
  LDA       - referência linear do Artigo 01 (sem ajuste de hiperparâmetros)
  RF        - Random Forest (previsto no Pré-TCC)
  SVM       - SVM com kernel RBF (previsto no Pré-TCC)
  XGBoost   - gradient boosting (sugerido pelo orientador)
  LightGBM  - gradient boosting (sugerido pelo orientador)
  XGBoost e LightGBM também SEM ponderação de classes, para medir o efeito
  do tratamento do desbalanceamento.

Protocolo (validação cruzada aninhada, agrupada por folha):
  - Externa: StratifiedGroupKFold com 5 partições, repetida 3 vezes com
    embaralhamentos diferentes (15 avaliações por modelo). Cada partição
    externa equivale a uma divisão 80/20 por folha.
  - Interna: StratifiedGroupKFold com 3 partições dentro do treino externo,
    para escolher hiperparâmetros por busca aleatória (20 combinações),
    maximizando a acurácia balanceada.
  - Lesões da mesma folha nunca aparecem em treino e teste ao mesmo tempo.
  - Desbalanceamento: pesos inversamente proporcionais à frequência da classe
    (class_weight='balanced' ou sample_weight equivalente).

Saídas (resultados_artigo02/tabelas/):
  metricas_por_particao.csv   uma linha por modelo x partição externa
  predicoes_oof.csv           predições de teste de cada repetição
  melhores_parametros.csv     hiperparâmetros escolhidos em cada partição
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             recall_score, cohen_kappa_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
PASTA_TAB = os.path.join("resultados_artigo02", "tabelas")
os.makedirs(PASTA_TAB, exist_ok=True)

SEMENTE = 42
N_REPETICOES = 3
N_EXTERNAS = 5
N_INTERNAS = 3
N_BUSCA = 20
N_JOBS = -1

df = pd.read_csv("base_tabular_tcc.csv")
ATRIBUTOS = [c for c in df.columns if c not in ("ID_Lesao", "ID_Folha", "Classe")]
X = df[ATRIBUTOS].to_numpy(dtype=float)
codificador = LabelEncoder().fit(df["Classe"])
CLASSES = list(codificador.classes_)
y = codificador.transform(df["Classe"])
grupos = df["ID_Folha"].to_numpy()
print(f"{len(df)} lesões, {len(ATRIBUTOS)} atributos, {len(np.unique(grupos))} folhas, classes={CLASSES}")


def modelos():
    """(nome, estimador, espaço de busca ou None, usa sample_weight?)"""
    return [
        ("LDA", make_pipeline(StandardScaler(), LDA()), None, False),
        ("Random Forest",
         RandomForestClassifier(class_weight="balanced", random_state=SEMENTE, n_jobs=1),
         {"n_estimators": randint(200, 601), "max_depth": [None, 8, 12, 20],
          "min_samples_leaf": randint(1, 6), "max_features": ["sqrt", 0.33, 0.5]}, False),
        ("SVM (RBF)",
         make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced")),
         {"svc__C": loguniform(0.1, 100), "svc__gamma": loguniform(1e-3, 1)}, False),
        ("XGBoost",
         XGBClassifier(objective="multi:softprob", tree_method="hist", random_state=SEMENTE,
                       n_jobs=1, verbosity=0),
         {"n_estimators": randint(100, 601), "max_depth": randint(2, 9),
          "learning_rate": loguniform(0.01, 0.3), "subsample": uniform(0.6, 0.4),
          "colsample_bytree": uniform(0.6, 0.4), "min_child_weight": randint(1, 11),
          "reg_lambda": loguniform(0.1, 10)}, True),
        ("LightGBM",
         LGBMClassifier(class_weight="balanced", subsample_freq=1, random_state=SEMENTE,
                        n_jobs=1, verbose=-1),
         {"n_estimators": randint(100, 601), "num_leaves": randint(8, 65),
          "learning_rate": loguniform(0.01, 0.3), "subsample": uniform(0.6, 0.4),
          "colsample_bytree": uniform(0.6, 0.4), "min_child_samples": randint(5, 51),
          "reg_lambda": loguniform(0.1, 10)}, False),
        ("XGBoost sem peso",
         XGBClassifier(objective="multi:softprob", tree_method="hist", random_state=SEMENTE,
                       n_jobs=1, verbosity=0),
         {"n_estimators": randint(100, 601), "max_depth": randint(2, 9),
          "learning_rate": loguniform(0.01, 0.3), "subsample": uniform(0.6, 0.4),
          "colsample_bytree": uniform(0.6, 0.4), "min_child_weight": randint(1, 11),
          "reg_lambda": loguniform(0.1, 10)}, False),
        ("LightGBM sem peso",
         LGBMClassifier(subsample_freq=1, random_state=SEMENTE, n_jobs=1, verbose=-1),
         {"n_estimators": randint(100, 601), "num_leaves": randint(8, 65),
          "learning_rate": loguniform(0.01, 0.3), "subsample": uniform(0.6, 0.4),
          "colsample_bytree": uniform(0.6, 0.4), "min_child_samples": randint(5, 51),
          "reg_lambda": loguniform(0.1, 10)}, False),
    ]


linhas, preds, params = [], [], []
inicio_total = time.time()
for rep in range(N_REPETICOES):
    externa = StratifiedGroupKFold(n_splits=N_EXTERNAS, shuffle=True, random_state=SEMENTE + rep)
    for k, (tr, te) in enumerate(externa.split(X, y, grupos)):
        Xtr, Xte, ytr, yte, gtr = X[tr], X[te], y[tr], y[te], grupos[tr]
        for nome, est, espaco, usa_peso in modelos():
            fit_kw = {"sample_weight": compute_sample_weight("balanced", ytr)} if usa_peso else {}
            t0 = time.perf_counter()
            if espaco is None:
                modelo = clone(est).fit(Xtr, ytr)
                melhor = {}
            else:
                interna = StratifiedGroupKFold(n_splits=N_INTERNAS, shuffle=True,
                                               random_state=SEMENTE + 100 * rep + k)
                busca = RandomizedSearchCV(clone(est), espaco, n_iter=N_BUSCA,
                                           scoring="balanced_accuracy", cv=interna,
                                           random_state=SEMENTE + k, n_jobs=N_JOBS, refit=True)
                busca.fit(Xtr, ytr, groups=gtr, **fit_kw)
                modelo, melhor = busca.best_estimator_, busca.best_params_
            t_treino = time.perf_counter() - t0
            t1 = time.perf_counter()
            yp = modelo.predict(Xte)
            t_pred = (time.perf_counter() - t1) / len(te) * 1000  # ms por lesão
            rec = recall_score(yte, yp, average=None, labels=range(len(CLASSES)))
            linha = {"Modelo": nome, "Repeticao": rep, "Particao": k, "n_teste": len(te),
                     "n_treino": len(tr),
                     "Acuracia": accuracy_score(yte, yp),
                     "Acuracia_balanceada": balanced_accuracy_score(yte, yp),
                     "F1_macro": f1_score(yte, yp, average="macro"),
                     "Kappa": cohen_kappa_score(yte, yp),
                     "Tempo_treino_s": t_treino, "Tempo_predicao_ms_por_lesao": t_pred}
            for c, r in zip(CLASSES, rec):
                linha[f"Sensibilidade_{c}"] = r
            linhas.append(linha)
            preds.append(pd.DataFrame({"Modelo": nome, "Repeticao": rep, "Particao": k,
                                       "ID_Lesao": df["ID_Lesao"].to_numpy()[te],
                                       "Real": codificador.inverse_transform(yte),
                                       "Previsto": codificador.inverse_transform(yp)}))
            params.append({"Modelo": nome, "Repeticao": rep, "Particao": k,
                           **{p: (round(v, 5) if isinstance(v, float) else v) for p, v in melhor.items()}})
            print(f"rep {rep} part {k} {nome:18s} AB={linha['Acuracia_balanceada']:.3f} "
                  f"({t_treino:.1f}s)  [{(time.time() - inicio_total) / 60:.1f} min]", flush=True)

res = pd.DataFrame(linhas)
res.to_csv(os.path.join(PASTA_TAB, "metricas_por_particao.csv"), index=False)
pd.concat(preds).to_csv(os.path.join(PASTA_TAB, "predicoes_oof.csv"), index=False)
pd.DataFrame(params).to_csv(os.path.join(PASTA_TAB, "melhores_parametros.csv"), index=False)

resumo = res.groupby("Modelo")[["Acuracia_balanceada", "F1_macro", "Acuracia", "Kappa"]].agg(["mean", "std"])
print("\nResumo (média ± desvio em 15 partições externas):")
print(resumo.round(3).sort_values(("Acuracia_balanceada", "mean"), ascending=False).to_string())
print(f"\nTempo total: {(time.time() - inicio_total) / 60:.1f} min")