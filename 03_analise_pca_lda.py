import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# 1. Importação da Base Tabular
NOME_CSV = 'base_tabular_tcc.csv'
print(f"A carregar os dados de {NOME_CSV}...")
df = pd.read_csv(NOME_CSV)

# Separação das Features (X) e Rótulos/Classes (y)
features = [
    'HSV_H', 'HSV_S', 'HSV_V', 
    'LAB_L', 'LAB_A', 'LAB_B', 
    'GLCM_Contraste', 'GLCM_Homogeneidade', 
    'GLCM_Energia', 'GLCM_Correlacao'
]
X = df[features]
y = df['Classe']

# 2. Normalização dos Dados
# Fundamental para o PCA/LDA, pois as escalas de cor e textura são muito diferentes
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("A aplicar redução dimensional (PCA e LDA)...")

# 3. Análise de Componentes Principais (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['Classe'] = y

# 4. Análise Discriminante Linear (LDA)
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)
df_lda = pd.DataFrame(data=X_lda, columns=['LD1', 'LD2'])
df_lda['Classe'] = y

# 5. Visualização (Matplotlib e Seaborn)
print("A gerar os gráficos de sobreposição linear...")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Paleta de cores para as 4 doenças
cores = {'Cercospora': '#e74c3c', 'Miner': '#3498db', 'Phoma': '#2ecc71', 'Rust': '#f1c40f'}

# Gráfico PCA
sns.scatterplot(
    x='PC1', y='PC2', hue='Classe', palette=cores, 
    data=df_pca, ax=axes[0], alpha=0.7, edgecolor=None
)
axes[0].set_title('Análise de Componentes Principais (PCA)')
axes[0].set_xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')

# Gráfico LDA
sns.scatterplot(
    x='LD1', y='LD2', hue='Classe', palette=cores, 
    data=df_lda, ax=axes[1], alpha=0.7, edgecolor=None
)
axes[1].set_title('Análise Discriminante Linear (LDA)')
axes[1].set_xlabel(f'Discriminante Linear 1 ({lda.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'Discriminante Linear 2 ({lda.explained_variance_ratio_[1]*100:.1f}%)')

plt.tight_layout()

# Salvar o gráfico em alta resolução para usar no Artigo 1
NOME_GRAFICO = 'graficos_multivariados_tcc.png'
plt.savefig(NOME_GRAFICO, dpi=300)
print(f"Processo finalizado! Gráficos guardados em: {NOME_GRAFICO}")

# (Opcional) Exibir o gráfico na tela
plt.show()