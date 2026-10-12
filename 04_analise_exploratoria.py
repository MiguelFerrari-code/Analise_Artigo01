import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar a base de dados
df = pd.read_csv('base_tabular_tcc.csv')

# Remover colunas não numéricas para a correlação
colunas_numericas = df.drop(columns=['Nome_Arquivo', 'Classe'])

print("Gerando Matriz de Correlação (Heatmap)...")
plt.figure(figsize=(10, 8))
matriz_corr = colunas_numericas.corr()
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlação de Pearson - Atributos de Cor e Textura')
plt.tight_layout()
plt.savefig('heatmap_correlacao.png', dpi=300)
plt.close()

print("Gerando Gráficos de Distribuição (Boxplots)...")
# Vamos escolher 4 variáveis chaves para mostrar a distribuição
features_destaque = ['HSV_H', 'LAB_B', 'GLCM_Contraste', 'GLCM_Energia']
cores = {'Cercospora': '#e74c3c', 'Miner': '#3498db', 'Phoma': '#2ecc71', 'Rust': '#f1c40f'}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, feature in enumerate(features_destaque):
    sns.boxplot(x='Classe', y=feature, data=df, palette=cores, ax=axes[i])
    axes[i].set_title(f'Distribuição de {feature} por Patologia')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Valor')

plt.tight_layout()
plt.savefig('boxplots_features.png', dpi=300)
plt.close()

print("Sucesso! Verifique as imagens 'heatmap_correlacao.png' e 'boxplots_features.png' na sua pasta.")