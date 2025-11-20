import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv("nba_processed.csv")

# Usamos salary_zscore para agrupar por "Estatus económico" sin importar el año
# rank también ayuda a ver qué tan arriba estaban en la tabla
X = df[['salary_zscore', 'rank']]

# Método del Codo rápido para confirmar K
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Aplicamos K-Means con 3 Clusters (Banca, Rotación, Estrellas)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Renombrar clusters basado en el promedio de salario (para que 0 sea bajo y 2 alto, por ejemplo)
cluster_map = df.groupby('cluster')['salary'].mean().sort_values().index
cluster_names = {cluster_map[0]: 'Rol / Banca', cluster_map[1]: 'Titulares', cluster_map[2]: 'Superestrellas'}
df['cluster_label'] = df['cluster'].map(cluster_names)

print("Centroides de los Clusters:")
print(df.groupby('cluster_label')[['salary', 'salary_zscore']].mean())

# Visualización
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='season', y='salary', hue='cluster_label', palette='deep', alpha=0.6)
plt.title('Clustering de Jugadores NBA: Tipos de Contrato', fontsize=15)
plt.xlabel('Temporada')
plt.ylabel('Salario Real ($)')
plt.legend(title='Categoría')
plt.savefig('Practica7/player_clusters.png')
plt.show()