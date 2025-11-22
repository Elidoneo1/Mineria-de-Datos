import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler ### NUEVO: Importante para K-Means
import os

# Crear carpeta si no existe
if not os.path.exists('Practica7'):
    os.makedirs('Practica7')

# Cargar datos
df = pd.read_csv("nba_processed.csv")

print("="*60)
print("CLUSTERING: Tipos de Jugadores (Banca vs Estrellas)")
print("="*60)

# Selección de variables
# Usamos salary_zscore (estatus económico relativo) y rank (estatus deportivo)
X = df[['salary_zscore', 'rank']]

# --- 1. ESCALADO DE DATOS (MEJORA TÉCNICA) ---
# Es necesario porque 'rank' va de 1 a 500, y 'zscore' de -2 a 4. 
# Sin esto, el rank domina todo el cálculo.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 2. MÉTODO DEL CODO (ELBOW METHOD) ---
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled) # Usamos datos escalados
    wcss.append(kmeans.inertia_)

# Guardamos el gráfico del codo para referencia
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Método del Codo')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')
plt.savefig('Practica7/elbow_method.png')
plt.close()

# --- 3. APLICACIÓN DE K-MEANS ---
# Usamos k=3 (Banca, Rotación, Estrellas)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Asignamos los clusters al dataframe original
df['cluster'] = clusters

# Renombrar clusters basado en el promedio de salario
# Calculamos el salario promedio por cluster para saber cuál es cuál
cluster_means = df.groupby('cluster')['salary'].mean().sort_values()
# Mapeamos: el de menor salario es 0, medio 1, mayor 2
cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(cluster_means.index)}
df['cluster_sorted'] = df['cluster'].map(cluster_mapping)

# Etiquetas legibles
cluster_names = {0: 'Rol / Banca', 1: 'Titulares', 2: 'Superestrellas'}
df['cluster_label'] = df['cluster_sorted'].map(cluster_names)

print("\nCentroides de los Clusters (Promedios):")
print(df.groupby('cluster_label')[['salary', 'salary_zscore', 'rank']].mean())

# --- 4. VISUALIZACIÓN 1: Serie de Tiempo (Original) ---
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='season', y='salary', hue='cluster_label', palette='viridis', alpha=0.6)
plt.title('Clustering de Jugadores NBA: Distribución en el Tiempo', fontsize=15)
plt.xlabel('Temporada')
plt.ylabel('Salario Real ($)')
plt.legend(title='Categoría')
plt.savefig('Practica7/player_clusters_timeline.png')
plt.show()

# --- 5. VISUALIZACIÓN 2: PCA (EL CAMBIO SOLICITADO) ---
# PCA reduce las dimensiones para ver mejor la separación matemática
print("\nGenerando visualización PCA...")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Crear DataFrame temporal para graficar
df_pca = pd.DataFrame(data=X_pca, columns=['Componente 1', 'Componente 2'])
df_pca['Categoría'] = df['cluster_label']

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='Componente 1', 
    y='Componente 2', 
    hue='Categoría', 
    data=df_pca, 
    palette='viridis', 
    s=100,
    alpha=0.8
)
plt.title('Clusters visualizados mediante PCA (Reducción de Dimensiones)', fontsize=15)
plt.xlabel('Componente Principal 1 (Combinación Rank/Salario)')
plt.ylabel('Componente Principal 2')
plt.legend(title='Tipo de Jugador')
plt.grid(True, alpha=0.3)
plt.savefig('Practica7/pca_clusters.png')
plt.show()

print("Gráficos guardados en la carpeta Practica7.")