import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)

def optimal_k_analysis(X, max_k=15, problem_name=""):
    """
    Análisis para encontrar el número óptimo de clusters
    """
    print(f"\n ANÁLISIS DE K ÓPTIMO - {problem_name}")
    print("-" * 50)
    
    wcss = []  # Within-Cluster Sum of Square
    silhouette_scores = []
    calinski_scores = []
    davies_scores = []
    
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
        calinski_scores.append(calinski_harabasz_score(X, labels))
        davies_scores.append(davies_bouldin_score(X, labels))
    
    # Gráfico de Métodos del Codo y Silhouette
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Método del Codo
    axes[0, 0].plot(k_range, wcss, 'bo-', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Número de Clusters (k)')
    axes[0, 0].set_ylabel('WCSS (Within-Cluster Sum of Squares)')
    axes[0, 0].set_title(f'Método del Codo - {problem_name}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Silhouette Score
    axes[0, 1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Número de Clusters (k)')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title(f'Silhouette Score - {problem_name}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Calinski-Harabasz Score
    axes[1, 0].plot(k_range, calinski_scores, 'go-', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Número de Clusters (k)')
    axes[1, 0].set_ylabel('Calinski-Harabasz Score')
    axes[1, 0].set_title(f'Calinski-Harabasz Score - {problem_name}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Davies-Bouldin Score (menor es mejor)
    axes[1, 1].plot(k_range, davies_scores, 'mo-', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Número de Clusters (k)')
    axes[1, 1].set_ylabel('Davies-Bouldin Score')
    axes[1, 1].set_title(f'Davies-Bouldin Score - {problem_name}')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'Practica7/optimal_k_analysis_{problem_name.lower()}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Encontrar k óptimo basado en silhouette score
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    optimal_k_calinski = k_range[np.argmax(calinski_scores)]
    optimal_k_davies = k_range[np.argmin(davies_scores)]
    
    print(f"K óptimo según Silhouette: {optimal_k_silhouette}")
    print(f"K óptimo según Calinski-Harabasz: {optimal_k_calinski}")
    print(f"K óptimo según Davies-Bouldin: {optimal_k_davies}")
    
    # Consenso: usar Silhouette como principal
    return optimal_k_silhouette

def analyze_clusters(df, features, k, problem_name, use_pca=False):
    """
    Análisis completo de clusters
    """
    print(f"\n ANÁLISIS DE CLUSTERS - {problem_name} (k={k})")
    print("-" * 50)
    
    # Preparar datos
    X = df[features].copy()
    
    # Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Añadir labels al dataframe
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    
    # Métricas de evaluación
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    calinski_score = calinski_harabasz_score(X_scaled, cluster_labels)
    davies_score = davies_bouldin_score(X_scaled, cluster_labels)
    
    print(f" MÉTRICAS DE EVALUACIÓN:")
    print(f"   Silhouette Score: {silhouette_avg:.4f}")
    print(f"   Calinski-Harabasz Score: {calinski_score:.4f}")
    print(f"   Davies-Bouldin Score: {davies_score:.4f}")
    
    # Análisis de clusters
    print(f"\n ANÁLISIS POR CLUSTER:")
    cluster_stats = df_clustered.groupby('cluster')[features].agg(['mean', 'std', 'count'])
    print(cluster_stats)
    
    # Visualizaciones
    create_cluster_visualizations(df_clustered, X_scaled, cluster_labels, 
                               features, kmeans, problem_name, use_pca)
    
    return df_clustered, kmeans, scaler

def create_cluster_visualizations(df, X_scaled, labels, features, kmeans, problem_name, use_pca):
    """
    Crear visualizaciones completas de clusters
    """
    # 1. Visualización 2D con PCA
    plt.figure(figsize=(15, 12))
    
    if use_pca or len(features) > 2:
        # Reducción de dimensionalidad con PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_scaled)
        x_label, y_label = 'PCA Component 1', 'PCA Component 2'
        variance_ratio = pca.explained_variance_ratio_
        print(f"Varianza explicada por PCA: {variance_ratio}")
    else:
        # Usar las dos primeras features
        X_2d = X_scaled[:, :2]
        x_label, y_label = features[0], features[1]
    
    # Subplot 1: Clusters
    plt.subplot(2, 3, 1)
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', 
                         alpha=0.7, s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'Clusters - {problem_name}')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Centros de clusters
    plt.subplot(2, 3, 2)
    if use_pca or len(features) > 2:
        centers_2d = pca.transform(kmeans.cluster_centers_)
    else:
        centers_2d = kmeans.cluster_centers_[:, :2]
    
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', 
                alpha=0.3, s=30)
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', 
                s=200, label='Centros')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'Centros de Clusters - {problem_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Distribución de clusters
    plt.subplot(2, 3, 3)
    cluster_counts = df['cluster'].value_counts().sort_index()
    plt.bar(cluster_counts.index, cluster_counts.values, 
            color=plt.cm.viridis(np.linspace(0, 1, len(cluster_counts))))
    plt.xlabel('Cluster')
    plt.ylabel('Número de Puntos')
    plt.title(f'Distribución de Puntos por Cluster - {problem_name}')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Heatmap de características por cluster
    plt.subplot(2, 3, 4)
    cluster_means = df.groupby('cluster')[features].mean()
    sns.heatmap(cluster_means, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title(f'Características Promedio por Cluster - {problem_name}')
    
    # Subplot 5: Boxplot de salario por cluster
    plt.subplot(2, 3, 5)
    if 'salary' in df.columns:
        df.boxplot(column='salary', by='cluster', grid=False, ax=plt.gca())
        plt.title('Distribución de Salario por Cluster')
        plt.suptitle('')
    
    # Subplot 6: T-SNE (si hay muchas dimensiones)
    if len(features) > 2:
        plt.subplot(2, 3, 6)
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', 
                   alpha=0.7, s=50)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title(f't-SNE Visualization - {problem_name}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'Practica7/cluster_analysis_{problem_name.lower()}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Gráfico de radar para características de clusters
    if len(features) <= 8:  # Límite para gráfico de radar legible
        create_radar_chart(df, features, problem_name)

def create_radar_chart(df, features, problem_name):
    """
    Crear gráfico de radar para comparar clusters
    """
    from math import pi
    
    # Normalizar características para radar chart
    cluster_means = df.groupby('cluster')[features].mean()
    cluster_means_normalized = cluster_means.apply(
        lambda x: (x - x.min()) / (x.max() - x.min()), axis=0
    )
    
    # Configurar ángulos
    angles = [n / float(len(features)) * 2 * pi for n in range(len(features))]
    angles += angles[:1]  # Cerrar el círculo
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for cluster in cluster_means_normalized.index:
        values = cluster_means_normalized.loc[cluster].values.tolist()
        values += values[:1]  # Cerrar el círculo
        ax.plot(angles, values, linewidth=2, label=f'Cluster {cluster}')
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_ylim(0, 1)
    plt.title(f'Perfil de Clusters - {problem_name}', size=16, y=1.05)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(f'Practica7/radar_chart_{problem_name.lower()}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

# EJECUCIÓN PRINCIPAL
print("=" * 80)
print("PRÁCTICA 7: CLUSTERING CON K-MEANS - NBA SALARIES")
print("=" * 80)

# Cargar datos
df = pd.read_csv("edited-salaries.csv")
print(f"Dataset shape: {df.shape}")
print(f"Columnas: {df.columns.tolist()}")

# PROBLEMA 1: CLUSTERING DE JUGADORES POR SALARIO Y TEMPORADA
print("\n" + "=" * 80)
print("PROBLEMA 1: CLUSTERING DE JUGADORES (SALARIO + TEMPORADA)")
print("=" * 80)

features1 = ['salary', 'season']
df_problem1 = df[features1].copy()

# Análisis de k óptimo
optimal_k1 = optimal_k_analysis(df_problem1, max_k=10, 
                               problem_name="Salario y Temporada")

# Análisis de clusters
df_clustered1, kmeans1, scaler1 = analyze_clusters(
    df_problem1, features1, optimal_k1, 
    "Jugadores por Salario y Temporada", use_pca=False
)

# PROBLEMA 2: CLUSTERING DE POSICIONES POR SALARIO PROMEDIO Y TEMPORADA
print("\n" + "=" * 80)
print("PROBLEMA 2: CLUSTERING DE POSICIONES (SALARIO PROMEDIO)")
print("=" * 80)

# Agrupar por posición y temporada
df_position = df.groupby(['season', 'position']).agg({
    'salary': 'mean',
    'rank': 'count'  # número de jugadores en esa posición/temporada
}).reset_index()

features2 = ['salary', 'season', 'rank']
df_problem2 = df_position[features2].copy()

# Análisis de k óptimo
optimal_k2 = optimal_k_analysis(df_problem2, max_k=8, 
                               problem_name="Posiciones por Salario Promedio")

# Análisis de clusters
df_clustered2, kmeans2, scaler2 = analyze_clusters(
    df_problem2, features2, optimal_k2,
    "Posiciones por Salario Promedio", use_pca=True
)

# Añadir información de posición al resultado
df_clustered2['position'] = df_position['position']
print("\n📋 CLUSTERS DE POSICIONES:")
for cluster in sorted(df_clustered2['cluster'].unique()):
    positions_in_cluster = df_clustered2[df_clustered2['cluster'] == cluster]['position'].unique()
    avg_salary = df_clustered2[df_clustered2['cluster'] == cluster]['salary'].mean()
    print(f"  Cluster {cluster}: {positions_in_cluster} | Salario promedio: ${avg_salary:,.2f}")

# PROBLEMA 3: CLUSTERING MULTIDIMENSIONAL COMPLETO
print("\n" + "=" * 80)
print("PROBLEMA 3: CLUSTERING MULTIDIMENSIONAL COMPLETO")
print("=" * 80)

# Preparar datos con más características
df_multidimensional = df[['salary', 'season', 'rank']].copy()

# Codificar posición si está disponible
if 'position' in df.columns:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df_multidimensional['position_encoded'] = le.fit_transform(df['position'])

features3 = df_multidimensional.columns.tolist()

# Análisis de k óptimo
optimal_k3 = optimal_k_analysis(df_multidimensional, max_k=12, 
                               problem_name="Multidimensional Completo")

# Análisis de clusters
df_clustered3, kmeans3, scaler3 = analyze_clusters(
    df_multidimensional, features3, optimal_k3,
    "Clustering Multidimensional", use_pca=True
)

# ANÁLISIS COMPARATIVO DE TODOS LOS MODELOS
print("\n" + "=" * 80)
print("ANÁLISIS COMPARATIVO DE MODELOS")
print("=" * 80)

# Evaluar todos los modelos
models_comparison = []

for i, (df_clustered, features, k, problem_name) in enumerate([
    (df_clustered1, features1, optimal_k1, "Salario+Temporada"),
    (df_clustered2, features2, optimal_k2, "Posiciones Promedio"), 
    (df_clustered3, features3, optimal_k3, "Multidimensional")
], 1):
    
    X = df_clustered[features]
    X_scaled = scaler1.fit_transform(X) if i == 1 else scaler2.fit_transform(X) if i == 2 else scaler3.fit_transform(X)
    labels = df_clustered['cluster']
    
    silhouette = silhouette_score(X_scaled, labels)
    calinski = calinski_harabasz_score(X_scaled, labels)
    davies = davies_bouldin_score(X_scaled, labels)
    
    models_comparison.append({
        'Modelo': problem_name,
        'k': k,
        'Características': len(features),
        'Silhouette': silhouette,
        'Calinski-Harabasz': calinski,
        'Davies-Bouldin': davies
    })

comparison_df = pd.DataFrame(models_comparison)
print("\n COMPARACIÓN DE MODELOS:")
print(comparison_df.to_string(index=False))

# Gráfico de comparación
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

metrics = ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin']
for idx, metric in enumerate(metrics):
    axes[idx].bar(comparison_df['Modelo'], comparison_df[metric], 
                 color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[idx].set_title(f'Comparación de {metric}')
    axes[idx].set_ylabel(metric)
    axes[idx].tick_params(axis='x', rotation=45)
    
    # Añadir valores en las barras
    for i, v in enumerate(comparison_df[metric]):
        axes[idx].text(i, v + 0.01 * v, f'{v:.3f}', 
                      ha='center', va='bottom')

plt.tight_layout()
plt.savefig('Practica7/models_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# PREDICCIONES PARA NUEVOS DATOS
# =============================================================================
print("\n" + "=" * 80)
print("PREDICCIONES PARA NUEVOS JUGADORES")
print("=" * 80)

# Datos de ejemplo para predicción
new_players = pd.DataFrame({
    'salary': [500000, 5000000, 15000000, 25000000],
    'season': [2005, 2005, 2005, 2005],
    'rank': [200, 50, 10, 1]
})

# Usar el mejor modelo (multidimensional) para predicciones
if 'position_encoded' in features3:
    # Estimar posición encoded para nuevos datos (promedio)
    new_players['position_encoded'] = df_multidimensional['position_encoded'].mean()

new_players_scaled = scaler3.transform(new_players[features3])
predicted_clusters = kmeans3.predict(new_players_scaled)

new_players['cluster'] = predicted_clusters
print("\n PREDICCIONES DE CLUSTERS:")
print(new_players.to_string(index=False))

# =============================================================================
# EXPORTACIÓN DE RESULTADOS
# =============================================================================
print("\n" + "=" * 80)
print("EXPORTACIÓN DE RESULTADOS")
print("=" * 80)

# Exportar datos clusterizados
df_clustered1.to_csv('Practica7/clustered_players_salary_season.csv', index=False)
df_clustered2.to_csv('Practica7/clustered_positions.csv', index=False)
df_clustered3.to_csv('Practica7/clustered_multidimensional.csv', index=False)
comparison_df.to_csv('Practica7/models_comparison.csv', index=False)

print("ARCHIVOS EXPORTADOS:")
print("  Practica7/clustered_players_salary_season.csv")
print("  Practica7/clustered_positions.csv") 
print("  Practica7/clustered_multidimensional.csv")
print("  Practica7/models_comparison.csv")
print("  Practica7/optimal_k_analysis_*.png")
print("  Practica7/cluster_analysis_*.png")
print("  Practica7/radar_chart_*.png")
print("  Practica7/models_comparison.png")

print("\n" + "=" * 80)
print("¡CLUSTERING CON K-MEANS COMPLETADO!")
print("=" * 80)