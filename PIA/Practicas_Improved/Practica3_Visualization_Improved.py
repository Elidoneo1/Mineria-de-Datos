import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos procesados
df = pd.read_csv("nba_processed.csv")
plt.style.use('seaborn-v0_8-darkgrid')

# 1. Inflación de Salarios: Boxplot por Año
# Muestra cómo ha subido la media y los bigotes (superestrellas)
plt.figure(figsize=(16, 8))
sns.boxplot(x='season', y='salary', data=df, palette="viridis")
plt.title('Explosión Salarial en la NBA (2000-2020)', fontsize=16)
plt.xticks(rotation=45)
plt.ylabel('Salario ($)')
plt.tight_layout()
plt.savefig('Practica3/1_inflacion_salarial.png')
plt.show()

# 2. ¿Quién gana más relativamente? (Z-Score por Posición)
# Usamos Z-Score para ver qué posición está "sobrepagada" comparada con el promedio de su año
plt.figure(figsize=(12, 6))
sns.violinplot(x='position', y='salary_zscore', data=df, palette="muted")
plt.axhline(0, color='red', linestyle='--', label='Promedio de la Liga')
plt.title('Distribución de Salarios Normalizados por Posición', fontsize=16)
plt.ylabel('Z-Score (Desviaciones Estándar)')
plt.legend()
plt.savefig('Practica3/2_posiciones_zscore.png')
plt.show()

# 3. Top 10 Equipos que más gastan (Histórico)
total_spend = df.groupby('team')['salary'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
total_spend.plot(kind='barh', color='teal')
plt.title('Top 10 Equipos con Mayor Gasto Histórico (2000-2020)', fontsize=16)
plt.xlabel('Gasto Total Acumulado ($)')
plt.gca().invert_yaxis() # Invertir para que el #1 esté arriba
plt.tight_layout()
plt.savefig('Practica3/3_equipos_gastadores.png')
plt.show()

# 4. Mapa de Calor de Correlaciones
# Usamos las variables codificadas para ver relaciones reales
corr_cols = ['salary', 'season', 'position_encoded', 'rank', 'salary_zscore']
plt.figure(figsize=(10, 8))
sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Matriz de Correlación', fontsize=16)
plt.savefig('Practica3/4_heatmap.png')
plt.show()