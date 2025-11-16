import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import numpy as np

# Configuración de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='orgtbl'))

# Leemos el csv
df = pd.read_csv("edited-salaries.csv")

# 1. HISTOGRAMA - Distribución de salarios
plt.figure(figsize=(10, 6))
plt.hist(df['salary'], bins=50, alpha=0.7, edgecolor='black')
plt.title('Distribución de Salarios en la NBA (2000-2009)')
plt.xlabel('Salario ($)')
plt.ylabel('Frecuencia')
plt.grid(True, alpha=0.3)
plt.savefig('Practica3/histograma_salarios.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. BOXPLOT - Salarios por posición
plt.figure(figsize=(12, 6))
df.boxplot(column='salary', by='position', grid=False)
plt.title('Distribución de Salarios por Posición')
plt.suptitle('')  # Elimina título automático
plt.xlabel('Posición')
plt.ylabel('Salario ($)')
plt.xticks(rotation=45)
plt.savefig('Practica3/boxplot_posiciones.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. DIAGRAMA DE TORTA (PIE) - Distribución de posiciones
plt.figure(figsize=(8, 8))
position_counts = df['position'].value_counts()
plt.pie(position_counts.values, labels=position_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribución de Jugadores por Posición')
plt.savefig('Practica3/pie_posiciones.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. GRÁFICO DE LÍNEAS - Evolución salarial por año
plt.figure(figsize=(12, 6))
salario_promedio_anual = df.groupby('season')['salary'].mean()
salario_maximo_anual = df.groupby('season')['salary'].max()

plt.plot(salario_promedio_anual.index, salario_promedio_anual.values, 
         marker='o', linewidth=2, label='Salario Promedio')
plt.plot(salario_maximo_anual.index, salario_maximo_anual.values, 
         marker='s', linewidth=2, label='Salario Máximo')
plt.title('Evolución de Salarios en la NBA (2000-2009)')
plt.xlabel('Año')
plt.ylabel('Salario ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('Practica3/linea_evolucion_salarial.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. SCATTER PLOT - Salario vs Año (con posición)
plt.figure(figsize=(12, 8))
positions = df['position'].unique()

for position in positions:
    subset = df[df['position'] == position]
    plt.scatter(subset['season'], subset['salary'], alpha=0.6, label=position, s=30)

plt.title('Relación entre Salario y Año por Posición')
plt.xlabel('Año')
plt.ylabel('Salario ($)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.savefig('Practica3/scatter_salario_año.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. DIAGRAMA DE BARRAS - Top 10 equipos con mayor salario promedio
plt.figure(figsize=(12, 6))
top_teams = df.groupby('team')['salary'].mean().nlargest(10)
top_teams.plot(kind='bar', color='skyblue')
plt.title('Top 10 Equipos con Mayor Salario Promedio (2000-2009)')
plt.xlabel('Equipo')
plt.ylabel('Salario Promedio ($)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.savefig('Practica3/bar_top_equipos.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. HEATMAP - Salario promedio por posición y año (USANDO LOOPS)
plt.figure(figsize=(12, 8))

# Preparamos datos para heatmap
heatmap_data = df.pivot_table(values='salary', index='position', columns='season', aggfunc='mean')

# Usamos loop para personalizar (opcional - demostración de uso de loops)
positions = heatmap_data.index
seasons = heatmap_data.columns

# Crear heatmap manualmente con loops (alternativa a seaborn)
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(heatmap_data.values, cmap='YlOrRd')

# Añadir etiquetas con loops
ax.set_xticks(np.arange(len(seasons)))
ax.set_yticks(np.arange(len(positions)))
ax.set_xticklabels(seasons)
ax.set_yticklabels(positions)

# Rotar etiquetas y añadir valores en cada celda
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop para añadir texto en cada celda
for i in range(len(positions)):
    for j in range(len(seasons)):
        text = ax.text(j, i, f'{heatmap_data.iloc[i, j]:.0f}',
                       ha="center", va="center", color="black", fontsize=8)

ax.set_title("Salario Promedio por Posición y Año")
fig.tight_layout()
plt.savefig('Practica3/heatmap_posicion_año.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. GRÁFICO DE VIOLÍN - Distribución salarial por posición
plt.figure(figsize=(12, 8))
sns.violinplot(x='position', y='salary', data=df)
plt.title('Distribución de Salarios por Posición (Violin Plot)')
plt.xlabel('Posición')
plt.ylabel('Salario ($)')
plt.xticks(rotation=45)
plt.savefig('Practica3/violin_posiciones.png', dpi=300, bbox_inches='tight')
plt.show()

# Mantenemos tus agrupaciones originales para análisis
print("=== AGRUPACIONES ORIGINALES ===")
df_by_team = df.groupby(["team","season"]).agg({'salary': ['sum','count','mean','min','max']})
print_tabulate(df_by_team.head())

df_by_sea = df.groupby(["season"]).agg({'salary': ['sum','count','mean','min','max']})
print_tabulate(df_by_sea.head())

print("¡Se generaron 8 diagramas diferentes! Revisa la carpeta Practica3/")