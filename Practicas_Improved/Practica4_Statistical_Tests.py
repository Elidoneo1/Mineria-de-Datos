import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo
plt.style.use('seaborn-v0_8')

# Cargar datos
df = pd.read_csv("nba_processed.csv")

print("="*60)
print("TEST 1: ANOVA - ¿Influye la posición en el salario relativo?")
print("Variable dependiente: Salary Z-Score")
print("="*60)

# 1. FILTRADO: Solo usamos las 5 posiciones principales para evitar errores de 'nan'
valid_positions = ['Center', 'Power Forward', 'Small Forward', 'Point Guard', 'Shooting Guard']
df_anova = df[df['position'].isin(valid_positions)]

# Verificar conteos antes del test
print("Conteo de muestras por posición (filtrado):")
print(df_anova['position'].value_counts())
print("-" * 30)

# 2. Preparar grupos para ANOVA
groups = [df_anova[df_anova['position'] == pos]['salary_zscore'] for pos in valid_positions]

# 3. Ejecutar ANOVA
f_stat, p_value = stats.f_oneway(*groups)

print(f"Estadístico F: {f_stat:.4f}")
print(f"Valor P: {p_value:.4e}")  # Notación científica si es muy pequeño

# 4. Interpretación
if p_value < 0.05:
    print("\nCONCLUSIÓN: RECHAZAMOS la hipótesis nula.")
    print("--> SÍ hay diferencias significativas en los salarios según la posición.")
    print("--> Esto significa que algunas posiciones se pagan mejor que otras sistemáticamente.")
else:
    print("\nCONCLUSIÓN: NO rechazamos la hipótesis nula.")
    print("--> No hay evidencia suficiente para afirmar que la posición determina el salario.")

# 5. Visualización de las medias (Post-Hoc visual)
print("\nPromedio de Z-Score por Posición:")
print(df_anova.groupby('position')['salary_zscore'].mean().sort_values(ascending=False))

plt.figure(figsize=(10, 6))
sns.barplot(x='position', y='salary_zscore', data=df_anova, estimator=lambda x: sum(x) / len(x), errorbar=None, palette='viridis')
plt.title('Promedio de Salario Relativo (Z-Score) por Posición')
plt.axhline(0, color='red', linestyle='--', label='Promedio de la Liga')
plt.ylabel('Z-Score (Desviación del promedio anual)')
plt.legend()
plt.tight_layout()
plt.savefig('Practica4/anova_results.png')
plt.show()


print("\n" + "="*60)
print("TEST 2: T-Test - Lakers vs Knicks")
print("¿Pagan los Lakers significativamente más que los Knicks (relativo al año)?")
print("="*60)

# Filtrar datos de ambos equipos
lakers = df[df['team'] == 'Los Angeles Lakers']['salary_zscore'].dropna()
knicks = df[df['team'] == 'New York Knicks']['salary_zscore'].dropna()

# Ejecutar T-Test
t_stat, p_val = stats.ttest_ind(lakers, knicks)

print(f"Promedio Z-Score Lakers: {lakers.mean():.4f}")
print(f"Promedio Z-Score Knicks: {knicks.mean():.4f}")
print(f"T-Statistic: {t_stat:.4f}")
print(f"P-Value: {p_val:.4f}")

if p_val < 0.05:
    print("\nCONCLUSIÓN: Hay una diferencia significativa entre lo que pagan Lakers y Knicks.")
else:
    print("\nCONCLUSIÓN: No hay diferencia significativa.")
    print("Aunque en dinero total pueda variar, RELATIVAMENTE a sus épocas, ambos equipos gastan de forma similar respecto al promedio de la liga.")