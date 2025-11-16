import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import f_oneway, kruskal, normaltest, levene
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
plt.style.use('seaborn-v0_8')
np.random.seed(42)

def check_normality(data_groups):
    """Verifica normalidad usando Shapiro-Wilk test"""
    print("=== VERIFICACIÓN DE NORMALIDAD ===")
    all_normal = True
    for group_name, group_data in data_groups.items():
        stat, p_value = stats.shapiro(group_data)
        print(f"{group_name}: p-value = {p_value:.4f}")
        if p_value < 0.05:
            print(f"  → {group_name} NO es normal (p < 0.05)")
            all_normal = False
        else:
            print(f"  → {group_name} es normal (p ≥ 0.05)")
    return all_normal

def check_homogeneity(data_groups):
    """Verifica homogeneidad de varianzas usando Levene test"""
    print("\n=== VERIFICACIÓN DE HOMOGENEIDAD DE VARIANZAS ===")
    stat, p_value = levene(*data_groups.values())
    print(f"Levene test: p-value = {p_value:.4f}")
    if p_value < 0.05:
        print("  → Las varianzas NO son homogéneas (p < 0.05)")
        return False
    else:
        print("  → Las varianzas son homogéneas (p ≥ 0.05)")
        return True

def perform_anova(data_groups):
    """Realiza ANOVA one-way"""
    print("\n=== ANOVA ONE-WAY ===")
    f_stat, p_value = f_oneway(*data_groups.values())
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("  → Hay diferencias significativas entre grupos (p < 0.05)")
        return True, f_stat, p_value
    else:
        print("  → NO hay diferencias significativas entre grupos (p ≥ 0.05)")
        return False, f_stat, p_value

def perform_kruskal(data_groups):
    """Realiza Kruskal-Wallis test (ANOVA no paramétrico)"""
    print("\n=== KRUSKAL-WALLIS TEST ===")
    h_stat, p_value = kruskal(*data_groups.values())
    print(f"H-statistic: {h_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("  → Hay diferencias significativas entre grupos (p < 0.05)")
        return True, h_stat, p_value
    else:
        print("  → NO hay diferencias significativas entre grupos (p ≥ 0.05)")
        return False, h_stat, p_value

def tukey_posthoc(data_df, group_col, value_col):
    """Realiza test post-hoc de Tukey"""
    print(f"\n=== POST-HOC TUKEY TEST ({group_col}) ===")
    tukey = pairwise_tukeyhsd(endog=data_df[value_col],
                            groups=data_df[group_col],
                            alpha=0.05)
    print(tukey.summary())
    
    # Gráfico de comparaciones
    fig, ax = plt.subplots(figsize=(10, 6))
    tukey.plot_simultaneous(ax=ax)
    plt.title(f'Comparaciones Múltiples de Tukey - {group_col}')
    plt.savefig(f'Practica4/tukey_{group_col}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Cargar datos
df = pd.read_csv("edited-salaries.csv")
print(f"Dataset shape: {df.shape}")
print(f"Posiciones únicas: {df['position'].unique()}")

# =============================================================================
# HIPÓTESIS 1: ¿Existen diferencias significativas en salarios entre POSICIONES?
# =============================================================================
print("\n" + "="*80)
print("HIPÓTESIS 1: DIFERENCIAS EN SALARIOS POR POSICIÓN")
print("="*80)

# Preparar datos para posición
position_groups = {}
for position in df['position'].unique():
    position_groups[position] = df[df['position'] == position]['salary'].values

# Verificar supuestos
is_normal = check_normality(position_groups)
is_homogeneous = check_homogeneity(position_groups)

# Seleccionar prueba estadística
if is_normal and is_homogeneous:
    print("\n→ Usando ANOVA paramétrico (datos normales y varianzas homogéneas)")
    has_differences, stat, p_val = perform_anova(position_groups)
else:
    print("\n→ Usando Kruskal-Wallis (datos no normales o varianzas no homogéneas)")
    has_differences, stat, p_val = perform_kruskal(position_groups)

# Post-hoc si hay diferencias
if has_differences:
    # Preparar datos para Tukey
    tukey_df = df[['position', 'salary']].copy()
    tukey_posthoc(tukey_df, 'position', 'salary')

# Gráfico de comparación
plt.figure(figsize=(12, 6))
df.boxplot(column='salary', by='position', grid=False)
plt.title('Comparación de Salarios por Posición')
plt.suptitle('')
plt.xlabel('Posición')
plt.ylabel('Salario ($)')
plt.xticks(rotation=45)
plt.savefig('Practica4/boxplot_positions_statistical.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# HIPÓTESIS 2: ¿Existen diferencias significativas en salarios entre AÑOS?
# =============================================================================
print("\n" + "="*80)
print("HIPÓTESIS 2: DIFERENCIAS EN SALARIOS POR AÑO")
print("="*80)

# Para años, usamos una muestra representativa de cada año (para evitar datos desbalanceados)
year_groups = {}
for year in sorted(df['season'].unique()):
    year_data = df[df['season'] == year]['salary']
    # Tomamos muestra aleatoria de 50 jugadores por año para balancear
    if len(year_data) > 50:
        year_data = year_data.sample(50, random_state=42)
    year_groups[year] = year_data.values

# Verificar supuestos
is_normal_years = check_normality(year_groups)
is_homogeneous_years = check_homogeneity(year_groups)

# Seleccionar prueba
if is_normal_years and is_homogeneous_years:
    print("\n→ Usando ANOVA paramétrico")
    has_differences_years, stat_years, p_val_years = perform_anova(year_groups)
else:
    print("\n→ Usando Kruskal-Wallis")
    has_differences_years, stat_years, p_val_years = perform_kruskal(year_groups)

# =============================================================================
# HIPÓTESIS 3: ¿Existen diferencias entre equipos de CONFERENCIA?
# =============================================================================
print("\n" + "="*80)
print("HIPÓTESIS 3: DIFERENCIAS ENTRE CONFERENCIAS (ESTE/OESTE)")
print("="*80)

# Crear variable de conferencia (simplificación)
east_teams = ['Boston Celtics', 'New York Knicks', 'Philadelphia 76ers', 'Miami Heat', 
              'Orlando Magic', 'Washington Wizards', 'Atlanta Hawks', 'Charlotte Hornets',
              'Chicago Bulls', 'Cleveland Cavaliers', 'Detroit Pistons', 'Indiana Pacers',
              'Milwaukee Bucks', 'New Jersey Nets', 'Toronto Raptors']

df['conference'] = df['team'].apply(lambda x: 'East' if x in east_teams else 'West')

# Test t para dos grupos independientes
east_salaries = df[df['conference'] == 'East']['salary']
west_salaries = df[df['conference'] == 'West']['salary']

print("=== T-TEST PARA MUESTRAS INDEPENDIENTES ===")
t_stat, p_value = stats.ttest_ind(east_salaries, west_salaries, equal_var=False)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("  → Hay diferencia significativa entre conferencias (p < 0.05)")
    print(f"  → Salario promedio Este: ${east_salaries.mean():,.2f}")
    print(f"  → Salario promedio Oeste: ${west_salaries.mean():,.2f}")
else:
    print("  → NO hay diferencia significativa entre conferencias (p ≥ 0.05)")

# Gráfico de conferencias
plt.figure(figsize=(10, 6))
df.boxplot(column='salary', by='conference', grid=False)
plt.title('Comparación de Salarios por Conferencia')
plt.suptitle('')
plt.xlabel('Conferencia')
plt.ylabel('Salario ($)')
plt.savefig('Practica4/boxplot_conference.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# RESUMEN ESTADÍSTICO
# =============================================================================
print("\n" + "="*80)
print("RESUMEN ESTADÍSTICO FINAL")
print("="*80)

print("\n1. POR POSICIÓN:")
for position in df['position'].unique():
    pos_data = df[df['position'] == position]['salary']
    print(f"   {position}: ${pos_data.mean():,.2f} ± ${pos_data.std():,.2f}")

print("\n2. POR AÑO:")
for year in sorted(df['season'].unique()):
    year_data = df[df['season'] == year]['salary']
    print(f"   {year}: ${year_data.mean():,.2f}")

print("\n3. POR CONFERENCIA:")
print(f"   Este: ${east_salaries.mean():,.2f}")
print(f"   Oeste: ${west_salaries.mean():,.2f}")

# Exportar resultados
results = {
    'Hipotesis': ['Posiciones', 'Años', 'Conferencias'],
    'Prueba_Utilizada': ['Kruskal-Wallis/ANOVA', 'Kruskal-Wallis/ANOVA', 'T-test'],
    'P_value': [p_val, p_val_years, p_value],
    'Diferencia_Significativa': [has_differences, has_differences_years, p_value < 0.05]
}

results_df = pd.DataFrame(results)
results_df.to_csv('Practica4/resultados_estadisticos.csv', index=False)
print(f"\nResultados exportados a: Practica4/resultados_estadisticos.csv")

print("\n" + "="*80)
print("¡ANÁLISIS ESTADÍSTICO COMPLETADO!")
print("="*80)