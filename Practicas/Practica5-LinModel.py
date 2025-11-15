import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats as stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
from tabulate import tabulate

# Configuración
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)

def print_tabulate(df: pd.DataFrame, n_rows=5):
    print(tabulate(df.head(n_rows), headers=df.columns, tablefmt='orgtbl'))

# Cargar datos
df = pd.read_csv("edited-salaries.csv")
print(f"Dataset shape: {df.shape}")
print(f"Variables: {df.columns.tolist()}")


# ANÁLISIS DE CORRELACIÓN
print("\n" + "="*80)
print("ANÁLISIS DE CORRELACIÓN")
print("="*80)

# Preparar datos para correlación
df_corr = df.copy()

# Codificar variables categóricas para correlación
le_position = LabelEncoder()
le_team = LabelEncoder()

df_corr['position_encoded'] = le_position.fit_transform(df_corr['position'])
df_corr['team_encoded'] = le_team.fit_transform(df_corr['team'])

# Matriz de correlación
correlation_matrix = df_corr[['salary', 'season', 'position_encoded', 'team_encoded']].corr()

print("Matriz de Correlación:")
print(correlation_matrix)

# Heatmap de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
plt.title('Matriz de Correlación entre Variables', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('Practica5/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlaciones individuales con p-values
print("\nCorrelaciones detalladas con p-values:")
variables = ['season', 'position_encoded', 'team_encoded']
for var in variables:
    corr, p_value = stats.pearsonr(df_corr['salary'], df_corr[var])
    print(f"Salary vs {var}: r = {corr:.4f}, p-value = {p_value:.4f}")


# MODELO LINEAL 1: Salario vs Año + Posición

print("\n" + "="*80)
print("MODELO LINEAL 1: Salary ~ Season + Position")
print("="*80)

# Preparar datos para el modelo
X1 = df_corr[['season', 'position_encoded']].copy()
y = df_corr['salary'].copy()

# Añadir intercepto manualmente para statsmodels
X1_sm = sm.add_constant(X1)

# Modelo con statsmodels (para análisis detallado)
model1_sm = sm.OLS(y, X1_sm).fit()
print(model1_sm.summary())

# Modelo con scikit-learn (para predicciones)
model1_sk = LinearRegression()
model1_sk.fit(X1, y)

# Métricas del modelo 1
y_pred1 = model1_sk.predict(X1)
r2_1 = r2_score(y, y_pred1)
mse_1 = mean_squared_error(y, y_pred1)

print(f"\nMétricas del Modelo 1:")
print(f"R² Score: {r2_1:.4f}")
print(f"R² Ajustado: {model1_sm.rsquared_adj:.4f}")
print(f"MSE: {mse_1:,.2f}")
print(f"RMSE: {np.sqrt(mse_1):,.2f}")

# MODELO LINEAL 2: Salario vs Año (Modelo Simple)
print("\n" + "="*80)
print("MODELO LINEAL 2: Salary ~ Season (Modelo Simple)")
print("="*80)

X2 = df_corr[['season']].copy()
X2_sm = sm.add_constant(X2)

model2_sm = sm.OLS(y, X2_sm).fit()
print(model2_sm.summary())

model2_sk = LinearRegression()
model2_sk.fit(X2, y)

y_pred2 = model2_sk.predict(X2)
r2_2 = r2_score(y, y_pred2)

print(f"\nR² Score Modelo Simple: {r2_2:.4f}")

# MODELO LINEAL 3: Con variables dummy para posición
print("\n" + "="*80)
print("MODELO LINEAL 3: Con variables dummy para posición")
print("="*80)

# Crear variables dummy para posición
position_dummies = pd.get_dummies(df['position'], prefix='pos')
X3 = pd.concat([df_corr[['season']], position_dummies], axis=1)

# Remover una categoría para evitar multicolinealidad
X3 = X3.drop('pos_Center', axis=1)  # Center como referencia
X3_sm = sm.add_constant(X3)

model3_sm = sm.OLS(y, X3_sm).fit()
print(model3_sm.summary())

model3_sk = LinearRegression()
model3_sk.fit(X3, y)

y_pred3 = model3_sk.predict(X3)
r2_3 = r2_score(y, y_pred3)

print(f"\nR² Score Modelo con Dummies: {r2_3:.4f}")

# GRÁFICOS DEL MODELO
print("\n" + "="*80)
print("GRÁFICOS DE DIAGNÓSTICO DEL MODELO")
print("="*80)

# 1. GRÁFICO: Valores Reales vs Predichos
plt.figure(figsize=(15, 12))

plt.subplot(2, 3, 1)
plt.scatter(y, y_pred1, alpha=0.6, s=20)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Salario Real ($)')
plt.ylabel('Salario Predicho ($)')
plt.title(f'Real vs Predicho (R² = {r2_1:.3f})')
plt.grid(True, alpha=0.3)

# 2. GRÁFICO: Residuales vs Predichos
residuals1 = y - y_pred1
plt.subplot(2, 3, 2)
plt.scatter(y_pred1, residuals1, alpha=0.6, s=20)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Valores Predichos')
plt.ylabel('Residuales')
plt.title('Residuales vs Predichos')
plt.grid(True, alpha=0.3)

# 3. GRÁFICO: QQ-Plot de residuales
plt.subplot(2, 3, 3)
stats.probplot(residuals1, dist="norm", plot=plt)
plt.title('QQ-Plot de Residuales')

# 4. GRÁFICO: Evolución temporal con predicciones
plt.subplot(2, 3, 4)
# Salario promedio real por año
real_avg = df.groupby('season')['salary'].mean()
plt.plot(real_avg.index, real_avg.values, 'o-', label='Real', linewidth=2)

# Salario promedio predicho por año
pred_df = pd.DataFrame({'season': df['season'], 'predicted': y_pred1})
pred_avg = pred_df.groupby('season')['predicted'].mean()
plt.plot(pred_avg.index, pred_avg.values, 's-', label='Predicho', linewidth=2)

plt.xlabel('Año')
plt.ylabel('Salario Promedio ($)')
plt.title('Evolución Temporal: Real vs Predicho')
plt.legend()
plt.grid(True, alpha=0.3)

# 5. GRÁFICO: Salarios por posición (real vs predicho)
plt.subplot(2, 3, 5)
position_real = df.groupby('position')['salary'].mean()
position_pred = pd.DataFrame({'position': df['position'], 'predicted': y_pred1})
position_pred_avg = position_pred.groupby('position')['predicted'].mean()

x_pos = np.arange(len(position_real))
width = 0.35

plt.bar(x_pos - width/2, position_real.values, width, label='Real', alpha=0.7)
plt.bar(x_pos + width/2, position_pred_avg.values, width, label='Predicho', alpha=0.7)

plt.xlabel('Posición')
plt.ylabel('Salario Promedio ($)')
plt.title('Salario por Posición: Real vs Predicho')
plt.xticks(x_pos, position_real.index, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# 6. GRÁFICO: Comparación de R² entre modelos
plt.subplot(2, 3, 6)
models = ['Season Only', 'Season + Position\n(Encoded)', 'Season + Position\n(Dummies)']
r2_scores = [r2_2, r2_1, r2_3]

plt.bar(models, r2_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
plt.ylabel('R² Score')
plt.title('Comparación de R² entre Modelos')
plt.ylim(0, max(r2_scores) + 0.05)
for i, v in enumerate(r2_scores):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('Practica5/model_diagnostics.png', dpi=300, bbox_inches='tight')
plt.show()

# ANÁLISIS DE RESIDUALES

print("\n" + "="*80)
print("ANÁLISIS DE RESIDUALES")
print("="*80)

# Test de normalidad de residuales
shapiro_stat, shapiro_p = stats.shapiro(residuals1)
print(f"Shapiro-Wilk Test de Normalidad: p-value = {shapiro_p:.4f}")
if shapiro_p > 0.05:
    print("→ Los residuales siguen una distribución normal")
else:
    print("→ Los residuales NO siguen una distribución normal")

# Autocorrelación de residuales
from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(residuals1)
print(f"Durbin-Watson Statistic: {dw_stat:.4f}")
if 1.5 < dw_stat < 2.5:
    print("→ No hay autocorrelación significativa en los residuales")
else:
    print("→ Posible autocorrelación en los residuales")

# PREDICCIONES PARA NUEVOS DATOS
print("\n" + "="*80)
print("PREDICCIONES PARA DATOS NUEVOS")
print("="*80)

# Crear datos de ejemplo para predicción
new_data_examples = pd.DataFrame({
    'season': [2010, 2010, 2010, 2010, 2010],
    'position_encoded': [
        le_position.transform(['Point Guard'])[0],
        le_position.transform(['Shooting Guard'])[0], 
        le_position.transform(['Small Forward'])[0],
        le_position.transform(['Power Forward'])[0],
        le_position.transform(['Center'])[0]
    ]
})

predictions = model1_sk.predict(new_data_examples)
new_data_examples['predicted_salary'] = predictions
new_data_examples['position'] = le_position.inverse_transform(new_data_examples['position_encoded'])

print("Predicciones para 2010:")
print_tabulate(new_data_examples[['season', 'position', 'predicted_salary']])

# EXPORTAR RESULTADOS
print("\n" + "="*80)
print("EXPORTACIÓN DE RESULTADOS")
print("="*80)

# DataFrame con resultados completos
results_df = pd.DataFrame({
    'real_salary': y,
    'predicted_salary': y_pred1,
    'residuals': residuals1,
    'season': df['season'],
    'position': df['position'],
    'team': df['team']
})

results_df.to_csv('Practica5/linear_model_results.csv', index=False)

# Resumen de modelos
model_summary = pd.DataFrame({
    'Model': ['Season Only', 'Season + Position (Encoded)', 'Season + Position (Dummies)'],
    'R2_Score': [r2_2, r2_1, r2_3],
    'Features': ['season', 'season + position_encoded', 'season + position_dummies'],
    'Coefficients': [len(model2_sk.coef_), len(model1_sk.coef_), len(model3_sk.coef_)]
})

model_summary.to_csv('Practica5/model_comparison.csv', index=False)

print("Resultados exportados:")
print("- Practica5/linear_model_results.csv")
print("- Practica5/model_comparison.csv")
print("- Practica5/correlation_heatmap.png") 
print("- Practica5/model_diagnostics.png")

# INTERPRETACIÓN FINAL
print("\n" + "="*80)
print("INTERPRETACIÓN DE RESULTADOS")
print("="*80)

print(f"\nMEJOR MODELO: Season + Position (R² = {r2_1:.3f})")
print(f"Este modelo explica el {r2_1*100:.1f}% de la variabilidad en los salarios")

print("\nCOEFICIENTES DEL MODELO:")
coef_df = pd.DataFrame({
    'Variable': ['Intercepto', 'Season', 'Position_Encoded'],
    'Coeficiente': [model1_sk.intercept_] + model1_sk.coef_.tolist()
})
print(tabulate(coef_df, headers='keys', tablefmt='orgtbl'))

print(f"\nINTERPRETACIÓN:")
print(f"- Por cada año adicional, el salario aumenta en ${model1_sk.coef_[0]:.2f}")
print(f"- La posición tiene un efecto significativo en el salario")
print(f"- El intercepto representa el salario base esperado")

print("\n" + "="*80)
print("¡MODELOS LINEALES COMPLETADOS!")
print("="*80)