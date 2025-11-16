import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)

def test_stationarity(timeseries, window=12):
    """
    Test de estacionaridad (Dickey-Fuller aumentado)
    """
    print('Test de Estacionaridad:')
    print('=' * 50)
    
    # Rolling statistics
    rolmean = timeseries.rolling(window=window).mean()
    rolstd = timeseries.rolling(window=window).std()
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Media Móvil')
    plt.plot(rolstd, color='black', label='Desviación Estándar')
    plt.legend(loc='best')
    plt.title('Estadísticas de Rodamiento')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Dickey-Fuller test
    print('Resultados del Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
                       index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput[f'Critical Value ({key})'] = value
    print(dfoutput)
    
    if dfoutput['p-value'] < 0.05:
        print("La serie es ESTACIONARIA (p-value < 0.05)")
        return True
    else:
        print("La serie NO es estacionaria (p-value ≥ 0.05)")
        return False

def create_time_series_features(df, date_column, value_column, freq='Y'):
    """
    Crear características para series de tiempo
    """
    # Convertir a serie de tiempo
    ts = df.set_index(date_column)[value_column]
    ts = ts.asfreq(freq)
    
    # Crear dataframe con características temporales
    ts_df = pd.DataFrame({
        'value': ts,
        'year': ts.index.year,
        'time_index': range(len(ts))
    })
    
    # Tendencia polinomial
    ts_df['time_squared'] = ts_df['time_index'] ** 2
    ts_df['time_cubed'] = ts_df['time_index'] ** 3
    
    # Diferencias para estacionaridad
    ts_df['value_diff'] = ts_df['value'].diff()
    ts_df['value_diff2'] = ts_df['value_diff'].diff()
    
    # Lags
    ts_df['value_lag1'] = ts_df['value'].shift(1)
    ts_df['value_lag2'] = ts_df['value'].shift(2)
    
    # Rolling statistics
    ts_df['rolling_mean_3'] = ts_df['value'].rolling(window=3).mean()
    ts_df['rolling_std_3'] = ts_df['value'].rolling(window=3).std()
    
    return ts_df.dropna()

def forecast_evaluation(y_true, y_pred, model_name=""):
    """
    Evaluación completa de pronósticos
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"MÉTRICAS DE EVALUACIÓN - {model_name}:")
    print(f"   MSE: {mse:,.2f}")
    print(f"   MAE: {mae:,.2f}") 
    print(f"   RMSE: {rmse:,.2f}")
    print(f"   R²: {r2:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

# CARGA Y PREPARACIÓN DE DATOS
print("=" * 80)
print("PRÁCTICA 8: FORECASTING CON SERIES DE TIEMPO - NBA SALARIES")
print("=" * 80)

# Cargar datos
df = pd.read_csv("edited-salaries.csv")
print(f"Dataset shape: {df.shape}")
print(f"Rango de años: {df['season'].min()} - {df['season'].max()}")

# PROBLEMA 1: FORECASTING DEL SALARIO PROMEDIO ANUAL
print("\n" + "=" * 80)
print("PROBLEMA 1: FORECASTING DEL SALARIO PROMEDIO ANUAL")
print("=" * 80)

# Crear serie de tiempo del salario promedio anual
yearly_avg_salary = df.groupby('season')['salary'].mean().sort_index()
print("\nSalario promedio por año:")
print(yearly_avg_salary)

# DataFrame para el modelo
ts_df = create_time_series_features(
    pd.DataFrame({'season': yearly_avg_salary.index, 'salary': yearly_avg_salary.values}),
    'season', 'salary', freq='Y'
)

print(f"\nDataFrame de series de tiempo:")
print(ts_df.head())

# Test de estacionaridad
is_stationary = test_stationarity(ts_df['value'])

# MODELO 1: REGRESIÓN LINEAL SIMPLE (TENDENCIA)
print("\n" + "-" * 50)
print("MODELO 1: REGRESIÓN LINEAL SIMPLE")
print("-" * 50)

# Dividir en train (80%) y test (20%)
train_size = int(len(ts_df) * 0.8)
train_df = ts_df.iloc[:train_size]
test_df = ts_df.iloc[train_size:]

# Características para modelo simple
X_train_simple = train_df[['time_index']]
X_test_simple = test_df[['time_index']]
y_train = train_df['value']
y_test = test_df['value']

# Entrenar modelo
model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train)

# Predicciones
y_pred_simple_train = model_simple.predict(X_train_simple)
y_pred_simple_test = model_simple.predict(X_test_simple)

# Evaluación
metrics_simple = forecast_evaluation(y_test, y_pred_simple_test, "Regresión Lineal Simple")

# MODELO 2: REGRESIÓN CON CARACTERÍSTICAS TEMPORALES
print("\n" + "-" * 50)
print("MODELO 2: REGRESIÓN CON MÚLTIPLES CARACTERÍSTICAS")
print("-" * 50)

# Características adicionales
features = ['time_index', 'time_squared', 'time_cubed', 'value_lag1', 'value_lag2', 
           'rolling_mean_3', 'rolling_std_3']

X_train_advanced = train_df[features].dropna()
X_test_advanced = test_df[features].dropna()

# Ajustar índices para que coincidan
y_train_adv = y_train.loc[X_train_advanced.index]
y_test_adv = y_test.loc[X_test_advanced.index]

# Entrenar modelo avanzado
model_advanced = LinearRegression()
model_advanced.fit(X_train_advanced, y_train_adv)

# Predicciones
y_pred_advanced_train = model_advanced.predict(X_train_advanced)
y_pred_advanced_test = model_advanced.predict(X_test_advanced)

# Evaluación
metrics_advanced = forecast_evaluation(y_test_adv, y_pred_advanced_test, "Regresión Avanzada")

# MODELO 3: STATSMODELS OLS (PARA ANÁLISIS DETALLADO)
print("\n" + "-" * 50)
print("MODELO 3: STATSMODELS OLS")
print("-" * 50)

# Preparar datos para statsmodels
X_sm = sm.add_constant(ts_df[['time_index', 'time_squared']].dropna())
y_sm = ts_df['value'].dropna()

# Modelo OLS
model_sm = sm.OLS(y_sm, X_sm).fit()
print(model_sm.summary())

# PREDICCIONES FUTURAS
print("\n" + "-" * 50)
print("PREDICCIONES FUTURAS (2010-2015)")
print("-" * 50)

# Crear datos futuros
future_years = list(range(2010, 2016))
future_time_index = list(range(len(ts_df), len(ts_df) + len(future_years)))

future_df = pd.DataFrame({
    'year': future_years,
    'time_index': future_time_index,
    'time_squared': [x**2 for x in future_time_index],
    'time_cubed': [x**3 for x in future_time_index]
})

# Usar el mejor modelo para predicciones futuras
future_X = future_df[['time_index', 'time_squared', 'time_cubed']]

# Para características que requieren datos históricos, usar los últimos disponibles
last_value = ts_df['value'].iloc[-1]
last_rolling_mean = ts_df['rolling_mean_3'].iloc[-1]
last_rolling_std = ts_df['rolling_std_3'].iloc[-1]

future_df['value_lag1'] = last_value
future_df['value_lag2'] = ts_df['value'].iloc[-2] if len(ts_df) > 1 else last_value
future_df['rolling_mean_3'] = last_rolling_mean
future_df['rolling_std_3'] = last_rolling_std

# Predicciones con modelo avanzado
future_predictions = model_advanced.predict(future_df[features])

future_df['predicted_salary'] = future_predictions

print("Predicciones de salario promedio para años futuros:")
print(future_df[['year', 'predicted_salary']].to_string(index=False))

# VISUALIZACIONES COMPLETAS
print("\n" + "-" * 50)
print("VISUALIZACIONES DE FORECASTING")
print("-" * 50)

# Crear visualizaciones completas
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Serie temporal original con tendencia
axes[0, 0].plot(yearly_avg_salary.index, yearly_avg_salary.values, 
                'o-', linewidth=2, markersize=6, label='Real', color='blue')

# Tendencia del modelo simple
all_time_index = ts_df['time_index']
all_predictions_simple = model_simple.predict(ts_df[['time_index']])
axes[0, 0].plot(yearly_avg_salary.index, all_predictions_simple, 
                '--', linewidth=2, label='Tendencia', color='red')

axes[0, 0].set_xlabel('Año')
axes[0, 0].set_ylabel('Salario Promedio ($)')
axes[0, 0].set_title('Evolución del Salario Promedio y Tendencia')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Train vs Test predictions
axes[0, 1].plot(train_df.index, y_train, 'o-', label='Train Real', color='blue')
axes[0, 1].plot(train_df.index, y_pred_simple_train, '--', label='Train Pred', color='lightblue')
axes[0, 1].plot(test_df.index, y_test, 's-', label='Test Real', color='green')
axes[0, 1].plot(test_df.index, y_pred_simple_test, '--', label='Test Pred', color='lightgreen')
axes[0, 1].set_xlabel('Año')
axes[0, 1].set_ylabel('Salario Promedio ($)')
axes[0, 1].set_title('Predicciones Train vs Test')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Predicciones futuras
all_years = list(yearly_avg_salary.index) + future_years
all_values = list(yearly_avg_salary.values) + list(future_predictions)

axes[1, 0].plot(yearly_avg_salary.index, yearly_avg_salary.values, 
                'o-', linewidth=2, label='Histórico', color='blue')
axes[1, 0].plot(future_df['year'], future_predictions, 
                's--', linewidth=2, label='Predicciones', color='red')
axes[1, 0].axvline(x=2009, color='gray', linestyle=':', alpha=0.7, label='Inicio Predicciones')
axes[1, 0].set_xlabel('Año')
axes[1, 0].set_ylabel('Salario Promedio ($)')
axes[1, 0].set_title('Predicciones Futuras 2010-2015')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Comparación de modelos
models = ['Simple', 'Avanzado']
rmse_values = [metrics_simple['RMSE'], metrics_advanced['RMSE']]
r2_values = [metrics_simple['R2'], metrics_advanced['R2']]

x_pos = np.arange(len(models))
width = 0.35

bars1 = axes[1, 1].bar(x_pos - width/2, rmse_values, width, label='RMSE', alpha=0.7)
bars2 = axes[1, 1].bar(x_pos + width/2, r2_values, width, label='R²', alpha=0.7)

axes[1, 1].set_xlabel('Modelo')
axes[1, 1].set_ylabel('Métrica')
axes[1, 1].set_title('Comparación de Modelos')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(models)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Añadir valores en las barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('Practica8/forecasting_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ANÁLISIS DE TENDENCIAS POR POSICIÓN
print("\n" + "=" * 80)
print("ANÁLISIS DE TENDENCIAS POR POSICIÓN")
print("=" * 80)

# Salario promedio por posición y año
position_yearly = df.groupby(['season', 'position'])['salary'].mean().unstack()

# Calcular crecimiento por posición
growth_rates = {}
for position in position_yearly.columns:
    if len(position_yearly[position].dropna()) > 1:
        start_val = position_yearly[position].dropna().iloc[0]
        end_val = position_yearly[position].dropna().iloc[-1]
        growth = ((end_val - start_val) / start_val) * 100
        growth_rates[position] = growth

print("\nCrecimiento salarial por posición (2000-2009):")
for position, growth in sorted(growth_rates.items(), key=lambda x: x[1], reverse=True):
    print(f"  {position}: {growth:+.1f}%")

# Visualización de tendencias por posición
plt.figure(figsize=(14, 8))
for position in position_yearly.columns:
    plt.plot(position_yearly.index, position_yearly[position], 
             marker='o', linewidth=2, label=position)

plt.xlabel('Año')
plt.ylabel('Salario Promedio ($)')
plt.title('Evolución del Salario Promedio por Posición (2000-2009)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Practica8/salary_trends_by_position.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# MODELO FINAL Y EXPORTACIÓN
# =============================================================================
print("\n" + "=" * 80)
print("MODELO FINAL Y EXPORTACIÓN")
print("=" * 80)

# Usar el mejor modelo (basado en métricas)
if metrics_advanced['RMSE'] < metrics_simple['RMSE']:
    best_model = model_advanced
    best_features = features
    print("MEJOR MODELO: Regresión Avanzada")
else:
    best_model = model_simple
    best_features = ['time_index']
    print("MEJOR MODELO: Regresión Lineal Simple")

# Entrenar modelo final con todos los datos
X_final = ts_df[best_features].dropna()
y_final = ts_df['value'].dropna()

final_model = LinearRegression()
final_model.fit(X_final, y_final)

# Predicciones finales incluyendo futuro
all_predictions = final_model.predict(X_final)
future_predictions_final = final_model.predict(future_df[best_features])

# Crear dataframe de resultados
results_df = pd.DataFrame({
    'year': list(ts_df.index) + list(future_df['year']),
    'actual_salary': list(ts_df['value']) + [None] * len(future_df),
    'predicted_salary': list(all_predictions) + list(future_predictions_final),
    'is_forecast': [False] * len(ts_df) + [True] * len(future_df)
})

print("\nResultados completos:")
print(results_df.to_string(index=False))

# Exportar resultados
results_df.to_csv('Practica8/forecasting_results.csv', index=False)

# Resumen de modelos
model_comparison = pd.DataFrame({
    'Modelo': ['Regresión Lineal Simple', 'Regresión Avanzada'],
    'RMSE': [metrics_simple['RMSE'], metrics_advanced['RMSE']],
    'R²': [metrics_simple['R2'], metrics_advanced['R2']],
    'MAE': [metrics_simple['MAE'], metrics_advanced['MAE']],
    'MAPE': [metrics_simple['MAPE'], metrics_advanced['MAPE']]
})

model_comparison.to_csv('Practica8/model_comparison.csv', index=False)

print("\n ARCHIVOS EXPORTADOS:")
print("  Practica8/forecasting_results.csv")
print("  Practica8/model_comparison.csv")
print("  Practica8/forecasting_analysis.png")
print("  Practica8/salary_trends_by_position.png")

print("\n" + "=" * 80)
print("¡FORECASTING CON SERIES DE TIEMPO COMPLETADO!")
print("=" * 80)