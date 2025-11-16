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

def test_stationarity(timeseries, window=3):
    """
    Test de estacionaridad (Dickey-Fuller aumentado)
    """
    print('Test de Estacionaridad:')
    print('=' * 50)
    
    # Verificar que la serie no esté vacía
    if len(timeseries) == 0:
        print("ERROR: La serie de tiempo está vacía")
        return False
    
    # Rolling statistics
    rolmean = timeseries.rolling(window=min(window, len(timeseries))).mean()
    rolstd = timeseries.rolling(window=min(window, len(timeseries))).std()
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(timeseries.index, timeseries.values, color='blue', label='Original', marker='o')
    plt.plot(rolmean.index, rolmean.values, color='red', label='Media Movil', marker='s')
    plt.plot(rolstd.index, rolstd.values, color='black', label='Desviacion Estandar', marker='^')
    plt.legend(loc='best')
    plt.title('Estadisticas de Rodamiento')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Dickey-Fuller test
    print('Resultados del Dickey-Fuller Test:')
    dftest = adfuller(timeseries.dropna(), autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
                       index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput[f'Critical Value ({key})'] = value
    print(dfoutput)
    
    if dfoutput['p-value'] < 0.05:
        print("La serie es ESTACIONARIA (p-value < 0.05)")
        return True
    else:
        print("La serie NO es estacionaria (p-value >= 0.05)")
        return False

def create_time_series_features(df, date_column, value_column):
    """
    Crear características para series de tiempo - VERSION CORREGIDA
    """
    # Crear dataframe con características temporales directamente
    ts_df = pd.DataFrame({
        'value': df[value_column].values,
        'year': df[date_column].values,
        'time_index': range(len(df))
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
    ts_df['rolling_mean_3'] = ts_df['value'].rolling(window=min(3, len(ts_df))).mean()
    ts_df['rolling_std_3'] = ts_df['value'].rolling(window=min(3, len(ts_df))).std()
    
    return ts_df

def forecast_evaluation(y_true, y_pred, model_name=""):
    """
    Evaluación completa de pronósticos
    """
    # Asegurar que no hay NaN en los datos
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        print(f"ERROR: No hay datos validos para evaluar {model_name}")
        return {'MSE': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan}
    
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
    
    print(f"METRICAS DE EVALUACION - {model_name}:")
    print(f"   MSE: {mse:,.2f}")
    print(f"   MAE: {mae:,.2f}") 
    print(f"   RMSE: {rmse:,.2f}")
    print(f"   R2: {r2:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

# CARGA Y PREPARACION DE DATOS
print("=" * 80)
print("PRACTICA 8: FORECASTING CON SERIES DE TIEMPO - NBA SALARIES")
print("=" * 80)

# Cargar datos
df = pd.read_csv("edited-salaries.csv")
print(f"Dataset shape: {df.shape}")
print(f"Rango de anos: {df['season'].min()} - {df['season'].max()}")

# FILTRAR SOLO LOS AÑOS 2000-2009
df = df[df['season'].between(2000, 2009)]
print(f"Dataset filtrado (2000-2009): {df.shape}")

# PROBLEMA 1: FORECASTING DEL SALARIO PROMEDIO ANUAL
print("\n" + "=" * 80)
print("PROBLEMA 1: FORECASTING DEL SALARIO PROMEDIO ANUAL")
print("=" * 80)

# Crear serie de tiempo del salario promedio anual
yearly_avg_salary = df.groupby('season')['salary'].mean().sort_index()
print("\nSalario promedio por ano:")
print(yearly_avg_salary)

# DataFrame para el modelo
salary_df = pd.DataFrame({
    'season': yearly_avg_salary.index,
    'salary': yearly_avg_salary.values
})

ts_df = create_time_series_features(salary_df, 'season', 'salary')

print(f"\nDataFrame de series de tiempo:")
print(ts_df)
print(f"Shape del DataFrame: {ts_df.shape}")

# Test de estacionaridad solo con datos no NaN
if len(ts_df['value'].dropna()) > 0:
    # Crear una serie temporal con años como índice para el test
    ts_series = pd.Series(ts_df['value'].values, index=ts_df['year'])
    is_stationary = test_stationarity(ts_series.dropna())
else:
    print("ERROR: No hay datos suficientes para el test de estacionaridad")
    is_stationary = False

# MODELO 1: REGRESION LINEAL SIMPLE (TENDENCIA)
print("\n" + "-" * 50)
print("MODELO 1: REGRESION LINEAL SIMPLE")
print("-" * 50)

# Usar solo las filas que tienen datos completos para el valor
valid_data = ts_df[['time_index', 'value', 'year']].dropna()

if len(valid_data) > 1:
    # Dividir en train (80%) y test (20%)
    train_size = int(len(valid_data) * 0.8)
    train_df = valid_data.iloc[:train_size]
    test_df = valid_data.iloc[train_size:]
    
    # Características para modelo simple
    X_train_simple = train_df[['time_index']]
    X_test_simple = test_df[['time_index']]
    y_train = train_df['value']
    y_test = test_df['value']
    years_test = test_df['year']
    
    # Entrenar modelo
    model_simple = LinearRegression()
    model_simple.fit(X_train_simple, y_train)
    
    # Predicciones
    y_pred_simple_train = model_simple.predict(X_train_simple)
    y_pred_simple_test = model_simple.predict(X_test_simple)
    
    # Evaluación
    metrics_simple = forecast_evaluation(y_test, y_pred_simple_test, "Regresion Lineal Simple")
else:
    print("No hay suficientes datos para el modelo simple")
    metrics_simple = {'RMSE': np.nan, 'R2': np.nan}

# MODELO 2: REGRESION CON CARACTERISTICAS TEMPORALES
print("\n" + "-" * 50)
print("MODELO 2: REGRESION CON MULTIPLES CARACTERISTICAS")
print("-" * 50)

# Usar características básicas que siempre tienen datos
simple_features = ['time_index', 'time_squared', 'time_cubed']

# Datos completos para estas características
advanced_data = ts_df[simple_features + ['value', 'year']].dropna()

if len(advanced_data) > 1:
    train_size_adv = int(len(advanced_data) * 0.8)
    train_df_adv = advanced_data.iloc[:train_size_adv]
    test_df_adv = advanced_data.iloc[train_size_adv:]
    
    X_train_advanced = train_df_adv[simple_features]
    X_test_advanced = test_df_adv[simple_features]
    y_train_adv = train_df_adv['value']
    y_test_adv = test_df_adv['value']
    years_test_adv = test_df_adv['year']
    
    # Entrenar modelo avanzado
    model_advanced = LinearRegression()
    model_advanced.fit(X_train_advanced, y_train_adv)
    
    # Predicciones
    y_pred_advanced_train = model_advanced.predict(X_train_advanced)
    y_pred_advanced_test = model_advanced.predict(X_test_advanced)
    
    # Evaluación
    metrics_advanced = forecast_evaluation(y_test_adv, y_pred_advanced_test, "Regresion con Caracteristicas Temporales")
else:
    print("No hay suficientes datos para el modelo avanzado")
    metrics_advanced = {'RMSE': np.nan, 'R2': np.nan}

# MODELO 3: STATSMODELS OLS (PARA ANALISIS DETALLADO)
print("\n" + "-" * 50)
print("MODELO 3: STATSMODELS OLS")
print("-" * 50)

# Preparar datos para statsmodels (solo datos completos)
ols_data = ts_df[['time_index', 'time_squared', 'value']].dropna()

if len(ols_data) > 0:
    X_sm = sm.add_constant(ols_data[['time_index', 'time_squared']])
    y_sm = ols_data['value']
    
    # Modelo OLS
    model_sm = sm.OLS(y_sm, X_sm).fit()
    print(model_sm.summary())
else:
    print("No hay datos suficientes para el modelo OLS")

# PREDICCIONES FUTURAS
print("\n" + "-" * 50)
print("PREDICCIONES FUTURAS")
print("-" * 50)

# Crear datos futuros
if len(ts_df) > 0 and not pd.isna(ts_df['year'].iloc[-1]):
    last_year = int(ts_df['year'].iloc[-1])
    future_years = list(range(last_year + 1, last_year + 6))  # Proximos 5 anos
    future_time_index = list(range(len(ts_df), len(ts_df) + len(future_years)))

    future_df = pd.DataFrame({
        'year': future_years,
        'time_index': future_time_index,
        'time_squared': [x**2 for x in future_time_index],
        'time_cubed': [x**3 for x in future_time_index]
    })

    # Hacer predicciones con el modelo que funcione
    if 'model_advanced' in locals() and len(advanced_data) > 0:
        future_predictions = model_advanced.predict(future_df[simple_features])
        future_df['predicted_salary'] = future_predictions
        
        print("Predicciones de salario promedio para anos futuros:")
        for _, row in future_df.iterrows():
            print(f"  {int(row['year'])}: ${row['predicted_salary']:,.2f}")
    elif 'model_simple' in locals() and len(valid_data) > 0:
        future_predictions = model_simple.predict(future_df[['time_index']])
        future_df['predicted_salary'] = future_predictions
        
        print("Predicciones de salario promedio para anos futuros:")
        for _, row in future_df.iterrows():
            print(f"  {int(row['year'])}: ${row['predicted_salary']:,.2f}")
    else:
        print("No se pudieron generar predicciones futuras")
else:
    print("No hay datos suficientes para generar predicciones futuras")
    future_df = pd.DataFrame()

# VISUALIZACIONES COMPLETAS
print("\n" + "-" * 50)
print("VISUALIZACIONES DE FORECASTING")
print("-" * 50)

# Crear visualizaciones
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Serie temporal original CON AÑOS CORRECTOS
axes[0, 0].plot(ts_df['year'], ts_df['value'], 
                'o-', linewidth=2, markersize=8, label='Datos Reales', color='blue')

# Solo graficar tendencia si el modelo se entreno
if 'model_simple' in locals() and len(valid_data) > 0:
    # Predicciones para todos los años históricos
    all_predictions_simple = model_simple.predict(ts_df[['time_index']])
    axes[0, 0].plot(ts_df['year'], all_predictions_simple, 
                    '--', linewidth=2, label='Tendencia Lineal', color='red')

axes[0, 0].set_xlabel('Año')
axes[0, 0].set_ylabel('Salario Promedio ($)')
axes[0, 0].set_title('Evolucion del Salario Promedio NBA (2000-2009)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(ts_df['year'])  # Mostrar todos los años en el eje X

# 2. Train vs Test predictions (si hay datos)
if 'y_test' in locals() and len(y_test) > 0:
    axes[0, 1].plot(train_df['year'], y_train, 'o-', label='Train Real', color='blue', markersize=6)
    axes[0, 1].plot(train_df['year'], y_pred_simple_train, '--', label='Train Pred', color='lightblue')
    axes[0, 1].plot(test_df['year'], y_test, 's-', label='Test Real', color='green', markersize=6)
    axes[0, 1].plot(test_df['year'], y_pred_simple_test, '--', label='Test Pred', color='lightgreen')
    axes[0, 1].set_xlabel('Año')
    axes[0, 1].set_ylabel('Salario Promedio ($)')
    axes[0, 1].set_title('Predicciones: Train vs Test')
    axes[0, 1].legend()
    axes[0, 1].set_xticks(test_df['year'])  # Mostrar años del test
else:
    axes[0, 1].text(0.5, 0.5, 'No hay suficientes datos\npara division train/test', 
                   ha='center', va='center', transform=axes[0, 1].transAxes)
    axes[0, 1].set_title('Predicciones: Train vs Test')

axes[0, 1].grid(True, alpha=0.3)

# 3. Predicciones futuras (si existen)
if 'future_predictions' in locals() and len(future_df) > 0:
    axes[1, 0].plot(ts_df['year'], ts_df['value'], 
                    'o-', linewidth=2, label='Datos Historicos', color='blue', markersize=6)
    axes[1, 0].plot(future_df['year'], future_df['predicted_salary'], 
                    's--', linewidth=2, label='Predicciones Futuras', color='red', markersize=6)
    axes[1, 0].axvline(x=last_year, color='gray', linestyle=':', alpha=0.7, label='Fin de Datos')
    axes[1, 0].set_xlabel('Año')
    axes[1, 0].set_ylabel('Salario Promedio ($)')
    axes[1, 0].set_title('Predicciones Futuras de Salarios NBA')
    axes[1, 0].legend()
    # Combinar años históricos y futuros en el eje X
    all_years = list(ts_df['year']) + list(future_df['year'])
    axes[1, 0].set_xticks(all_years)
else:
    axes[1, 0].text(0.5, 0.5, 'No se pudieron generar\npredicciones futuras', 
                   ha='center', va='center', transform=axes[1, 0].transAxes)
    axes[1, 0].set_title('Predicciones Futuras')

axes[1, 0].grid(True, alpha=0.3)

# 4. Comparacion de modelos (si ambos modelos funcionaron)
if (not np.isnan(metrics_simple.get('RMSE', np.nan)) and 
    not np.isnan(metrics_advanced.get('RMSE', np.nan))):
    
    models = ['Lineal Simple', 'Con Features']
    rmse_values = [metrics_simple['RMSE'], metrics_advanced['RMSE']]
    r2_values = [metrics_simple['R2'], metrics_advanced['R2']]

    x_pos = np.arange(len(models))
    width = 0.35

    bars1 = axes[1, 1].bar(x_pos - width/2, rmse_values, width, label='RMSE', alpha=0.7)
    bars2 = axes[1, 1].bar(x_pos + width/2, r2_values, width, label='R2', alpha=0.7)

    axes[1, 1].set_xlabel('Modelo')
    axes[1, 1].set_ylabel('Metrica')
    axes[1, 1].set_title('Comparacion de Modelos')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(models)
    axes[1, 1].legend()
    
    # Anadir valores en las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom')
else:
    axes[1, 1].text(0.5, 0.5, 'No hay suficientes datos\npara comparar modelos', 
                   ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Comparacion de Modelos')

axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('forecasting_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ANALISIS DE TENDENCIAS POR POSICION
print("\n" + "=" * 80)
print("ANALISIS DE TENDENCIAS POR POSICION")
print("=" * 80)

# Salario promedio por posicion y ano (solo 2000-2009)
position_yearly = df.groupby(['season', 'position'])['salary'].mean().unstack()

print("\nSalarios promedio por posicion (2000-2009):")
print(position_yearly)

# Calcular crecimiento por posicion
growth_rates = {}
for position in position_yearly.columns:
    position_data = position_yearly[position].dropna()
    if len(position_data) > 1:
        start_val = position_data.iloc[0]
        end_val = position_data.iloc[-1]
        growth = ((end_val - start_val) / start_val) * 100
        growth_rates[position] = growth

if growth_rates:
    print("\nCrecimiento salarial por posicion (2000-2009):")
    for position, growth in sorted(growth_rates.items(), key=lambda x: x[1], reverse=True):
        print(f"  {position}: {growth:+.1f}%")
else:
    print("\nNo hay datos suficientes para calcular crecimiento por posicion")

# Visualizacion de tendencias por posicion
plt.figure(figsize=(14, 8))
for position in position_yearly.columns:
    plt.plot(position_yearly.index, position_yearly[position], 
             marker='o', linewidth=2, label=position, markersize=6)

plt.xlabel('Año')
plt.ylabel('Salario Promedio ($)')
plt.title('Evolucion del Salario Promedio por Posicion (2000-2009)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(position_yearly.index)  # Mostrar todos los años
plt.tight_layout()
plt.savefig('salary_trends_by_position.png', dpi=300, bbox_inches='tight')
plt.show()

# RESULTADOS FINALES
print("\n" + "=" * 80)
print("RESUMEN FINAL")
print("=" * 80)

print(f"\nPeriodo analizado: {df['season'].min()} - {df['season'].max()}")
print(f"Total de anos analizados: {len(yearly_avg_salary)}")
print(f"Rango salarial: ${yearly_avg_salary.min():,.2f} - ${yearly_avg_salary.max():,.2f}")

if 'future_predictions' in locals() and len(future_df) > 0:
    print(f"\nPredicciones para proximos anos:")
    for _, row in future_df.iterrows():
        print(f"  {int(row['year'])}: ${row['predicted_salary']:,.2f}")

print("\n" + "=" * 80)
print("ANALISIS DE FORECASTING COMPLETADO!")
print("=" * 80)