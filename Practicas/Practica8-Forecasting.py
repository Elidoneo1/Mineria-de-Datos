import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cargar datos
df = pd.read_csv("nba_processed.csv")

# Agrupar por temporada para obtener la serie de tiempo
# Calculamos el promedio y el máximo salario por año
yearly_stats = df.groupby('season')['salary'].agg(['mean', 'max']).reset_index()

print("="*60)
print("FORECASTING: Tendencia Salarial (2000-2020 + Predicción)")
print("="*60)

# Preparar datos para regresión
X = yearly_stats[['season']] # Años
y_mean = yearly_stats['mean'] # Salario Promedio
y_max = yearly_stats['max']   # Salario Máximo (Superestrellas)

# Entrenar modelos separados
model_mean = LinearRegression().fit(X, y_mean)
model_max = LinearRegression().fit(X, y_max)

# Crear años futuros para predicción
future_years = pd.DataFrame({'season': [2021, 2022, 2023, 2024, 2025]})
pred_mean = model_mean.predict(future_years)
pred_max = model_max.predict(future_years)

# Visualización
plt.figure(figsize=(12, 6))

# Datos históricos
plt.plot(yearly_stats['season'], yearly_stats['mean'], 'o-', label='Promedio Histórico', color='blue')
plt.plot(yearly_stats['season'], yearly_stats['max'], 'o-', label='Máximo Histórico', color='green')

# Predicciones
plt.plot(future_years['season'], pred_mean, 'x--', label='Predicción Promedio', color='cyan')
plt.plot(future_years['season'], pred_max, 'x--', label='Predicción Máximo', color='lime')

plt.title('Forecasting de Salarios NBA (Histórico vs Futuro)')
plt.xlabel('Temporada')
plt.ylabel('Salario ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('Practica8/full_forecasting.png')
plt.show()

print("\nPredicciones de Salario Máximo (Superestrellas):")
for year, salary in zip(future_years['season'], pred_max):
    print(f"Año {year}: ${salary:,.2f}")