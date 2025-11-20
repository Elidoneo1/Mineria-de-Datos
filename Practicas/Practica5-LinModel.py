import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# Cargar datos procesados
df = pd.read_csv("nba_processed.csv")

# --- MODELO MEJORADO: Log-Linear Regression ---
# Queremos predecir el salario basándonos en el Año y la Posición
# Usamos logaritmo del salario porque los salarios crecen exponencialmente

X = df[['season', 'position_encoded']]
y = np.log(df['salary'])  # Transformación Logarítmica

# División Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar
model = LinearRegression()
model.fit(X_train, y_train)

# Predecir
y_pred_log = model.predict(X_test)
y_pred_original_scale = np.exp(y_pred_log) # Revertir logaritmo para ver $$$ reales
y_test_original_scale = np.exp(y_test)

# Métricas
r2 = r2_score(y_test, y_pred_log)
print("="*50)
print(f"R² Score (Modelo Logarítmico): {r2:.4f}")
print("="*50)
print("Interpretación: Un R² más alto indica que el modelo entiende mejor la curva de crecimiento salarial.")

# Visualización
plt.figure(figsize=(10, 6))
plt.scatter(X_test['season'], y_test_original_scale, alpha=0.3, label='Datos Reales')
plt.scatter(X_test['season'], y_pred_original_scale, color='red', alpha=0.3, label='Predicción')
plt.title('Predicción de Salarios (Escala Logarítmica Revertida)')
plt.xlabel('Temporada')
plt.ylabel('Salario ($)')
plt.legend()
plt.savefig('Practica5/improved_linear_model.png')
plt.show()