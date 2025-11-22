import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import shap

# Configuración visual
plt.style.use('seaborn-v0_8')

# Crear carpeta si no existe
if not os.path.exists('Practica6'):
    os.makedirs('Practica6')

# Cargar datos
try:
    df = pd.read_csv("nba_processed.csv")
except FileNotFoundError:
    print("Error: No se encuentra nba_processed.csv. Ejecuta la Práctica 1 primero.")
    exit()

print("="*60)
print("PROBLEMA: Predecir si un jugador es 'High Earner' (Top 25% de salarios)")
print("Optimización con GridSearch + Balanceo de Clases")
print("="*60)

# Features: Año, Equipo, Posición
X = df[['season', 'team_encoded', 'position_encoded']]
y = df['high_earner']

# División
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Iniciando búsqueda de mejores parámetros (GridSearch)...")

# 1. Definir la "rejilla"
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# 2. Configurar el modelo base CON BALANCEO
# Importante: class_weight='balanced' ayuda a que el modelo preste atención a la clase minoritaria
rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')

# 3. Configurar GridSearch
grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, n_jobs=-1, scoring='f1_weighted', verbose=1)

# 4. Entrenar
grid_search.fit(X_train, y_train)

# 5. Obtener el mejor modelo
best_clf = grid_search.best_estimator_

print("\n¡Mejores parámetros encontrados!:")
print(grid_search.best_params_)

# Predicción
y_pred = best_clf.predict(X_test)

# Resultados
print("\nAccuracy del mejor modelo:", accuracy_score(y_test, y_pred))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))

# Importancia de variables (Feature Importance nativo)
feature_imp = pd.Series(best_clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nVariables más importantes (Random Forest nativo):")
print(feature_imp)

# Matriz de Confusión
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión (Modelo Optimizado)')
plt.ylabel('Realidad')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('Practica6/rf_confusion_matrix.png')
plt.show()

print("\nGenerando análisis SHAP...")

try:
    # Crear el explicador
    explainer = shap.TreeExplainer(best_clf)
    
    # Calcular valores SHAP (check_additivity=False evita errores de redondeo)
    shap_values = explainer.shap_values(X_test, check_additivity=False)
    
    # Lógica robusta para seleccionar la clase positiva (1)
    vals_to_plot = None
    
    if isinstance(shap_values, list):
        # Versión clásica: Lista de arrays [clase_0, clase_1]
        vals_to_plot = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        # Versión nueva a veces devuelve array (samples, features, classes)
        vals_to_plot = shap_values[:, :, 1]
    else:
        # Fallback
        vals_to_plot = shap_values

    # Guardar gráfico
    plt.figure()
    shap.summary_plot(vals_to_plot, X_test, show=False)
    plt.savefig('Practica6/shap_importance.png', bbox_inches='tight')
    plt.close()
    print("Gráfico SHAP generado correctamente en 'Practica6/shap_importance.png'.")

except Exception as e:
    print(f"No se pudo generar el gráfico SHAP debido a un error de compatibilidad: {e}")
    # Fallback simple si SHAP falla
    plt.figure(figsize=(10,6))
    feature_imp.plot(kind='bar')
    plt.title("Feature Importance (Alternativa)")
    plt.savefig('Practica6/shap_importance.png', bbox_inches='tight')

print("\n¡Práctica 6 finalizada!")