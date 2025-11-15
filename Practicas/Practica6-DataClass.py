import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)

# PROBLEMA 1: CLASIFICACIÓN DE POSICIONES (Position)

print("=" * 80)
print("PROBLEMA 1: CLASIFICACIÓN DE POSICIONES DE JUGADORES")
print("=" * 80)

# Cargar datos
df = pd.read_csv("edited-salaries.csv")
print(f"Dataset shape: {df.shape}")
print(f"Posiciones únicas: {df['position'].unique()}")

# Preparar datos para clasificación de posiciones
df_position = df[['salary', 'season', 'position']].copy()

# Codificar la variable objetivo (position)
le_position = LabelEncoder()
df_position['position_encoded'] = le_position.fit_transform(df_position['position'])

print(f"\nMapping de posiciones:")
for i, pos in enumerate(le_position.classes_):
    print(f"  {pos} → {i}")

# Características y target
X_pos = df_position[['salary', 'season']]
y_pos = df_position['position_encoded']

print(f"\nDistribución de clases:")
print(y_pos.value_counts().sort_index())

# División train-test (80-20)
X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(
    X_pos, y_pos, test_size=0.2, random_state=42, stratify=y_pos
)

print(f"\nDivision de datos:")
print(f"Training set: {X_train_pos.shape[0]} muestras")
print(f"Test set: {X_test_pos.shape[0]} muestras")


# BÚSQUEDA DEL MEJOR K CON VALIDACIÓN CRUZADA
print("\n" + "-" * 50)
print("BÚSQUEDA DEL MEJOR PARÁMETRO K")
print("-" * 50)

# Probar diferentes valores de k
k_range = range(1, 31)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # Validación cruzada con 5 folds
    scores = cross_val_score(knn, X_train_pos, y_train_pos, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

# Encontrar el mejor k
best_k_index = np.argmax(k_scores)
best_k = k_range[best_k_index]
best_score = k_scores[best_k_index]

print(f"Mejor k: {best_k} con accuracy: {best_score:.4f}")

# Gráfico de búsqueda de k
plt.figure(figsize=(12, 6))
plt.plot(k_range, k_scores, marker='o', linestyle='-', linewidth=2, markersize=6)
plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Mejor k = {best_k}')
plt.xlabel('Número de Vecinos (k)')
plt.ylabel('Accuracy Promedio (5-fold CV)')
plt.title('Búsqueda del Mejor Parámetro k para KNN')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('Practica6/k_search_position.png', dpi=300, bbox_inches='tight')
plt.show()


# ENTRENAMIENTO DEL MODELO FINAL

print("\n" + "-" * 50)
print("ENTRENAMIENTO DEL MODELO FINAL")
print("-" * 50)

# Pipeline con escalado y KNN
pipeline_pos = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=best_k))
])

# Entrenar modelo
pipeline_pos.fit(X_train_pos, y_train_pos)

# Predicciones
y_pred_pos = pipeline_pos.predict(X_test_pos)
y_pred_proba_pos = pipeline_pos.predict_proba(X_test_pos)

# Métricas
accuracy_pos = accuracy_score(y_test_pos, y_pred_pos)

print(f"Accuracy en test set: {accuracy_pos:.4f}")

# Reporte de clasificación detallado
print("\nREPORTE DE CLASIFICACIÓN:")
print(classification_report(y_test_pos, y_pred_pos, 
                          target_names=le_position.classes_))


# MATRIZ DE CONFUSIÓN
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_pos, y_pred_pos)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_position.classes_,
            yticklabels=le_position.classes_)
plt.title('Matriz de Confusión - Clasificación de Posiciones')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('Practica6/confusion_matrix_position.png', dpi=300, bbox_inches='tight')
plt.show()


# GRÁFICOS DE VISUALIZACIÓN

print("\n" + "-" * 50)
print("VISUALIZACIÓN DE RESULTADOS")
print("-" * 50)

# 1. Gráfico de dispersión con predicciones correctas/incorrectas
plt.figure(figsize=(15, 12))

# Subplot 1: Datos reales
plt.subplot(2, 2, 1)
scatter = plt.scatter(X_test_pos['salary'], X_test_pos['season'], 
                     c=y_test_pos, cmap='viridis', alpha=0.6, s=30)
plt.colorbar(scatter, label='Posición Real')
plt.xlabel('Salario ($)')
plt.ylabel('Año')
plt.title('Datos Reales - Posiciones')
plt.grid(True, alpha=0.3)

# Subplot 2: Predicciones
plt.subplot(2, 2, 2)
scatter = plt.scatter(X_test_pos['salary'], X_test_pos['season'], 
                     c=y_pred_pos, cmap='viridis', alpha=0.6, s=30)
plt.colorbar(scatter, label='Posición Predicha')
plt.xlabel('Salario ($)')
plt.ylabel('Año')
plt.title('Predicciones del Modelo')
plt.grid(True, alpha=0.3)

# Subplot 3: Correctas vs Incorrectas
correct_predictions = (y_test_pos == y_pred_pos)
plt.subplot(2, 2, 3)
colors = ['red' if not correct else 'green' for correct in correct_predictions]
plt.scatter(X_test_pos['salary'], X_test_pos['season'], 
           c=colors, alpha=0.6, s=30)
plt.xlabel('Salario ($)')
plt.ylabel('Año')
plt.title('Predicciones Correctas (Verde) vs Incorrectas (Rojo)')
plt.grid(True, alpha=0.3)

# Subplot 4: Probabilidades de predicción
plt.subplot(2, 2, 4)
max_proba = np.max(y_pred_proba_pos, axis=1)
scatter = plt.scatter(X_test_pos['salary'], X_test_pos['season'], 
                     c=max_proba, cmap='coolwarm', alpha=0.6, s=30)
plt.colorbar(scatter, label='Probabilidad Máxima')
plt.xlabel('Salario ($)')
plt.ylabel('Año')
plt.title('Confianza de las Predicciones')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Practica6/classification_visualization_position.png', dpi=300, bbox_inches='tight')
plt.show()

# PROBLEMA 2: CLASIFICACIÓN DE RANGO SALARIAL (Range)
print("\n" + "=" * 80)
print("PROBLEMA 2: CLASIFICACIÓN DE RANGOS SALARIALES")
print("=" * 80)

# Preparar datos para clasificación de rangos
df_range = df[['salary', 'season', 'rank', 'range']].copy()

# Verificar que exista la columna range (de prácticas anteriores)
if 'range' not in df_range.columns:
    # Crear rangos salariales si no existen
    def categorizesalary(salary: int) -> str:
        if salary < 1000000:
            return 'Below 1M'
        elif salary < 5000000:
            return '1M-5M'
        elif salary < 10000000:
            return '5M-10M'
        elif salary < 15000000:
            return '10M-15M'
        else:
            return 'Above 15M'
    
    df_range['range'] = df_range['salary'].apply(categorizesalary)

print(f"Rangos salariales únicos: {df_range['range'].unique()}")

# Codificar rangos
le_range = LabelEncoder()
df_range['range_encoded'] = le_range.fit_transform(df_range['range'])

print(f"\nMapping de rangos salariales:")
for i, rng in enumerate(le_range.classes_):
    print(f"  {rng} → {i}")

# Características y target para rangos
X_rng = df_range[['salary', 'season', 'rank']]
y_rng = df_range['range_encoded']

print(f"\nDistribución de rangos:")
print(y_rng.value_counts().sort_index())

# División train-test
X_train_rng, X_test_rng, y_train_rng, y_test_rng = train_test_split(
    X_rng, y_rng, test_size=0.2, random_state=42, stratify=y_rng
)

# Búsqueda del mejor k para rangos
k_range_rng = range(1, 31)
k_scores_rng = []

for k in k_range_rng:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_rng, y_train_rng, cv=5, scoring='accuracy')
    k_scores_rng.append(scores.mean())

best_k_rng_index = np.argmax(k_scores_rng)
best_k_rng = k_range_rng[best_k_rng_index]
best_score_rng = k_scores_rng[best_k_rng_index]

print(f"\nMejor k para rangos: {best_k_rng} con accuracy: {best_score_rng:.4f}")

# Pipeline para rangos
pipeline_rng = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=best_k_rng))
])

pipeline_rng.fit(X_train_rng, y_train_rng)
y_pred_rng = pipeline_rng.predict(X_test_rng)
accuracy_rng = accuracy_score(y_test_rng, y_pred_rng)

print(f"Accuracy en test set (rangos): {accuracy_rng:.4f}")

print("\nREPORTE DE CLASIFICACIÓN (RANGOS):")
print(classification_report(y_test_rng, y_pred_rng, 
                          target_names=le_range.classes_))

# PREDICCIONES CON DATOS NUEVOS

print("\n" + "=" * 80)
print("PREDICCIONES CON DATOS NUEVOS")
print("=" * 80)

# Datos de ejemplo para predicción
new_players = pd.DataFrame({
    'salary': [500000, 2000000, 8000000, 12000000, 20000000],
    'season': [2005, 2005, 2005, 2005, 2005],
    'rank': [150, 100, 50, 20, 5]
})

# Predecir posiciones (usando solo salary y season)
new_players_pos = new_players[['salary', 'season']]
predicted_positions_encoded = pipeline_pos.predict(new_players_pos)
predicted_positions = le_position.inverse_transform(predicted_positions_encoded)

# Predecir rangos
predicted_ranges_encoded = pipeline_rng.predict(new_players[['salary', 'season', 'rank']])
predicted_ranges = le_range.inverse_transform(predicted_ranges_encoded)

# Mostrar resultados
new_players['predicted_position'] = predicted_positions
new_players['predicted_range'] = predicted_ranges

print("Predicciones para nuevos jugadores:")
print(new_players.to_string(index=False))

# ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS
print("\n" + "=" * 80)
print("ANÁLISIS DE CARACTERÍSTICAS")
print("=" * 80)

# Evaluar modelo con diferentes combinaciones de características
feature_combinations = {
    'Solo Salario': ['salary'],
    'Salario + Año': ['salary', 'season'],
    'Salario + Rank': ['salary', 'rank'],
    'Todas': ['salary', 'season', 'rank']
}

results = []

for name, features in feature_combinations.items():
    X_temp = df_range[features]
    y_temp = df_range['range_encoded']
    
    X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    pipeline_temp = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=best_k_rng))
    ])
    
    pipeline_temp.fit(X_train_temp, y_train_temp)
    y_pred_temp = pipeline_temp.predict(X_test_temp)
    accuracy_temp = accuracy_score(y_test_temp, y_pred_temp)
    
    results.append({
        'Características': name,
        'Accuracy': accuracy_temp,
        'N_Features': len(features)
    })

results_df = pd.DataFrame(results)
print("\nComparación de conjuntos de características:")
print(results_df.to_string(index=False))

# =============================================================================
# EXPORTACIÓN DE RESULTADOS
# =============================================================================
print("\n" + "=" * 80)
print("EXPORTACIÓN DE RESULTADOS")
print("=" * 80)

# Guardar predicciones del test set
test_results_pos = X_test_pos.copy()
test_results_pos['real_position'] = le_position.inverse_transform(y_test_pos)
test_results_pos['predicted_position'] = le_position.inverse_transform(y_pred_pos)
test_results_pos['correct'] = (y_test_pos == y_pred_pos)

test_results_pos.to_csv('Practica6/test_predictions_position.csv', index=False)

test_results_rng = X_test_rng.copy()
test_results_rng['real_range'] = le_range.inverse_transform(y_test_rng)
test_results_rng['predicted_range'] = le_range.inverse_transform(y_pred_rng)
test_results_rng['correct'] = (y_test_rng == y_pred_rng)

test_results_rng.to_csv('Practica6/test_predictions_range.csv', index=False)

# Resumen de modelos
model_summary = pd.DataFrame({
    'Problema': ['Clasificación de Posiciones', 'Clasificación de Rangos'],
    'Mejor k': [best_k, best_k_rng],
    'Accuracy Train (CV)': [best_score, best_score_rng],
    'Accuracy Test': [accuracy_pos, accuracy_rng],
    'N_Clases': [len(le_position.classes_), len(le_range.classes_)],
    'Características': ['salary, season', 'salary, season, rank']
})

model_summary.to_csv('Practica6/model_summary.csv', index=False)

print("Archivos exportados:")
print("- Practica6/test_predictions_position.csv")
print("- Practica6/test_predictions_range.csv") 
print("- Practica6/model_summary.csv")
print("- Practica6/k_search_position.png")
print("- Practica6/confusion_matrix_position.png")
print("- Practica6/classification_visualization_position.png")

print("\n" + "=" * 80)
print("¡CLASIFICACIÓN CON KNN COMPLETADA!")
print("=" * 80)