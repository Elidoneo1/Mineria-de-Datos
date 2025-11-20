import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Cargar datos
df = pd.read_csv("nba_processed.csv")

print("="*60)
print("PROBLEMA: Predecir si un jugador es 'High Earner' (Top 25% de salarios)")
print("Sin usar el Ranking (para evitar data leakage)")
print("="*60)

# Features: Año, Equipo, Posición
X = df[['season', 'team_encoded', 'position_encoded']]
y = df['high_earner']

# División
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo: Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predicción
y_pred = clf.predict(X_test)

# Resultados
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))

# Importancia de variables
feature_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nVariables más importantes para determinar un sueldo alto:")
print(feature_imp)

# Matriz de Confusión
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión: High Earner vs Normal')
plt.ylabel('Realidad')
plt.xlabel('Predicción')
plt.savefig('Practica6/rf_confusion_matrix.png')
plt.show()