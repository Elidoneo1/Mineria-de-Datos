import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 1. Cargar el dataset original
print("Cargando datos...")
df = pd.read_csv("nba-salaries.csv")

# 2. Mapeo de Equipos (Corregir nombres antiguos)
team_map = {
    'Seattle SuperSonics': 'Oklahoma City Thunder',
    'New Jersey Nets': 'Brooklyn Nets',
    'New Orleans Hornets': 'New Orleans Pelicans',
    'Charlotte Bobcats': 'Charlotte Hornets',
    'Vancouver Grizzlies': 'Memphis Grizzlies',
    'NO/Oklahoma City Hornets': 'New Orleans Pelicans'
}
df['team'] = df['team'].replace(team_map)

# 3. Limpieza de Posiciones
# Mapeamos posiciones compuestas a la principal (ej: 'C-PF' -> 'C')
def clean_position(pos):
    if not isinstance(pos, str): return 'Unknown'
    # Tomar la primera posición si hay varias
    return pos.split('-')[0].strip()

df['position'] = df['position'].apply(clean_position)

# Mapeo a nombres completos
pos_map = {
    'C': 'Center',
    'PF': 'Power Forward',
    'SF': 'Small Forward',
    'SG': 'Shooting Guard',
    'PG': 'Point Guard',
    'G': 'Guard',
    'F': 'Forward'
}
df['position'] = df['position'].map(pos_map).fillna(df['position'])

# 4. INGENIERÍA DE DATOS: Normalización de Salario (Z-Score)
# Esto permite comparar el salario de 2000 con el de 2020
def get_zscore(x):
    if x.std() == 0: return 0
    return (x - x.mean()) / x.std()

df['salary_zscore'] = df.groupby('season')['salary'].transform(get_zscore)

# 5. Codificación (Label Encoding) para Modelos
le_pos = LabelEncoder()
le_team = LabelEncoder()
df['position_encoded'] = le_pos.fit_transform(df['position'])
df['team_encoded'] = le_team.fit_transform(df['team'])

# 6. Crear variable binaria para clasificación (Top 25% de salarios)
# 1 si gana mucho, 0 si gana normal/poco
df['high_earner'] = df.groupby('season')['salary'].transform(
    lambda x: x > x.quantile(0.75)
).astype(int)

# 7. Guardar
print("Guardando archivo procesado: nba_processed.csv")
df.to_csv('nba_processed.csv', index=False)
print(df.head())

#Las variables del dataset son rank, name, position, team, salary y season
#Numericos: rank, salary
#Alfanumerico: name, position, team
#Fecha: season que es el año de la temporada
