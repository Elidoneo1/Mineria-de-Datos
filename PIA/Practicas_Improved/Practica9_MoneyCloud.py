import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

# 1. Crear la carpeta 'Practica9' si no existe
if not os.path.exists('Practica9'):
    os.makedirs('Practica9')
    print("Carpeta 'Practica9' creada.")

# 2. Cargar datos
try:
    df = pd.read_csv("nba_processed.csv")
except FileNotFoundError:
    print("ERROR: No se encontró 'nba_processed.csv'. Asegúrate de haber corrido la Practica 1.")
    exit()

# 3. Agrupar por equipo y sumar salarios
# Convertimos a dict para manipularlo fácilmente
team_spending_raw = df.groupby('team')['salary'].sum().fillna(0).to_dict()

# --- CORRECCIÓN DEL ERROR ---
# El error "anchor not supported for multiline text" ocurre porque algunos equipos
# tienen un salto de línea (\n) oculto en el nombre.
# Vamos a limpiar las claves del diccionario:
team_spending = {}
for team_name, amount in team_spending_raw.items():
    # Convertir a string, quitar saltos de línea y espacios extra
    clean_name = str(team_name).replace('\n', ' ').strip()
    
    # Si ya existe el equipo limpio, sumamos el monto (para no duplicar)
    if clean_name in team_spending:
        team_spending[clean_name] += amount
    else:
        team_spending[clean_name] = amount

print("Generando nube de palabras basada en gasto salarial...")

try:
    # Crear el objeto WordCloud
    # Preferimos una fuente simple para evitar problemas de renderizado
    wordcloud = WordCloud(width=1600, height=800, 
                          background_color='black', 
                          colormap='Pastel1', 
                          contour_width=1, 
                          contour_color='white',
                          random_state=42,
                          # Evitamos que intente collocations que a veces fallan con textos cortos
                          collocations=False).generate_from_frequencies(team_spending)

    # Visualizar
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("NBA Moneyball: Tamaño = Gasto Total en Salarios (2000-2020)", fontsize=20, color='black')

    # Guardar imagen
    output_path = 'Practica9/money_wordcloud.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Gráfico guardado exitosamente en: {output_path}")

    plt.show()

except Exception as e:
    print("\nOcurrió un error al generar la nube de palabras:")
    print(e)
    print("\nIntenta actualizar tu librería wordcloud: pip install --upgrade wordcloud")

# Extra: Top 5 equipos que más gastan
print("\nTop 5 Equipos con mayor inversión histórica:")
sorted_spending = sorted(team_spending.items(), key=lambda x: x[1], reverse=True)

for i, (team, money) in enumerate(sorted_spending[:5], 1):
    print(f"{i}. {team}: ${money:,.0f}")