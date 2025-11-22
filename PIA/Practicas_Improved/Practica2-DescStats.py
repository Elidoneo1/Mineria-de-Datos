import pandas as pd #importamos pandas


def categorizesalary(salary:int)->str:
    if salary<1000000:
        return 'Below 1 million'
    if salary>=1000000 and salary <5000000:
        return 'Above 1 million'
    if salary>=5000000 and salary <10000000:
        return 'Above 5 million'
    if salary>=10000000 and salary <15000000:
        return 'Above 10 million'
    if salary>=15000000:
        return 'Above 15 million'
#Lo que mencionamos en la primera practica sobre las posiciones no es del todo correcto, ya que hay jugadores que juegan en varias posiciones
#  pero para simplificar el analisis vamos a categorizar las posiciones de la siguiente manera
def categorize(position:str)->str:
    if 'C' in position:
        return 'Center'
    if 'PG' in position:
        return 'Point Guard'
    if 'SF' in position:
        return 'Small Forward'
    if 'PF' in position:
        return 'Power Forward'
    if 'SG' in position:
        return 'Shooting Guard'
    if 'G' in position:
        return 'Guard'
    if 'PF' in position:
        return 'Point Forward'
    if 'FC' in position:
        return 'Forward Center'
    if 'F' in position:
        return 'Forward'
    if 'S' in position:
        return 'Swingman'


df= pd.read_csv("nba-salaries.csv")
range = df['salary']
df['range'] = range.map(categorizesalary)
position = df['position']
newpos = position.map(categorize)
df['position'] = newpos
df.to_csv('edited-salaries.csv',index = False)
print(df.head())

# Normalizar salario usando Z-score por temporada
# Esto nos dice qué tan alto es el salario comparado con OTROS jugadores de ESE MISMO AÑO
def get_zscore(x):
    return (x - x.mean()) / x.std()

df['salary_zscore'] = df.groupby('season')['salary'].transform(get_zscore)

# O calcular el porcentaje del tope salarial (aproximado usando el máximo de ese año)
df['max_salary_year'] = df.groupby('season')['salary'].transform('max')
df['salary_share'] = df['salary'] / df['max_salary_year']