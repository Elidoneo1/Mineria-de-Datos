import pandas as pd  
url = "https://raw.githubusercontent.com/erikgregorywebb/datasets/master/nba-salaries.csv" 
df = pd.read_csv(url) 
df.to_csv('nba-salaries.csv', index =False)

#Las variables del dataset son rank, name, position, team, salary y season
#Numericos: rank, salary
#Alfanumerico: name, position, team
#Fecha: season que es el año de la temporada