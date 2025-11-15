import pandas as pd  
url = "https://raw.githubusercontent.com/erikgregorywebb/datasets/master/nba-salaries.csv" 
df = pd.read_csv(url) 
df.to_csv('nba-salaries.csv', index =False)