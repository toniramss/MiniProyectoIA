import pandas as pd

url_red_wine = "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/refs/heads/master/winequality-red.csv"
irl_white_wine = "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/refs/heads/master/winequality-white.csv"

df_red_wine = pd.read_csv(url_red_wine, sep=';')
df_white_wine = pd.read_csv(irl_white_wine, sep=';')

print(df_red_wine.head())


print("Describe")
#print(df_red_wine.describe())

print("Cantidad de nulos por columna")
print(df_red_wine.isnull().sum())

