import pandas as pd

url_red_wine = "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/refs/heads/master/winequality-red.csv"
irl_white_wine = "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/refs/heads/master/winequality-white.csv"

df_red_wine = pd.read_csv(url_red_wine, sep=';')
df_white_wine = pd.read_csv(irl_white_wine, sep=';')

print(df_red_wine.head())


print("Describe")
print(df_red_wine.describe())

print("Cantidad de nulos por columna")
print(df_red_wine.isnull().sum())

print("\nCantidad de duplicados")
print(df_red_wine.duplicated().sum())


# Mostrar filas duplicadas (posible informe)
duplicates = df_red_wine[df_red_wine.duplicated(keep=False)]
duplicates_sorted = duplicates.sort_values(list(df_red_wine.columns))

print(duplicates_sorted)

#Eliminar las filas duplicadas
df_red_wine = df_red_wine.drop_duplicates()

# Quitar espacios en nombres de columnas
df_red_wine.columns = df_red_wine.columns.str.strip().str.lower().str.replace(' ', '_')
print(df_red_wine.columns)
df_white_wine.columns = df_white_wine.columns.str.strip().str.lower().str.replace(' ', '_')
print(df_white_wine.columns)
