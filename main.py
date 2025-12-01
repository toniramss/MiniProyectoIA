import pandas as pd

# --- 1. Descarga y carga de datos ---
url_red_wine = "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/refs/heads/master/winequality-red.csv"
url_white_wine = "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/refs/heads/master/winequality-white.csv"

df_red_wine = pd.read_csv(url_red_wine, sep=';')
df_white_wine = pd.read_csv(url_white_wine, sep=';')

# --- 2. Informes iniciales ---
print("Primeras filas del vino tinto:")
print(df_red_wine.head())
print("\nPrimeras filas del vino blanco:")
print(df_white_wine.head())

print("\nDistribución estadística vino tinto:")
print(df_red_wine.describe())

print("\nDistribución estadística vino blanco:")
print(df_white_wine.describe())

print("\nInformación vino tinto:")
print(df_red_wine.info())

print("\nInformación vino blanco:")
print(df_white_wine.info())

print("\nNulos por columna vino tinto:")
print(df_red_wine.isnull().sum())

print("\nNulos por columna vino blanco:")
print(df_white_wine.isnull().sum())

print("\nDuplicados vino tinto:", df_red_wine.duplicated().sum())
print("Duplicados vino blanco:", df_white_wine.duplicated().sum())

print("\nFilas duplicadas en vino tinto:")
duplicates = df_red_wine[df_red_wine.duplicated(keep=False)]
duplicates_sorted = duplicates.sort_values(list(df_red_wine.columns))
print(duplicates_sorted)

# --- 3. Limpieza de datos ---
df_red_wine = df_red_wine.drop_duplicates()
df_white_wine = df_white_wine.drop_duplicates()

# Quitar espacios y normalizar nombres de columnas
df_red_wine.columns = df_red_wine.columns.str.strip().str.lower().str.replace(' ', '_')
df_white_wine.columns = df_white_wine.columns.str.strip().str.lower().str.replace(' ', '_')

print("\nColumnas normalizadas vino tinto:")
print(df_red_wine.columns)
print("Columnas normalizadas vino blanco:")
print(df_white_wine.columns)

print("\nFilas tras limpieza vino tinto:", len(df_red_wine))
print("Filas tras limpieza vino blanco:", len(df_white_wine))

# --- 4. Añadir columna color al final ---
df_red_wine['color'] = 1
df_white_wine['color'] = 0

# --- 5. Fusionar ambos DataFrames ---
df_wine = pd.concat([df_red_wine, df_white_wine], ignore_index=True)

print("\nInformación del DataFrame combinado:")
print(df_wine.info())
print("\nPrimeras filas del DataFrame combinado:")
print(df_wine.head(10))

# --- 6. Guardar resultado en CSV ---
#df_wine.to_csv('wine_quality_merged.csv', index=False)
