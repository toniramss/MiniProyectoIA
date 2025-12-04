import pandas as pd
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("wine_quality_merged.csv")

print(df.head())


# Separar las características y la variable objetivo
X = df.drop("quality", axis=1)
y = df["quality"]

# Comprobar que se ha separado correctamente
print(X.head())
print(y.head())

# Escalar las características del dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled[:5])