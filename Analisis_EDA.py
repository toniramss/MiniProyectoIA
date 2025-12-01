import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el CSV
df = pd.read_csv("wine_quality_merged.csv")

print(df.head())
print("\nDescripción estadística del DataFrame combinado:\n")
print(df.describe())

# Matriz de correlación
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de correlación del Wine Quality")
plt.tight_layout()
plt.show()

# Histograma de la variable 'quality'
plt.figure(figsize=(8,5))
sns.histplot(data=df, x="quality", bins=range(int(df["quality"].min()), int(df["quality"].max()) + 2), 
             discrete=True, kde=False)
plt.title("Histograma de la variable 'quality'")
plt.xlabel("Quality")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()