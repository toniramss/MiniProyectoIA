import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("wine_quality_merged.csv")

print(df.head())

print("\nDescripción estadística del DataFrame combinado:\n")
print(df.describe())

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de correlación del Wine Quality")
plt.show()
