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

# Histograma de la variable 'quality' con conteos y media
plt.figure(figsize=(8, 5))
ax = sns.histplot(
    data=df,
    x="quality",
    bins=range(int(df["quality"].min()), int(df["quality"].max()) + 2),
    discrete=True,
    kde=False
)

# Añadir el número de vinos encima de cada barra
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(
            f"{int(height)}",
            (p.get_x() + p.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=10
        )

# Añadir una línea vertical en la media
mean_quality = df["quality"].mean()
plt.axvline(mean_quality, color="red", linestyle="--", label=f"Media = {mean_quality:.2f}")

plt.title("Histograma de la variable 'quality'")
plt.xlabel("Quality")
plt.ylabel("Frecuencia")
plt.legend()
plt.tight_layout()
plt.show()
