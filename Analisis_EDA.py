import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set(style="whitegrid")

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

print("\nCorrelación de cada característica con la calidad del vino:")
print(df.corr()["quality"].sort_values(ascending=False))

# Verificar si hay valores negativos en las columnas numéricas
print("\nValores negativos por columna:")
print((df < 0).sum())

# Histograma de la variable 'quality' con conteos y media
plt.figure(figsize=(8, 5))
ax = sns.histplot(
    data=df,
    x="quality",
    bins=range(int(df["quality"].min()), int(df["quality"].max()) + 2),
    discrete=True,
    kde=False
)

# Añadir el número de vinos encima de cada barra (Histograma)
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

# Añadir una línea vertical en la media (Histograma)
mean_quality = df["quality"].mean()
plt.axvline(mean_quality, color="red", linestyle="--", label=f"Media = {mean_quality:.2f}")

plt.title("Histograma de la variable 'quality'")
plt.xlabel("Quality")
plt.ylabel("Frecuencia")
plt.legend()
plt.tight_layout()
plt.show()


#Informe: 
"""En este boxplot se observa que la mayoría de los vinos tienen un contenido de alcohol 
entre aproximadamente 9.8° y 11.1°, con una mediana cercana a 10.3°, mientras que los valores 
normales se extienden desde unos 8.4° hasta 14°, destacando únicamente un vino con alrededor 
de 15° que aparece como un outlier por estar fuera del rango habitual."""

# Boxplot de la variable 'alcohol'
plt.figure(figsize=(6, 4))
sns.boxplot(x=df["alcohol"], color="violet")
plt.title("Boxplot de Alcohol")
plt.xlabel("Alcohol")
plt.show()

"""En este boxplot se ve que la mayoría de los vinos tienen valores de sulphates bastante bajos, 
concentrados entre aproximadamente 0.45 y 0.65. Como la mayor parte de los datos está muy junta, 
todos los valores que se alejan un poco hacia la derecha aparecen como outliers. Por eso salen 
tantos puntos sueltos: no es un error, simplemente esta variable es muy irregular y tiene muchos 
valores altos que se consideran extremos."""
# Boxplot de la variable 'sulphates'
plt.figure(figsize=(6, 4))
sns.boxplot(x=df["sulphates"], color="skyblue")
plt.title("Boxplot de Sulphates")
plt.xlabel("Sulphates")
plt.show()

# Histogramas de variables seleccionadas por tipo de vino
vars_plot = ["density", "residual_sugar", "chlorides", "alcohol"]
color_map = {0: "white", 1: "red"}  # 0 -> white, 1 -> red

for var in vars_plot:
    stats = df.groupby("color")[var].agg(["mean", "count"]).round(2)

    g = sns.FacetGrid(df, col="color", sharex=False, sharey=False)
    g.map_dataframe(
        sns.histplot,
        x=var,
        kde=True,
        stat="density",
        common_norm=False,
        bins=30
    )

    # Recorrer ejes y poner título específico con red/white
    for ax, (color_val, row) in zip(g.axes.flat, stats.iterrows()):
        mean_val = row["mean"]
        n_val = int(row["count"])
        color_name = color_map[color_val]  # convierte 0/1 a red/white
        ax.set_title(f"{color_name} wine | {var}\nmedia={mean_val}, n={n_val}")

    g.set_axis_labels(var, "Densidad")
    plt.suptitle(f"Distribución de {var} por tipo de vino", y=1.05)
    plt.tight_layout()
    plt.show()
