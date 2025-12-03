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

# Histogramas de variables seleccionadas por tipo de vino
vars_plot = ["density", "residual_sugar", "chlorides", "alcohol"]
color_map = {0: "white", 1: "red"}

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

    for ax, (color_val, row) in zip(g.axes.flat, stats.iterrows()):
        mean_val = row["mean"]
        n_val = int(row["count"])
        color_name = color_map[color_val]
        ax.set_title(f"{color_name} wine | {var}\nmedia={mean_val}, n={n_val}")

    g.set_axis_labels(var, "Densidad")
    plt.suptitle(f"Distribución de {var} por tipo de vino", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])  # deja espacio arriba
    plt.show()



#BOXPLOTS

#Informe: 
"""En este boxplot se observa que la mayoría de los vinos tienen un contenido de alcohol 
entre aproximadamente 9.8° y 11.1°, con una mediana cercana a 10.3°, mientras que los valores 
normales se extienden desde unos 8.4° hasta 14°, destacando únicamente un vino con alrededor 
de 15° que aparece como un outlier por estar fuera del rango habitual."""

# Boxplot de la variable 'alcohol'
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="color", y="alcohol")
plt.xticks([0,1], ["white","red"])
plt.title("Alcohol por tipo de vino")
plt.show()

# Boxplot de la variable 'residual_sugar'
"""En este boxplot se observa que la mayoría de los vinos tienen un contenido de azúcar residual 
bastante bajo, concentrado entre aproximadamente 1 y 15 gramos por litro. Sin embargo, hay una 
gran cantidad de outliers que se extienden mucho más allá de este rango, llegando incluso a valores 
superiores a 200 gramos por litro. Esto indica que, aunque la mayoría de los vinos tienen niveles 
bajos de azúcar residual, existen algunos vinos con cantidades extremadamente altas que se consideran 
atípicos en comparación con el resto de la muestra."""
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="color", y="residual_sugar")
plt.xticks([0,1], ["white","red"])
plt.title("Residual sugar por tipo de vino")
plt.show()

# Boxplot de la variable 'density'
"""En este boxplot se observa que la mayoría de los vinos tienen una densidad concentrada en un rango 
estrecho, aproximadamente entre 0.98 y 1.01. Sin embargo, hay varios outliers que se extienden más allá de este rango,
lo que indica que existen vinos con densidades atípicas en comparación con la mayoría de la muestra. Estos outliers pueden 
representar vinos con características físicas inusuales."""
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="color", y="density")
plt.xticks([0,1], ["white","red"])
plt.title("Density por tipo de vino")
plt.show()

# Boxplot de la variable 'chlorides'
"""En este boxplot se observa que la mayoría de los vinos tienen niveles de cloruros concentrados en un rango 
estrecho, aproximadamente entre 0.03 y 0.09. Sin embargo, hay varios outliers que se extienden más allá de este rango,
lo que indica que existen vinos con niveles de cloruros atípicos en comparación con la mayoría de la muestra. Estos outliers pueden 
representar vinos con características químicas inusuales."""
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="color", y="chlorides")
plt.xticks([0,1], ["white","red"])
plt.title("Chlorides por tipo de vino")
plt.show()

# Boxplot de la variable 'sulphates'
"""En este boxplot se ve que la mayoría de los vinos tienen valores de sulphates bastante bajos, 
concentrados entre aproximadamente 0.45 y 0.65. Como la mayor parte de los datos está muy junta, 
todos los valores que se alejan un poco hacia la derecha aparecen como outliers. Por eso salen 
tantos puntos sueltos: no es un error, simplemente esta variable es muy irregular y tiene muchos 
valores altos que se consideran extremos."""
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="color", y="sulphates")
plt.xticks([0,1], ["white","red"])
plt.title("Boxplot de Sulphates")
plt.show()