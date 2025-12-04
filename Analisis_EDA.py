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
        kde=False,
        stat="count",
        bins=30
    )

    for ax, (color_val, row) in zip(g.axes.flat, stats.iterrows()):
        mean_val = row["mean"]
        n_val = int(row["count"])
        color_name = color_map[color_val]
        ax.set_title(f"{color_name} wine | {var}\nmedia={mean_val}, n={n_val}")

    g.set_axis_labels(var, "Recuento de vinos")
    plt.suptitle(f"Recuento de vinos por intervalo de {var} y tipo de vino", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
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

# Gráfica de líneas de volatile_acidity por calidad y color
# Media de volatile_acidity por calidad y color
va_mean = (
    df.groupby(["quality", "color"])["volatile_acidity"]
      .mean()
      .reset_index()
)

va_mean["color_name"] = va_mean["color"].map({0: "white", 1: "red"})

plt.figure(figsize=(8, 5))
sns.lineplot(
    data=va_mean,
    x="quality",
    y="volatile_acidity",
    hue="color_name",
    marker="o"
)
plt.title("Acidez volátil media por calidad y tipo de vino")
plt.xlabel("Calidad")
plt.ylabel("Volatile acidity media")
plt.legend(title="Tipo de vino")
plt.tight_layout()
plt.show()

# Gráfica de líneas de sulphates por calidad y color
# Media de sulphates por calidad y color
sul_mean = (
    df.groupby(["quality", "color"])["sulphates"]
      .mean()
      .reset_index()
)

sul_mean["color_name"] = sul_mean["color"].map({0: "white", 1: "red"})

plt.figure(figsize=(8, 5))
sns.lineplot(
    data=sul_mean,
    x="quality",
    y="sulphates",
    hue="color_name",
    marker="o"
)
plt.title("Sulphates medios por calidad y tipo de vino")
plt.xlabel("Calidad")
plt.ylabel("Sulphates medios")
plt.legend(title="Tipo de vino")
plt.tight_layout()
plt.show()

# Gráfica de líneas de residual_sugar por alcohol y color
# Definir bins de alcohol
n_bins = 8  # ajusta si quieres más/menos resolución
df["alcohol_bin"] = pd.cut(df["alcohol"], bins=n_bins)

alco_sugar = (
    df.groupby(["alcohol_bin", "color"])["residual_sugar"]
      .mean()
      .reset_index()
)

alco_sugar["color_name"] = alco_sugar["color"].map({0: "white", 1: "red"})
# Usamos el centro del bin para el eje X
alco_sugar["alcohol_bin_center"] = alco_sugar["alcohol_bin"].apply(lambda x: x.mid)

plt.figure(figsize=(8, 5))
sns.lineplot(
    data=alco_sugar,
    x="alcohol_bin_center",
    y="residual_sugar",
    hue="color_name",
    marker="o"
)
plt.title("Azúcar residual media por bin de alcohol y tipo de vino")
plt.xlabel("Alcohol (centro del bin)")
plt.ylabel("Residual sugar media")
plt.legend(title="Tipo de vino")
plt.tight_layout()
plt.show()

# Scatterplot de Alcohol vs Density por tipo de vino
"""Visualización de la relación entre Alcohol y Density por tipo de vino"""
"""En el gráfico se observa que los vinos blancos se concentran en valores de
densidad más bajos, mientras que los vinos tintos presentan densidades ligeramente
superiores. Esto indica que, en este dataset específico, la densidad media del vino
tinto es algo mayor que la del vino blanco."""
 
custom_palette = {0: "#8FFFA0", 1: "#FF4B4B"}  # 0=blanco(verde), 1=tinto(rojo)
 
ax = sns.scatterplot(
    data=df,
    x="density",
    y="alcohol",
    hue="color",
    palette=custom_palette
)
 
plt.title("Alcohol vs Density por tipo de vino")
 
# Obtener handles y labels verdaderos
handles, labels = ax.get_legend_handles_labels()
 
# Cambiar solo el texto asociado a 0 y 1 sin invertir posiciones
ax.legend(handles, ["Vino Blanco", "Vino Tinto"], title="Tipo de Vino")
 
plt.show()