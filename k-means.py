import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1. Cargar datos
df = pd.read_csv("wine_quality_merged.csv")

# 2. Definir X sin usar quality ni color
features = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide",
    "density", "ph", "sulphates", "alcohol"
]
X = df[features].copy()


# 3. Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Búsqueda de k (codo + silhouette)
inertias = []
silhouettes = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    sil = silhouette_score(X_scaled, labels)
    silhouettes.append(sil)

# Gráfico del codo
plt.figure(figsize=(6, 4))
plt.plot(k_values, inertias, marker="o")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inercia (SSE)")
plt.title("Método del codo")
plt.tight_layout()
plt.show()

# Gráfico de silhouette
plt.figure(figsize=(6, 4))
plt.plot(k_values, silhouettes, marker="o")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Silhouette")
plt.title("Silhouette por k")
plt.tight_layout()
plt.show()

print("Inercia por k:", dict(zip(k_values, inertias)))
print("Silhouette por k:", dict(zip(k_values, silhouettes)))

# 5. Entrenar modelo final con el k elegido
k_optimo = 4  # Usamos 2 o 4 según los gráficos obtenidos

"""
Se puede justificar usar tanto k=2 como k=4 porque cada uno responde a una idea distinta de 
“tipo de vino” en tus datos físico‑químicos.

Por qué k = 2:
    - Con k=2 el silhouette es máximo, lo que indica dos grupos muy compactos y bien separados
      en el espacio de variables estandarizadas.

    - Es razonable interpretarlos como dos grandes familias de vinos con perfiles químicos 
      diferentes (por ejemplo, más cercanos a blanco vs tinto), lo que encaja con que el dataset 
      mezcla ambos tipos.

Por qué k = 4:
    - El método del codo muestra una mejora clara de la inercia hasta alrededor de k=4 y, a partir de ahí, 
      las ganancias son menores, por lo que k=4 sigue siendo un valor eficiente.

    - Con k=4 el silhouette sigue siendo aceptable y obtienes más detalle: en lugar de solo dos grandes 
      grupos, aparecen cuatro subperfiles de vino (distintas combinaciones de alcohol, acidez, azúcar, etc.), 
      lo que permite una interpretación enológica más rica sin usar la calidad.
"""

kmeans_final = KMeans(n_clusters=k_optimo, n_init=10, random_state=42)
df["cluster"] = kmeans_final.fit_predict(X_scaled)

# 6. Perfil de clusters (solo con variables físico-químicas)
cluster_profile = df.groupby("cluster")[features].mean()
print("\nMedias de las variables físico-químicas por cluster:")
print(cluster_profile)

# 7. PCA para visualizar clusters (sin quality)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

plt.figure(figsize=(6, 5))
sns.scatterplot(
    data=df, x="PC1", y="PC2",
    hue="cluster", palette="tab10", alpha=0.6
)
plt.title(f"Clusters K-means (k={k_optimo}) en espacio PCA")
plt.tight_layout()
plt.show()

print("Varianza explicada por PC1 y PC2:", pca.explained_variance_ratio_)
