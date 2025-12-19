# 1. IMPORTACIÓN DE LIBRERÍAS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree

# 2. CARGA DEL DATASET
df = pd.read_csv("wine_quality_merged.csv")
# df = pd.read_csv("wine_quality_binario.csv")

print("Primeras filas del dataset:")
print(df.head(), "\n")

# 3. SELECCIÓN DE VARIABLES (FEATURES) Y VARIABLE OBJETIVO
# Variable objetivo
y = df["quality"]

# Variables predictoras
X = df.drop(columns=["quality"])

# 4. DIVISIÓN EN CONJUNTO DE ENTRENAMIENTO Y TEST
# División del dataset -> 80% entrenamiento, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. CREACIÓN Y ENTRENAMIENTO DEL RANDOM FOREST
modelo_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=2
)

# Entrenamos el modelo con los datos de entrenamiento
modelo_rf.fit(X_train, y_train)

# 6. PREDICCIÓN Y EVALUACIÓN DEL MODELO
y_pred = modelo_rf.predict(X_test)

# (accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy en test (Random Forest): {accuracy:.4f}\n")

# (precision, recall, f1-score por clase)
print("Informe de clasificación (Random Forest):")
print(classification_report(y_test, y_pred), "\n")

# (numérica)
print("Matriz de confusión (Random Forest):")
cm = confusion_matrix(y_test, y_pred)
print(cm, "\n")

# 6.1 MATRIZ DE CONFUSIÓN GRÁFICA
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=sorted(y.unique()),
    yticklabels=sorted(y.unique()),
    linewidths=1,
    linecolor="black"
)
plt.xlabel("Predicción del modelo")
plt.ylabel("Valor real")
plt.title("Matriz de Confusión - Random Forest")
plt.tight_layout()
plt.show()

# 7. IMPORTANCIA DE LAS VARIABLES
importancias = pd.Series(modelo_rf.feature_importances_, index=X.columns)
importancias_ordenadas = importancias.sort_values(ascending=False)

print("Importancia de variables (Random Forest):")
print(importancias_ordenadas, "\n")

# Representación gráfica de la importancia de las variables
plt.figure(figsize=(10, 6))
importancias_ordenadas.plot(kind="bar")
plt.title("Importancia de las variables - Random Forest")
plt.ylabel("Importancia")
plt.xlabel("Variables")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()