# 1. IMPORTACIÓN DE LIBRERÍAS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. CARGA DEL DATASET
df = pd.read_csv('wine_quality_merged.csv')
# df = pd.read_csv('wine_quality_binario.csv')
print("Primeras filas")
print(df.head())

# 3. SELECCIÓN DE VARIABLES (FEATURES) Y VARIABLE OBJETIVO
# Variable objetivo
y = df["quality"]

# Variables predictoras
X = df.drop(columns=["quality"])

# 4. DIVISIÓN EN CONJUNTO DE ENTRENAMIENTO Y TEST
# División del dataset -> 80% entrenamiento -> 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. CREACIÓN Y ENTRENAMIENTO DEL ÁRBOL DE DECISIÓN
arbol = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

# Entrenamos el modelo con los datos de entrenamiento
arbol.fit(X_train, y_train)

# 6. PREDICCIÓN Y EVALUACIÓN DEL MODELO
y_pred = arbol.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy en test: {accuracy:.4f}\n")

# Precisión, recall, f1-score (texto)
print("Informe de clasificación:")
print(classification_report(y_test, y_pred, zero_division=0), "\n")

# Matriz de confusión
print("Matriz de confusión:")
cm = confusion_matrix(y_test, y_pred)
print(cm, "\n")

# Matriz de confusión gráfica
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
plt.title("Matriz de Confusión - Árbol de Decisión")
plt.show()

# 7. IMPORTANCIA DE LAS VARIABLES
importancias = pd.Series(arbol.feature_importances_, index=X.columns)
importancias_ordenadas = importancias.sort_values(ascending=False)

print("Importancia de variables:")
print(importancias_ordenadas, "\n")

# 8. VISUALIZACIÓN DEL ÁRBOL DE DECISIÓN
plt.figure(figsize=(40, 20))
plot_tree(
    arbol,
    feature_names=X.columns,
    class_names=[str(c) for c in sorted(y.unique())],
    filled=True,
    rounded=True,
    fontsize=6
)

plt.title("Decision tree wine quality")
plt.tight_layout()
plt.show()

# 9. EXTRAER TODAS LAS HOJAS FINALES DEL ÁRBOL
tree = arbol.tree_
children_left = tree.children_left
children_right = tree.children_right
n_nodes = tree.node_count

# 1) Calcular profundidad de cada nodo
node_depth = {}
stack = [(0, 0)]

while stack:
    nodo, depth = stack.pop()
    node_depth[nodo] = depth

    # Si no es hoja, añadimos los hijos
    if children_left[nodo] != -1:
        stack.append((children_left[nodo], depth + 1))
    if children_right[nodo] != -1:
        stack.append((children_right[nodo], depth + 1))

# 2) Identificar hojas
leaf_nodes = [
    nodo for nodo in range(n_nodes)
    if children_left[nodo] == -1 and children_right[nodo] == -1
]

print("\n======= HOJAS FINALES DEL ÁRBOL =======\n")

# 3) Imprimir info de cada hoja
for nodo in leaf_nodes:
    depth = node_depth[nodo]
    samples = tree.n_node_samples[nodo]
    value = tree.value[nodo][0]
    predicted_class = value.argmax()

    print(f"--- Hoja nodo {nodo} ---")
    print(f"Profundidad: {depth}")
    print(f"Muestras en la hoja: {samples}")
    print(f"Distribución de clases (value): {value}")
    print(f"Clase predicha: {predicted_class}")
    print("---------------------------------------\n")

# 10. GRÁFICO CON LAS MÉTRICAS DEL INFORME DE CLASIFICACIÓN
# Obtenemos el informe en formato diccionario
report_dict = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).T

# Ordenamos las filas para que coincidan con la tabla de sklearn
orden_filas = [str(c) for c in sorted(y.unique())] + ["accuracy", "macro avg", "weighted avg"]
report_df = report_df.loc[orden_filas, ["precision", "recall", "f1-score", "support"]]

# Redondeamos las métricas y dejamos 'support' como entero
report_df[["precision", "recall", "f1-score"]] = report_df[["precision", "recall", "f1-score"]].round(2)
report_df["support"] = report_df["support"].astype(int)

# Creamos una figura tipo "tabla" para visualizar los resultados
plt.figure(figsize=(8, 3.5))
plt.title("Métricas por clase - Árbol de decisión", pad=20)
plt.axis('off')

tabla = plt.table(
    cellText=report_df.values,
    rowLabels=report_df.index,
    colLabels=report_df.columns,
    loc='center'
)

tabla.auto_set_font_size(False)
tabla.set_fontsize(6)
tabla.scale(1.2, 1.4)

plt.show()
