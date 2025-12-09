import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.metrics import roc_curve, roc_auc_score

# 1. Cargar datos
df = pd.read_csv("wine_quality_merged.csv")

# 2. Crear variable binaria: 1 = buen vino (quality >= 6), 0 = resto
df["good_wine"] = (df["quality"] >= 6).astype(int)

# 3. Definir X (features) e y (target)
feature_cols = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol"
]

X = df[feature_cols]
y = df["good_wine"]

# 4. Train-test split (estratificado para mantener proporción de clases)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Pipeline: escalado + regresión logística
pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # útil si la clase 1 es minoritaria
        random_state=42
    ))
])

# 6. Entrenamiento
pipe.fit(X_train, y_train)

# 7. Predicciones
y_pred = pipe.predict(X_test)

# 8. Evaluación
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# 9. Gráfica de la matriz de confusión
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Pred. 0 (no bueno)", "Pred. 1 (bueno)"],
    yticklabels=["Real 0 (no bueno)", "Real 1 (bueno)"]
)
plt.xlabel("Predicción")
plt.ylabel("Valor real")
plt.title("Matriz de confusión - Regresión logística")
plt.tight_layout()
plt.show()

# 10. Curva ROC y AUC
# Probabilidades de clase positiva
y_prob = pipe.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", label="Azar")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC - Regresión logística")
plt.legend()
plt.tight_layout()
plt.show()

# 11. Visualización de la distribución de la variable objetivo
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="good_wine")
plt.xticks([0, 1], ["0 (no bueno)", "1 (bueno)"])
plt.title("Distribución de la variable objetivo good_wine")
plt.xlabel("Clase")
plt.ylabel("Número de muestras")
plt.tight_layout()
plt.show()

# 12. Visualización de las probabilidades predichas
# Probabilidades para la clase positiva
y_prob = pipe.predict_proba(X_test)[:, 1]

plt.figure(figsize=(5, 4))
sns.histplot(y_prob, bins=20, kde=True)
plt.xlabel("Probabilidad predicha de buen vino (clase 1)")
plt.ylabel("Frecuencia")
plt.title("Distribución de probabilidades predichas")
plt.tight_layout()
plt.show()

print("=== Métricas en test ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print("\nMatriz de confusión:")
print(cm)

print("\n=== Classification report ===")
print(classification_report(y_test, y_pred))

# -------------------
# EXPLICACIONES
# -------------------
"""
Umbral de calidad (quality >= 6):
    - La decisión de transformar quality en binaria con good_wine = (quality >= 6) es clave, porque 
      convierte un problema originalmente ordinal/multiclase en clasificación binaria.
      
      Conviene explicar por qué 6: por ejemplo, que en la distribución de quality los vinos con 6–8 
      se consideran aceptables/buenos, o que ese umbral equilibra mejor el número de clases 0 y 1.

Train-test split estratificado:
    - La opción stratify=y en train_test_split asegura que la proporción de vinos buenos/no buenos en train 
      y test sea similar a la del dataset completo.​

    - Esto es importante si una de las clases es minoritaria, porque evita que el conjunto de test quede 
      desbalanceado y las métricas sean engañosas.

StandardScaler + LogisticRegression en Pipeline:
El Pipeline con StandardScaler antes de la regresión logística merece una mención explícita:

    - El escalado pone todas las variables fisicoquímicas en una escala comparable, lo que mejora la 
      estabilidad numérica y la convergencia del algoritmo.​

    - El pipeline garantiza que el mismo escalado aprendido en train se aplique a test, evitando fugas 
      de información.​

class_weight="balanced" en la regresión:
    - El parámetro class_weight="balanced" indica que el modelo va a penalizar más los errores en la
      clase minoritaria.​

    - En tu caso tiene sentido porque el número de vinos buenos y no buenos no es exactamente igual, 
      y esto ayuda a mejorar métricas como recall/F1 de la clase 1.

Uso de probabilidades y curva ROC:
    - El uso de predict_proba para obtener y_prob y luego graficar la curva ROC y calcular AUC muestra 
      que no solo miras la predicción dura (0/1), sino todo el rango de probabilidades.​
    - Esto es importante de comentar: la ROC/AUC evalúa el modelo para todos los posibles umbrales y da una 
      medida global de capacidad discriminativa.​
"""