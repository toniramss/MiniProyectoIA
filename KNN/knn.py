import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv("wine_quality_merged.csv")

#print(df.head())

print("\n-----Cantidad de vinos por calidad-----")
print(df["quality"].value_counts().sort_index())



# Crear variable binaria
# Calidad alta 1 (6, 7, 8, 9) - Calidad baja 0 (3, 4, 5)
df["quality_bin"] = df["quality"].apply(lambda x: 1 if x >= 6 else 0)
print("\nVinos con calidad binaria (0=baja, 1=alta):")
print(df[["quality", "quality_bin"]].head())

# Separar características y variable objetivo
X = df.drop(["quality", "quality_bin"], axis=1)
y = df["quality_bin"]

print("\n-----Cantidad de vinos por calidad binaria-----")
print(df["quality_bin"].value_counts().sort_index())

#print(X.head())
#print(y.head())

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#print(X_scaled[:5])

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# KNN con k=17, es el valor que mejor accuracy da
knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
#print(cm)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de confusión - KNN binario")
plt.show()

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))


"""El modelo KNN que he hecho para predecir si un vino es bueno o malo me ha 
dado una accuracy de alrededor del 75%. Esto quiere decir que acierta más o 
menos 3 de cada 4 veces, lo cual está bastante bien. Viendo la matriz de confusión, 
el modelo acierta sobre todo con los vinos buenos, porque en el dataset hay muchos 
más vinos de calidad alta que baja. Por eso también se equivoca más cuando tiene 
que detectar vinos malos. Aun así, el resultado es bastante sólido y mejora mucho 
comparado con cuando intentábamos predecir todas las calidades por separado."""




# Buscar mejor k
acc_list = []

for k in range(1, 41):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc_list.append(accuracy_score(y_test, y_pred))

plt.figure(figsize=(8,4))
plt.plot(range(1, 41), acc_list, marker="o")
plt.xlabel("Número de vecinos (k)")
plt.ylabel("Accuracy")
plt.title("Precisión del modelo KNN binario según k")
plt.grid(True)
plt.show()