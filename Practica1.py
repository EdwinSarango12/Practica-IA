
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Crear el dataset
data = {
    'HorasEstudio': [10, 2, 8, 4, 9, 1, 7, 6, 5, 3],
    'ConocimientoPrevio': [7, 2, 6, 3, 8, 1, 5, 4, 6, 2],
    'Asistencia': [90, 40, 85, 60, 95, 30, 80, 75, 65, 50],
    'PromedioTareas': [8, 4, 7, 5, 9, 3, 6, 7, 6, 4],
    'EstudiaSolo': [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
    'Resultado': [1, 0, 1, 0, 1, 0, 1, 1, 1, 0]
}

df = pd.DataFrame(data)

# 2. Separar características y etiquetas
X = df.drop('Resultado', axis=1)
y = df['Resultado']

# 3. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Entrenar el modelo
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)

# 5. Predecir y evaluar
y_pred = modelo.predict(X_test)
print("Precisión:", accuracy_score(y_test, y_pred))
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred))

# 6. Hacer una predicción con un nuevo estudiante
nuevo_estudiante = pd.DataFrame([{
    'HorasEstudio': 6,
    'ConocimientoPrevio': 5,
    'Asistencia': 80,
    'PromedioTareas': 7,
    'EstudiaSolo': 0
}])

resultado = modelo.predict(nuevo_estudiante)
print("Resultado nuevo estudiante:", "Aprobado" if resultado[0] == 1 else "No Aprobado")

