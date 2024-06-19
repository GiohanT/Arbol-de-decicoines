import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Cargar los datos desde el archivo CSV
datos = pd.read_csv('./Data_set/dataset_heladas_suelo.csv')

# Convertir la columna 'Helada de Suelo' a valores numéricos (0 para 'No' y 1 para 'Sí')
datos['Helada de Suelo (Sí/No)'] = datos['Helada de Suelo (Sí/No)'].map({'No': 0, 'Si': 1})

# Dividir los datos en características (X) y etiquetas (y)
X = datos[['Humedad del Suelo (%)', 'Temperatura de la Zona (°C)']]  # Seleccionar solo las columnas de características
y = datos['Helada de Suelo (Sí/No)']  # Seleccionar la columna de etiquetas


# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de árbol de decisiones
modelo = DecisionTreeClassifier()

# Entrenar el modelo
modelo.fit(X_train, y_train)

# Predecir sobre el conjunto de prueba
predicciones = modelo.predict(X_test)

# Calcular la precisión del modelo
precision = accuracy_score(y_test, predicciones)
print("Precisión del modelo:", precision)

joblib.dump(modelo, 'modelo_arbol_decision2.pkl',protocol=4)