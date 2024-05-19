# Importa las bibliotecas necesarias
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Carga los datos desde el archivo CSV
data = pd.read_csv("./Data_set/dataset_heladas_suelo.csv")

# Crea un objeto LabelEncoder
label_encoder = LabelEncoder()

# Itera sobre todas las columnas del DataFrame
for column in data.columns:
    # Si el tipo de dato es 'object' (es decir, string), aplica la codificación de etiquetas
    if data[column].dtype == 'Helada de Suelo (Sí/No)':
        data[column] = label_encoder.fit_transform(data[column])

# Divide los datos en características (X) y etiquetas (y)
X = data.drop(columns=["Humedad del Suelo (%)","Temperatura de la Zona (°C)"])  # Ajusta "target_column" al nombre de tu columna de etiquetas
y = data["Helada de Suelo (Sí/No)"]  # Ajusta "target_column" al nombre de tu columna de etiquetas

# División de datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo de árbol de decisiones
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
# Predicción en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
joblib.dump(model, 'modelo_arbol_decision.pkl',protocol=4)