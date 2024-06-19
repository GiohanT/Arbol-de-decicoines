import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Cargar el modelo entrenado
modelo = joblib.load('./Modelos AI/modelo_arbol_decision2.pkl')

# Cargar los datos de prueba desde el archivo CSV
datos_prueba = pd.read_csv('./Data_set/Prueba.csv')

# Suponiendo que los datos de prueba contienen características similares a las que se usaron para entrenar el modelo
# Si las columnas del CSV son diferentes, necesitarás preprocesar los datos para que coincidan con las características utilizadas durante el entrenamiento.

# Hacer predicciones en los datos de prueba
predicciones = modelo.predict(datos_prueba)

# Agregar las predicciones a los datos de prueba como una nueva columna
datos_prueba['predicciones'] = predicciones
#datos_prueba['predicciones'] = datos_prueba['predicciones'].astype(str({ '0': 'no', '1': 'si'}))

# Guardar los datos con predicciones en un nuevo archivo CSV
datos_prueba.to_csv('Resultados2.csv', index=False)
