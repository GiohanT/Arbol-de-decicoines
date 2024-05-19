import pandas as pd
import numpy as np

# Configuración de la semilla aleatoria para reproducibilidad
np.random.seed(0)

# Generación de datos
num_samples = 1000
humedad_suelo = np.random.uniform(10, 100, num_samples)  # Humedad entre 10% y 100%
temperatura_zona = np.random.uniform(-10, 30, num_samples)  # Temperatura entre -10°C y 5°C

# Función para determinar si hay helada
def helada(humedad, temp):
    if temp < 0 or (humedad > 80 and temp < 1):
        return 'Si'
    else:
        return "No"

# Crear columna de heladas
heladas = [helada(h, t) for h, t in zip(humedad_suelo, temperatura_zona)]

# Crear DataFrame
df = pd.DataFrame({
    'Humedad del Suelo (%)': humedad_suelo,
    'Temperatura de la Zona (°C)': temperatura_zona,
    'Helada de Suelo (Sí/No)': heladas
})

# Muestra del DataFrame
print(df.head())

# Opcional: guardar el DataFrame en un archivo CSV
df.to_csv('dataset_heladas_suelo.csv', index=False)
