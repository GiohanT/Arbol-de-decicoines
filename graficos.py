import pandas as pd
import matplotlib.pyplot as plt

# Leer el archivo CSV
df = pd.read_csv('./Resultados.csv')

# Generar un gráfico de barras
df.plot(x='Humedad del Suelo (%)', y='Temperatura de la Zona (°C)', kind='bar')
plt.title('Ventas anuales')
plt.xlabel('Humedad del Suelo (%)')
plt.ylabel('Temperatura de la Zona (°C)')
plt.grid(True)

# Mostrar el gráfico
plt.savefig('grafico_ventas.png')

