# Clasificador de Perros y Gatos

Este proyecto implementa un clasificador de imágenes para distinguir entre perros y gatos usando redes neuronales con TensorFlow y Keras. Se utilizan tres modelos diferentes: un modelo denso, un modelo convolucional simple y un modelo convolucional con dropout.

## Contenido
- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura del Código](#estructura-del-código)
- [Entrenamiento de Modelos](#entrenamiento-de-modelos)
- [Resultados](#resultados)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

## Instalación
Para ejecutar este proyecto, asegúrate de tener instalados los siguientes paquetes:

- TensorFlow
- TensorFlow Datasets
- OpenCV
- Matplotlib
- NumPy

Puedes instalar las dependencias utilizando pip:

```bash
pip install tensorflow tensorflow-datasets opencv-python matplotlib numpy
```

## Uso
Para ejecutar el proyecto, sigue estos pasos:

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu_usuario/clasificador-perros-gatos.git
   cd clasificador-perros-gatos
   ```

2. Ejecuta el script principal:
   ```bash
   python clasificador.py
   ```

## Estructura del Código
El código se divide en las siguientes secciones:

- **Importar librerías y datos**: Se importa el dataset `cats_vs_dogs` de TensorFlow Datasets y se visualizan algunas imágenes.
- **Preprocesamiento de Datos**: Se redimensionan las imágenes, se convierten a escala de grises y se normalizan.
- **Definición de Modelos**: Se definen tres modelos: uno denso, un convolucional simple y otro con dropout.
- **Compilación y Entrenamiento de Modelos**: Se compilan y entrenan los modelos utilizando diferentes configuraciones.

## Entrenamiento de Modelos
Se compilan y entrenan tres modelos:

### Modelo Denso:
```python
modeloDenso = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(100, 100, 1)),
    tf.keras.layers.Dense(150, activation='relu'),
    tf.keras.layers.Dense(150, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### Modelo Convolucional:
```python
modeloCNN = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### Modelo Convolucional con Dropout:
```python
modeloCNN2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(250, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

## Resultados
Los modelos se entrenan y evalúan utilizando TensorBoard para visualizar las métricas de rendimiento. Se utilizan 100 épocas y un batch size de 32, con un 15% de los datos para validación.

## Contribuciones
¡Las contribuciones son bienvenidas! Si tienes alguna sugerencia o encuentras algún problema, por favor abre un issue o realiza un pull request.

## Estudiantes
Andres Hurtado
Daniel Almarza
Gerardo Estevez
Jeferson Ayala

## Licencia
Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.
