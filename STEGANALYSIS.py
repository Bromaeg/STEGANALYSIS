# Tecnologico de Monterrey - Campus Queretaro
# Miguel Angel Tena Garcia - A01709653
# STEGANALYSIS.py
#
# Este script es parte de un proyecto de análisis de imágenes para detectar esteganografía.
#
# (Obtencion) Se obtuvo un set de datos de la competencia de deteccion de esteganografia ALASKA2,
# utilizando directamente dos directorios de imágenes .jpg que son iguales,
# siendo una de ellas procesada utilizando JMiPOD.
# 
# Dataset: https://www.kaggle.com/competitions/alaska2-image-steganalysis/data
# 
# En este caso, solo se utilizo el dataset sin aumentarle o generar datos ya que podria interferir 
# con las sutiles señales de esteganografía que se buscan.
#
# (Preprocesado y escalamiento) Cargamos las imagenes convirtiendolas a float32 y despues normalizandolas
# dividiendo entre 255 para que los valores de los pixeles esten entre 0 y 1 sin perder la informacion 
# relativa a las perturbaciones que se buscan.
#
# (Segmentacion) Separamos los datos en tres conjuntos: entrenamiento (70%), validacion (20%) y prueba (10%).
# utilizando train_test_split de sklearn para dividir las imagenes en las tres categorias.
#
# El objetivo es que la red neuronal aprenda a detectar si una imagen esconde 
# información o no con esteganografía.
#
#

import os
import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# Dataset
# Directorios de las imágenes
dir_cover = 'Cover'
dir_jmipod = 'JMiPOD'

# Obtener rutas para cada clase
cover_paths = sorted(glob.glob(os.path.join(dir_cover, '*.jpg')))
jmipod_paths = sorted(glob.glob(os.path.join(dir_jmipod, '*.jpg')))

# Print para confirmar que se cargaron las imágenes correctamente
print(f"Se encontraron {len(cover_paths)} imágenes en '{dir_cover}' y {len(jmipod_paths)} en '{dir_jmipod}'.")

import random
num_samples = 10000
cover_paths = random.sample(cover_paths, num_samples)
jmipod_paths = random.sample(jmipod_paths, num_samples)

# Asignar etiquetas: 0 para Cover (original) y 1 para JMiPOD (procesada)
cover_labels = [0] * len(cover_paths)
jmipod_labels = [1] * len(jmipod_paths)

# Combinar las listas de rutas y etiquetas
all_paths = cover_paths + jmipod_paths
all_labels = cover_labels + jmipod_labels

# Dividir los datos en entrenamiento (70%), validación (20%) y prueba (10%)
train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
    all_paths, all_labels, test_size=0.10, random_state=42, stratify=all_labels)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_val_paths, train_val_labels, test_size=0.22222, random_state=42, stratify=train_val_labels)


# Validacion de separacion de datos
print(f"Conjunto de imágenes:\n  Entrenamiento: {len(train_paths)}\n  Validación: {len(val_paths)}\n  Prueba: {len(test_paths)}")

# Pipeline de datos
def load_image(filename):

    # Carga la imagen sin redimensionarla para preservar los mensajes de la esteganografía.
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image, [512, 512])

    image = tf.cast(image, tf.float32)

    # Normalización: escalar a [0, 1] preservando las relaciones entre valores.
    image /= 255.0
    return image

def create_dataset(paths, labels, batch_size=8, shuffle=False, shuffle_buffer=1000):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    # Cargamos la imagen y conservamos la ruta
    ds = ds.map(lambda x, y: (load_image(x), y, x), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_optimization.apply_default_optimizations = True
    ds = ds.with_options(options)
    return ds

# Tamaño del batch
batch_size = 20

# Crear los datasets entrenamiento, validación y prueba
train_ds = create_dataset(train_paths, train_labels, batch_size, shuffle=True, shuffle_buffer=1000)
val_ds = create_dataset(val_paths, val_labels, batch_size, shuffle=False)
test_ds = create_dataset(test_paths, test_labels, batch_size, shuffle=False)

train_ds_model = train_ds.map(lambda image, label, path: (image, label))
val_ds_model = val_ds.map(lambda image, label, path: (image, label))
test_ds_model = test_ds.map(lambda image, label, path: (image, label))

train_ds_model = train_ds_model.apply(tf.data.experimental.prefetch_to_device('/GPU:1'))
val_ds_model = val_ds_model.apply(tf.data.experimental.prefetch_to_device('/GPU:1'))
test_ds_model = test_ds_model.apply(tf.data.experimental.prefetch_to_device('/GPU:1'))

# DEV mostrar la forma de un lote, del conjunto de entrenamiento

# for images, labels, paths in train_ds.take(1):
#     print("Forma del batch de imágenes:", images.shape)
#     print("Forma del batch de etiquetas:", labels.shape)
#     print("Batch de rutas:", paths.shape)

# # Prueba con una imagen aleatoria del entrenamiento 
# for images, labels, paths in train_ds.take(1):
#     idx = random.randint(0, images.shape[0] - 1)
#     chosen_image = images[idx]
#     chosen_label = labels[idx].numpy() 
#     chosen_path = paths[idx].numpy().decode('utf-8')  

#     print("Índice en el batch:", idx)
#     print("Etiqueta asignada:", chosen_label)
#     print("Ruta del archivo:", chosen_path)

#     # Mostramos el label
#     plt.imshow((chosen_image.numpy() * 255).astype("uint8"))
#     plt.title(f"Etiqueta: {chosen_label} | Archivo: {os.path.basename(chosen_path)}")
#     plt.axis("off")
#     plt.show()

# Modelo de aprendizaje 

# Imagen de entrada con tamaño variable y 3 canales (RGB)
input_shape = (512, 512, 3)  

model = models.Sequential([
    layers.Input(shape=input_shape),

    # Capas convolucionales y de agrupamiento
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  
])

# Compilación del modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Entrenamiento del modelo
history = model.fit(
    train_ds_model,
    validation_data=val_ds_model,
    epochs=3,
    batch_size=batch_size
)


# Matriz de confusión y reporte de clasificación
y_pred = model.predict(test_ds)
y_pred_classes = np.round(y_pred).flatten()
y_true = np.concatenate([y for _, y, _ in test_ds], axis=0).numpy()
y_true_classes = y_true.flatten()

# Matriz de confusión
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cover', 'JMiPOD'], yticklabels=['Cover', 'JMiPOD'])
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()




