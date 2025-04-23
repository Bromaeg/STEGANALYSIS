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
from keras import layers, models
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print("TensorFlow version:", tf.__version__)
print("GPUs disponibles:", tf.config.list_physical_devices("GPU"))

# Configuración para evitar OOM
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Limitar el uso de memoria GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Dataset
dir_cover = "Cover"
dir_jmipod = "JMiPOD"

# Sorted para asegurar que las imagenes esten en el mismo orden
cover_paths  = sorted(glob.glob(os.path.join(dir_cover,  "*.jpg")))
jmipod_paths = sorted(glob.glob(os.path.join(dir_jmipod, "*.jpg")))

# Número de muestras
num_samples   = 15000
cover_paths   = random.sample(cover_paths,   num_samples)
jmipod_paths  = random.sample(jmipod_paths,  num_samples)

print(f"Se encontraron {len(cover_paths)} imágenes en '{dir_cover}' y {len(jmipod_paths)} en '{dir_jmipod}'.")

# Asignar etiquetas
cover_labels  = [0] * len(cover_paths)
jmipod_labels = [1] * len(jmipod_paths)

# Combinar las listas
all_paths  = cover_paths  + jmipod_paths
all_labels = cover_labels + jmipod_labels

# Dividir los datos
train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
    all_paths, all_labels, test_size=0.10, random_state=42, stratify=all_labels)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_val_paths, train_val_labels,
    test_size=0.22222, random_state=42, stratify=train_val_labels)

print(
    f"Conjunto de imágenes:\n  Entrenamiento: {len(train_paths)}\n  Validación: {len(val_paths)}\n  Prueba: {len(test_paths)}"
)

# Reduccion tamaño de escalado
image_size = 256

# Pipeline de datos
def load_image(filename):
    image  = tf.io.read_file(filename)
    image  = tf.image.decode_jpeg(image, channels=3)

    # Convertir a float32 y normalizar, tambien se escala a 256x256
    image  = tf.image.resize( image, [image_size, image_size],
                              method=tf.image.ResizeMethod.LANCZOS5,
                              antialias=True)
    image  = tf.cast(image, tf.float32) / 255.0
    return image

# Funcion crear dataset, ruta imagenes y etiquetas
def create_dataset(paths, labels, batch_size, shuffle=False, shuffle_buffer=1000):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda x, y: (load_image(x), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# Crear los datasets
batch_size = 8
train_ds = create_dataset(train_paths, train_labels, batch_size, shuffle=True)
val_ds   = create_dataset(val_paths,   val_labels,   batch_size)
test_ds  = create_dataset(test_paths,  test_labels,  batch_size)
input_shape = (image_size, image_size, 3)



# Defincion del filtrado con banco kernels (state of the art) 
# solo use 3 kernels en vez de 30 por restricciones de memoria
k1 = np.array([[0,  0,  0,  0, 0],
               [0, -1,  2, -1, 0],
               [0,  2, -4,  2, 0],
               [0, -1,  2, -1, 0],
               [0,  0,  0,  0, 0]], dtype=np.float32) / 4

k2 = np.array([[ -1,  2,  -2,  2, -1],
               [  2, -6,   8, -6,  2],
               [ -2,  8, -12,  8, -2],
               [  2, -6,   8, -6,  2],
               [ -1,  2,  -2,  2, -1]], dtype=np.float32) / 12

k3 = np.array([[0,  0,  0,  0, 0],
               [0,  1, -2,  1, 0],
               [0, -2,  4, -2, 0],
               [0,  1, -2,  1, 0],
               [0,  0,  0,  0, 0]], dtype=np.float32) / 4

srm_list = [k1, k2, k3]  
srm_bank = np.stack(srm_list, axis=-1)              
srm_bank = np.stack([srm_bank]*3, axis=-2) 



# Construcción del modelo, usando el banco srm como primer bloque
fixed_srm = layers.Conv2D(
    filters=srm_bank.shape[-1],
    kernel_size=(5,5),
    padding="same",
    use_bias=False,
    trainable=False,
    input_shape=input_shape
)

# Pooling para extraer caracteristicas y reducir dimensionalidad
def l2_pool(x):
    return tf.sqrt(
        tf.nn.avg_pool2d(tf.square(x), ksize=3, strides=1, padding='SAME'))

# Definimos el modelo secuencial
# 3 Bloques de convolucion y pooling, seguido de un clasificador denso
model = models.Sequential([
    fixed_srm,
    layers.Lambda(l2_pool),

    # Bloque 1
    layers.Conv2D(32,  3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(32,  3, padding="same", activation="relu"),
    layers.Dropout(0.25),

    # Bloque 2
    layers.Conv2D(64,  3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(64,  3, padding="same", activation="relu"),
    layers.Dropout(0.25),

    # Bloque 3
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(128,  3, padding="same", activation="relu"),
    layers.Dropout(0.20),

    # Clasificador denso, sigmoide para clasificacion binaria y relu para la capa oculta
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dense(1,   activation="sigmoid")

])

# Cargamos los pesos fijos para el primer bloque 
model.layers[0].set_weights([srm_bank])

# Compilación del modelo 
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Callbacks para conservar el mejor modelo y detener el entrenamiento si no mejora
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    "model.h5", monitor="val_loss", save_weights_only=False,
    save_best_only=True, verbose=1,)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=7,
    restore_best_weights=True, verbose=1)

# Entrenamiento, 5 epocas por practicidad
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=[checkpoint_callback, early_stopping])

# Evaluación por lotes para optimizar memoria
y_true = []
y_pred = []


# Evaluacion del modelo con el conjunto de prueba (aqui truena por la GPU)
for x_batch, y_batch in test_ds:
    y_batch_pred = model(x_batch, training=False)
    y_true.extend(y_batch.numpy())
    y_pred.extend(y_batch_pred.numpy().flatten())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_pred_classes = np.round(y_pred)

# Matriz de Confusión 
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Cover","JMiPOD"],
            yticklabels=["Cover","JMiPOD"])
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.savefig("confusion_matrix.png")
plt.close()

plt.figure(figsize=(10,5))
plt.plot(history.history["accuracy"], label="Entrenamiento")
plt.plot(history.history["val_accuracy"], label="Validación")
plt.title("Precisión del Modelo")

plt.figure(figsize=(10,5))
plt.plot(history.history["loss"], label="Entrenamiento")
plt.plot(history.history["val_loss"], label="Validación")
plt.title("Pérdida del Modelo")

# Reporte
print(classification_report(
    y_true, y_pred_classes,
    target_names=["Cover","JMiPOD"]
))
