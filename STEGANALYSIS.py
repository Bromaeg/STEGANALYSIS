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

# Obtener rutas para cada clase
cover_paths = sorted(glob.glob(os.path.join(dir_cover, "*.jpg")))
jmipod_paths = sorted(glob.glob(os.path.join(dir_jmipod, "*.jpg")))

# Número de muestras
num_samples = 10000  
cover_paths = random.sample(cover_paths, num_samples)
jmipod_paths = random.sample(jmipod_paths, num_samples)

print(
    f"Se encontraron {len(cover_paths)} imágenes en '{dir_cover}' y {len(jmipod_paths)} en '{dir_jmipod}'."
)

# Asignar etiquetas
cover_labels = [0] * len(cover_paths)
jmipod_labels = [1] * len(jmipod_paths)

# Combinar las listas
all_paths = cover_paths + jmipod_paths
all_labels = cover_labels + jmipod_labels

# Dividir los datos
train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
    all_paths, all_labels, test_size=0.10, random_state=42, stratify=all_labels
)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_val_paths,
    train_val_labels,
    test_size=0.22222,
    random_state=42,
    stratify=train_val_labels,
)

print(
    f"Conjunto de imágenes:\n  Entrenamiento: {len(train_paths)}\n  Validación: {len(val_paths)}\n  Prueba: {len(test_paths)}"
)


# Pipeline de datos
def load_image(filename):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image, [512, 512])
    image = tf.cast(image, tf.float32)
    image /= 255.0
    return image


def create_dataset(paths, labels, batch_size=8, shuffle=False, shuffle_buffer=1000):
    # Convertir listas a tensors
    paths_tensor = tf.constant(paths)
    labels_tensor = tf.constant(labels)

    # Crear dataset de tensors
    ds = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))
    ds = ds.map(lambda x, y: (load_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

# Tamaño del batch
batch_size = 20 

# Crear los datasets
train_ds = create_dataset(train_paths, train_labels, batch_size, shuffle=True)
val_ds = create_dataset(val_paths, val_labels, batch_size)
test_ds = create_dataset(test_paths, test_labels, batch_size)
input_shape = (512, 512, 3) 


# Seleccionar dispositivo
strategy = (
    tf.distribute.MirroredStrategy() if len(gpus) > 1 else tf.distribute.get_strategy()
)

with strategy.scope():
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            # Bloque 1:
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Bloque 2
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Bloque 3
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Capa final
            layers.GlobalAveragePooling2D(), 

            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    # Compilación del modelo
    model.compile(
        optimizer='adam',
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

model.summary()


# Callback para detener el entrenamiento si la memoria está por agotarse
class MemoryCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):

        # Forzar liberación de memoria
        tf.keras.backend.clear_session()


# Checkpoint para guardar el mejor modelo en h5
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    "model.h5",
    monitor="val_loss",
    save_best_only=True,
    verbose =1)

# Entrenar con callbacks
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,  
    callbacks=[
        MemoryCallback(),
        checkpoint_callback,
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    ],
)

# Evaluación en lotes para evitar OOM
y_true = []
y_pred = []

# Predicción por lotes
for x_batch, y_batch in test_ds:
    y_batch_pred = model.predict(x_batch)
    y_true.extend(y_batch.numpy())
    y_pred.extend(y_batch_pred.flatten())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_pred_classes = np.round(y_pred)

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Cover", "JMiPOD"],
    yticklabels=["Cover", "JMiPOD"],
)
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.savefig("confusion_matrix.png")  # Guardar en lugar de mostrar
plt.close()

# Reporte de clasificación
print(classification_report(y_true, y_pred_classes, target_names=["Cover", "JMiPOD"]))
