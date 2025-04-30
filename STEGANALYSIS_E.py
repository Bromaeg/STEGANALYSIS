# Tecnologico de Monterrey - Campus Queretaro
# Miguel Angel Tena Garcia - A01709653
# STEGANALYSIS.py
#
# Este script es parte de un proyecto de análisis de imágenes para detectar esteganografía,
# ahora usando transferencia de aprendizaje con VGG16 como extractor de características.
#
# (Obtencion) Se obtuvo un set de datos de la competencia de deteccion de esteganografia ALASKA2,
# utilizando directamente dos directorios de imágenes .jpg que son iguales,
# siendo una de ellas procesada utilizando JMiPOD.
#
# Dataset: https://www.kaggle.com/competitions/alaska2-image-steganalysis/data
#
# (Preprocesado y escalamiento) Cargamos las imagenes convirtiendolas a float32 y despues normalizandolas
# dividiendo entre 255 para que los valores de los pixeles esten entre 0 y 1 sin perder la informacion
# relativa a las perturbaciones que se buscan.
#
# (Segmentacion) Separamos los datos en tres conjuntos: entrenamiento (70%), validacion (20%) y prueba (10%).
# utilizando train_test_split de sklearn para dividir las imagenes en las tres categorias.
#
# Ahora usamos VGG16 pre-entrenado para extraer características, congelando sus capas convolucionales.
# Luego añadimos un clasificador denso propio para detectar steganografía.
#

import os
import glob
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from keras import layers, models
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

print("TensorFlow version:", tf.__version__)
print("GPUs disponibles:", tf.config.list_physical_devices("GPU"))

# Configuración para evitar OOM en GPU
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Directorios de datos
dir_cover  = "Cover"
dir_jmipod = "JMiPOD"

# Listado y muestreo de rutas
cover_paths  = sorted(glob.glob(os.path.join(dir_cover,  "*.jpg")))
jmipod_paths = sorted(glob.glob(os.path.join(dir_jmipod, "*.jpg")))
num_samples  = 20000
cover_paths  = random.sample(cover_paths,  num_samples)
jmipod_paths = random.sample(jmipod_paths, num_samples)

print(f"Se encontraron {len(cover_paths)} imágenes en '{dir_cover}' y {len(jmipod_paths)} en '{dir_jmipod}'.")

# Etiquetas
cover_labels  = [0] * len(cover_paths)
jmipod_labels = [1] * len(jmipod_paths)

# Combinar y dividir
all_paths  = cover_paths  + jmipod_paths
all_labels = cover_labels + jmipod_labels

train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
    all_paths, all_labels, test_size=0.10, random_state=42, stratify=all_labels)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_val_paths, train_val_labels,
    test_size=0.22222, random_state=42, stratify=train_val_labels)

print(
    f"Conjunto de imágenes:\n"
    f"  Entrenamiento: {len(train_paths)}\n"
    f"  Validación:    {len(val_paths)}\n"
    f"  Prueba:        {len(test_paths)}"
)

# Parámetros de imagen
image_size = 256
batch_size = 12
input_shape = (image_size, image_size, 3)

# Función de carga y preprocesado
def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [image_size, image_size],
                          method=tf.image.ResizeMethod.LANCZOS5,
                          antialias=True)
    return tf.cast(img, tf.float32) / 255.0

# Crear tf.data.Dataset
def create_dataset(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: (load_image(p), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = create_dataset(train_paths, train_labels, shuffle=True)
val_ds   = create_dataset(val_paths,   val_labels)
test_ds  = create_dataset(test_paths,  test_labels)

# --- Construcción del modelo con VGG16 + cabezal personalizado ---

# 1) Cargar VGG16 sin la "cabeza" de clasificación y congelar capas
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=input_shape
)
base_model.trainable = False


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



fixed_srm = layers.Conv2D(
    filters=srm_bank.shape[-1],
    kernel_size=(5,5),
    padding="same",
    use_bias=False,
    trainable=False,
    input_shape=input_shape
)


# Pooling para extraer caracteristicas
def l2_pool(x):
    return tf.sqrt(
        tf.nn.avg_pool2d(tf.square(x), ksize=3, strides=1, padding='SAME'))


# 2) Añadir clasificadores propios
model = models.Sequential([

    # Capa de convolución con los filtros de SRM
    fixed_srm,
    layers.Lambda(l2_pool),

    # Bloque 1
    layers.Conv2D(32,  3, padding="same", activation="relu"),

    # Batchnorm para normalizar la salida de la capa convolucional
    layers.BatchNormalization(),
    layers.Conv2D(32,  3, padding="same", activation="relu"),
    

    # Bloque 2
    layers.Conv2D(64,  3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(64,  3, padding="same", activation="relu"),

    # Bloque 3
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(128,  3, padding="same", activation="relu"),

    # Clasificador denso, sigmoide para clasificacion binaria y relu para la capa oculta
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dense(1,   activation="sigmoid")

])

# Compilación
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Callbacks
checkpoint = ModelCheckpoint(
    "vgg16_stegano.h5", monitor="val_accuracy",
    save_best_only=True, verbose=1
)


# Entrenamiento
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[checkpoint]
)

# Cargar mejor modelo
best_model = tf.keras.models.load_model("vgg16_stegano.h5")

# Evaluación en test
loss, acc = best_model.evaluate(test_ds, verbose=1)
print(f"\nTest loss: {loss:.4f} — Test accuracy: {acc:.4f}\n")

# Matriz de Confusión y Reporte
y_true, y_pred = [], []
for x, y in test_ds:
    probs = best_model.predict(x).ravel()
    y_true.extend(y.numpy())
    y_pred.extend((probs >= 0.5).astype(int))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Cover","JMiPOD"],
            yticklabels=["Cover","JMiPOD"])
plt.title("Matriz de Confusión (Test)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.savefig("confusion_matrix.png")
plt.close()

print(classification_report(
    y_true, y_pred,
    target_names=["Cover", "JMiPOD"]
))
