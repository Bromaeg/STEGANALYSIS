# Tecnologico de Monterrey - Campus Queretaro
# Miguel Angel Tena Garcia - A01709653
# evaluacion_modelo.py
#
# Este script reconstruye el test set, recrea la arquitectura exacta
# y carga sólo los pesos de model.h5 para evitar el error de marshal.

import os, glob, random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras import layers, models

print("TensorFlow version:", tf.__version__)
print("GPUs disponibles:", tf.config.list_physical_devices("GPU"))

# --- 1) Reconstruir rutas y etiquetas con la misma semilla ---
random.seed(42)
dir_cover, dir_jmipod = "Cover", "JMiPOD"

cover_paths  = sorted(glob.glob(os.path.join(dir_cover,  "*.jpg")))
jmipod_paths = sorted(glob.glob(os.path.join(dir_jmipod, "*.jpg")))

num_samples = 25000
cover_paths  = random.sample(cover_paths,  num_samples)
jmipod_paths = random.sample(jmipod_paths, num_samples)

cover_labels  = [0]*num_samples
jmipod_labels = [1]*num_samples

all_paths  = cover_paths + jmipod_paths
all_labels = cover_labels + jmipod_labels

# mismo split train/val/test que en STEGANALYSIS.py
_, test_paths, _, test_labels = train_test_split(
    all_paths, all_labels, test_size=0.10,
    random_state=42, stratify=all_labels
)

print(f"Núm. de samples de test = {len(test_paths)}")

# --- 2) Crear sólo el test_ds ---
image_size = 300   # idéntico al entrenamiento
batch_size = 10

def load_image(fn):
    img = tf.io.read_file(fn)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img,
        [image_size, image_size],
        method=tf.image.ResizeMethod.LANCZOS5,
        antialias=True
    )
    return tf.cast(img, tf.float32) / 255.0

def create_dataset(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda x,y: (load_image(x), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_ds = create_dataset(test_paths, test_labels)

# --- 3) Reconstruir EXACTAMENTE la arquitectura ---
#   a) L2‐pool como función “named” (no lambda)
def l2_pool(x):
    return tf.sqrt(
        tf.nn.avg_pool2d(tf.square(x),
                         ksize=3, strides=1, padding='SAME')
    )

input_shape = (image_size, image_size, 3)

# 3b) Banco SRM (idéntico al tuyo: 3 kernels)
k1 = np.array([[0,0,0,0,0],
               [0,-1,2,-1,0],
               [0,2,-4,2,0],
               [0,-1,2,-1,0],
               [0,0,0,0,0]],dtype=np.float32)/4
k2 = np.array([[-1,2,-2,2,-1],
               [2,-6,8,-6,2],
               [-2,8,-12,8,-2],
               [2,-6,8,-6,2],
               [-1,2,-2,2,-1]],dtype=np.float32)/12
k3 = np.array([[0,0,0,0,0],
               [0,1,-2,1,0],
               [0,-2,4,-2,0],
               [0,1,-2,1,0],
               [0,0,0,0,0]],dtype=np.float32)/4

srm_bank = np.stack([k1,k2,k3],axis=-1)     # (5,5,3)
srm_bank = np.stack([srm_bank]*3,axis=-2)   # (5,5,3,3)

# Construir modelo secuencial:
model = models.Sequential([
    # capa fija SRM
    layers.Conv2D(
        filters=3, kernel_size=5, padding="same",
        use_bias=False, trainable=False,
        input_shape=input_shape,
        name="fixed_srm"
    ),
    layers.Lambda(l2_pool, name="l2_pool"),

    # Bloque 1
    layers.Conv2D(32,3,padding="same",activation="relu", name="c1"),
    layers.BatchNormalization(name="bn1"),
    layers.Conv2D(32,3,padding="same",activation="relu", name="c1b"),
    layers.Dropout(0.25, name="d1"),

    # Bloque 2
    layers.Conv2D(64,3,padding="same",activation="relu", name="c2"),
    layers.BatchNormalization(name="bn2"),
    layers.Conv2D(64,3,padding="same",activation="relu", name="c2b"),
    layers.Dropout(0.25, name="d2"),

    # Bloque 3
    layers.Conv2D(128,3,padding="same",activation="relu", name="c3"),
    layers.BatchNormalization(name="bn3"),
    layers.Conv2D(128,3,padding="same",activation="relu", name="c3b"),
    layers.Dropout(0.20, name="d3"),

    # Clasificador
    layers.GlobalAveragePooling2D(name="gap"),
    layers.Dense(128,activation="relu",name="fc1"),
    layers.Dense(1,activation="sigmoid",name="out")
])

# 3c) Cargar pesos fijos SRM en la primera capa
model.get_layer("fixed_srm").set_weights([srm_bank])

# Compilar
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# 4) Cargar pesos (¡sólo pesos, no config!)
model.load_weights("model.h5")

# 5) Evaluar
loss, acc = model.evaluate(test_ds, verbose=1)
print(f"\nTest loss: {loss:.4f} — Test acc: {acc:.4f}\n")

# 6) Matriz de confusión + reporte
y_true, y_pred = [], []
for x,y in test_ds:
    p = model(x, training=False).numpy().flatten()
    y_true.extend(y.numpy())
    y_pred.extend(p)

y_true        = np.array(y_true)
y_pred_labels = np.round(np.array(y_pred))

cm = confusion_matrix(y_true, y_pred_labels)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Cover","JMiPOD"],
            yticklabels=["Cover","JMiPOD"])
plt.title("Matriz de Confusión (Test)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.savefig("confusion_matrix_test.png")
plt.close()

print(classification_report(y_true, y_pred_labels,
      target_names=["Cover","JMiPOD"]))
