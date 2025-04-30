# Test para probar con el data set de RESERVADAS 
# Las primeras 4 imagenes son STEGO la 5 no lo es.

import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt

# reconstrucción exacta de la arquitectura sin lambdas serializadas
def l2_pool(x):
    return tf.sqrt(
        tf.nn.avg_pool2d(tf.square(x), ksize=3, strides=1, padding='SAME')
    )

image_size = 300
input_shape = (image_size, image_size, 3)

# SRM bank de 3 kernels
k1 = np.array([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]],dtype=np.float32)/4
k2 = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],dtype=np.float32)/12
k3 = np.array([[0,0,0,0,0],[0,1,-2,1,0],[0,-2,4,-2,0],[0,1,-2,1,0],[0,0,0,0,0]],dtype=np.float32)/4
srm = np.stack([k1,k2,k3],axis=-1)      
srm = np.stack([srm]*3,axis=-2)         

# Reconstrucción de arquitectura para solo cargar pesos
model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(3,5,padding="same",use_bias=False,trainable=False, name="fixed_srm"),
    layers.Lambda(l2_pool,name="l2_pool"),
    layers.Conv2D(32,3,padding="same",activation="relu",name="c1"),
    layers.BatchNormalization(name="bn1"),
    layers.Conv2D(32,3,padding="same",activation="relu",name="c1b"),
    layers.Dropout(0.25,name="d1"),
    layers.Conv2D(64,3,padding="same",activation="relu",name="c2"),
    layers.BatchNormalization(name="bn2"),
    layers.Conv2D(64,3,padding="same",activation="relu",name="c2b"),
    layers.Dropout(0.25,name="d2"),
    layers.Conv2D(128,3,padding="same",activation="relu",name="c3"),
    layers.BatchNormalization(name="bn3"),
    layers.Conv2D(128,3,padding="same",activation="relu",name="c3b"),
    layers.Dropout(0.20,name="d3"),
    layers.GlobalAveragePooling2D(name="gap"),
    layers.Dense(128,activation="relu",name="fc1"),
    layers.Dense(1,activation="sigmoid",name="out")
])

# Cargar pesos fijos SRM y del modelo entrenado
model.get_layer("fixed_srm").set_weights([srm])
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.load_weights("vgg16_stegano.h5")

def classify(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size,image_size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)/255.0
    arr = arr[None,...]
    p = float(model.predict(arr, verbose=0)[0,0])
    return "JMiPOD" if p>=0.5 else "Cover", p

test_images = [
    "RESERVADAS/80001.jpg", "RESERVADAS/80002.jpg",
    "RESERVADAS/80003.jpg", "RESERVADAS/80004.jpg",
    "RESERVADAS/80005.jpg"
]

labels, probs = [], []
for fn in test_images:
    lbl, p = classify(fn)
    labels.append(lbl)
    probs.append(p)
    print(f"{fn:40s} → {p:.4f}  {lbl}")

# Plot de barras
x = np.arange(len(test_images))
colors = ["C0" if lbl=="Cover" else "C1" for lbl in labels]

plt.figure(figsize=(8,4))
bars = plt.bar(x, probs, color=colors)
plt.ylim(0,1)
plt.xticks(x, [f"{i+1}" for i in x], rotation=0)
plt.ylabel("Probabilidad de JMiPOD")
plt.xlabel("Imagen #")

# Añadir etiquetas encima de cada barra
for bar, p, lbl in zip(bars, probs, labels):
    plt.text(bar.get_x() + bar.get_width()/2, p + 0.02,
             f"{p:.2f}\n{lbl}", ha="center", va="bottom", fontsize=9)

plt.title("Clasificación de imágenes individuales")
plt.tight_layout()
plt.show()
