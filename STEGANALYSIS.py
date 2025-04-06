# Miguel Angel Tena Garcia - A01709653
#
# Red neuronal que analizará y aprenderá de dos sets de imágenes .jpg que son iguales,
# pero una de ellas fue procesada con esteganografía utilizando JMiPOD.
# 
# Ademas se utiliza un dataset en carpetas independientes de imágenes originales (Cover) y otro de imágenes procesadas (JMiPOD).
#
# Las imagenes son cargadas y preprocesadas, convirtiendolas a float32 y luego normalizandolas a [0, 1].
#
# El objetivo es que la red neuronal aprenda a detectar si una imagen esconde información o no con esteganografía.

import os
import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt


# Dataset
# Directorios de las imágenes
dir_cover = 'Cover'
dir_jmipod = 'JMiPOD'

# Obtener la lista de rutas para cada clase
cover_paths = sorted(glob.glob(os.path.join(dir_cover, '*.jpg')))
jmipod_paths = sorted(glob.glob(os.path.join(dir_jmipod, '*.jpg')))

print(f"Se encontraron {len(cover_paths)} imágenes en '{dir_cover}' y {len(jmipod_paths)} en '{dir_jmipod}'.")

# Asignar etiquetas: 0 para Cover (original) y 1 para JMiPOD (procesada)
cover_labels = [0] * len(cover_paths)
jmipod_labels = [1] * len(jmipod_paths)

# Combinar las listas de rutas y etiquetas
all_paths = cover_paths + jmipod_paths
all_labels = cover_labels + jmipod_labels

print(f"Total de imágenes combinadas: {len(all_paths)}")

# Dividir los datos en entrenamiento (64%), validación (16%) y prueba (20%)
train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
    all_paths, all_labels, test_size=0.20, random_state=42, stratify=all_labels
)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_val_paths, train_val_labels, test_size=0.20, random_state=42, stratify=train_val_labels
)

print(f"Conjunto de imágenes:\n  Entrenamiento: {len(train_paths)}\n  Validación: {len(val_paths)}\n  Prueba: {len(test_paths)}")

# Pipeline de datos

def load_image(filename):
    # Carga la imagen sin redimensionarla para preservar las sutilezas de la esteganografía.
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image, tf.float32)
    # Normalización: escalar a [0, 1] preservando las relaciones entre valores.
    image /= 255.0
    return image

def create_dataset(paths, labels, batch_size=8, shuffle=False, shuffle_buffer=1000):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    # Cargamos la imagen y conservamos la ruta (si la necesitas para otras verificaciones)
    ds = ds.map(lambda x, y: (load_image(x), y, x), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

batch_size = 44

train_ds = create_dataset(train_paths, train_labels, batch_size, shuffle=True, shuffle_buffer=1000)
val_ds = create_dataset(val_paths, val_labels, batch_size, shuffle=False)
test_ds = create_dataset(test_paths, test_labels, batch_size, shuffle=False)

# Ejemplo: mostrar la forma de un lote del conjunto de entrenamiento
for images, labels, paths in train_ds.take(1):
    print("Forma del batch de imágenes:", images.shape)
    print("Forma del batch de etiquetas:", labels.shape)
    print("Batch de rutas:", paths.shape)

# --- Prueba con una imagen aleatoria del entrenamiento ---

for images, labels, paths in train_ds.take(1):
    idx = random.randint(0, images.shape[0] - 1)
    chosen_image = images[idx]
    chosen_label = labels[idx].numpy()  # etiqueta: 0 o 1
    chosen_path = paths[idx].numpy().decode('utf-8')  

    print("Índice en el batch:", idx)
    print("Etiqueta asignada:", chosen_label)
    print("Ruta del archivo:", chosen_path)

    # Aquí solo mostramos el label sin comparar con la carpeta de origen.
    plt.imshow((chosen_image.numpy() * 255).astype("uint8"))
    plt.title(f"Etiqueta: {chosen_label} | Archivo: {os.path.basename(chosen_path)}")
    plt.axis("off")
    plt.show()


# --- PRUEBA CON IMAGEN ESPECÍFICA ---

# # Especifica la ruta de la imagen que deseas probar
# test_image_path = "1.jpg"  

# # Cargar y preparar la imagen
# test_image = load_and_prepare_image(test_image_path)

# # Realizar la predicción con el modelo
# # Se asume que 'model' está definido y entrenado previamente.
# prediction = model.predict(test_image)

# # Interpretar la predicción (por ejemplo, umbral de 0.5)
# predicted_class = 1 if prediction[0] >= 0.5 else 0

# print("Predicción del modelo:", prediction)
# print("El modelo indica que la imagen es:", "JMiPOD (modificada)" if predicted_class == 1 else "Cover (original)")

# plt.imshow(tf.squeeze(test_image).numpy().astype("uint8"))
# plt.title(f"Predicción: {predicted_class} - {'JMiPOD' if predicted_class == 1 else 'Cover'}")
# plt.axis("off")
# plt.show()
