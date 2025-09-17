import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
import matplotlib.pyplot as plt

# Пути к данным
color_dir = '/content/drive/MyDrive/set2.0/color'
grayscale_dir = '/content/drive/MyDrive/set2.0/grayscale'
model_save_path = '/content/drive/MyDrive/set2.0/model.h5'
test_dir = '/content/drive/MyDrive/set2.0/test'
output_dir = '/content/drive/MyDrive/set2.0/colored_images'

# Параметры
img_size = 256
batch_size = 32
epochs = 60


# Функция для загрузки и предобработки изображений
def load_images(color_dir, grayscale_dir):
    color_images = []
    grayscale_images = []

    for i in range(1, 1171):
        color_img_name = f"image_{i}.jpg"
        grayscale_img_name = f"image_{i}.jpg"

        color_img_path = os.path.join(color_dir, color_img_name)
        grayscale_img_path = os.path.join(grayscale_dir, grayscale_img_name)

        color_img = load_img(color_img_path, target_size=(img_size, img_size))
        color_img = img_to_array(color_img)
        color_img = rgb2lab(color_img / 255.0)
        color_images.append(color_img[:, :, 1:])  # Используем только каналы a и b

        grayscale_img = load_img(grayscale_img_path, target_size=(img_size, img_size), color_mode='grayscale')
        grayscale_img = img_to_array(grayscale_img)
        grayscale_images.append(grayscale_img)

    return np.array(color_images), np.array(grayscale_images)


# Загрузка данных
color_images, grayscale_images = load_images(color_dir, grayscale_dir)

# Преобразование данных
X = grayscale_images / 255.0
Y = color_images / 100.0  # Нормализация значений LAB

# Инициализация модели
model = tf.keras.Sequential([

tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(img_size, img_size, 1)),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=2),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', strides=2),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same'),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.UpSampling2D((2, 2)),
tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.UpSampling2D((2, 2)),
tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.UpSampling2D((2, 2)),
tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.UpSampling2D((2, 2)),
tf.keras.layers.Conv2D(2, (3, 3), activation='tanh', padding='same')
])

model.compile(optimizer='adam', loss='mse', metrics = ['accuracy'])

# Обучение модели
history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Сохранение модели
model.save(model_save_path)


# Функция для раскрашивания изображения
def colorize_image(model, image_path):
    img = load_img(image_path, target_size=(img_size, img_size), color_mode='grayscale')
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0

    pred = model.predict(img)
    pred = pred * 100.0

    # Создание пустого изображения в пространстве LAB
    lab_img = np.zeros((img_size, img_size, 3))
    lab_img[:, :, 0] = img[0, :, :, 0] * 100.0
    lab_img[:, :, 1:] = pred[0]

    # Преобразование в RGB
    rgb_img = lab2rgb(lab_img)

    # Преобразование типа данных в uint8
    rgb_img = (rgb_img * 255).astype(np.uint8)

    return rgb_img


# Создание папки для сохранения раскрашенных изображений
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    # Раскрашивание всех изображений в папке test
for i in range(1, 5):
    test_image_path = os.path.join(test_dir, f"{i}.jpg")
    colored_image = colorize_image(model, test_image_path)
    output_image_path = os.path.join(output_dir, f"colored_{i}.jpg")
    imsave(output_image_path, colored_image)
