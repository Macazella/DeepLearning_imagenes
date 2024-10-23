import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score

# Parte 1: Carga de datos y preprocesamiento

# Ruta a las imágenes de entrenamiento
train_images_path = 'C:/DB_Covid19Arg/csv_archivos_limpios/Amazon_test/GTSRB/Final_Training/GTSRB/Final_Training/Images/'

# Función para cargar las imágenes y sus etiquetas
def load_data(images_path):
    images = []
    labels = []
    for root, dirs, files in os.walk(images_path):
        for file in files:
            if file.endswith(".ppm"):  # Formato de imagen .ppm
                image = cv2.imread(os.path.join(root, file))
                image = cv2.resize(image, (32, 32))  # Redimensionar a 32x32 píxeles
                label = int(os.path.basename(root))  # El nombre del directorio es la etiqueta
                images.append(image)
                labels.append(label)
    return np.array(images), np.array(labels)

# Cargar imágenes y etiquetas
X, y = load_data(train_images_path)

# Normalizar los valores de las imágenes
X = X / 255.0

# Dividir los datos en entrenamiento y validación (80% para entrenar y 20% para validar)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Datos de entrenamiento: {X_train.shape}, Datos de validación: {X_val.shape}")

# Parte 2: Aumentación de datos

# Generador de imágenes con aumentación para evitar sobreajuste
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Ajustar el generador a los datos de entrenamiento
train_datagen.fit(X_train)

# Parte 3: Construcción del modelo CNN

# Definir el modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(43, activation='softmax')  # 43 clases
])

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Parte 4: Entrenamiento del modelo

# Early stopping para evitar sobreajuste
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=20,
    callbacks=[early_stopping]
)

# Guardar el modelo entrenado
model.save('C:/DB_Covid19Arg/csv_archivos_limpios/Amazon_test/traffic_sign_classifier.h5')

# Evaluar el modelo en el conjunto de validación
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")
