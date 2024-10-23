import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directorio de las imágenes de entrenamiento
train_data_directory = 'C:/DB_Covid19Arg/csv_archivos_limpios/Amazon_test/GTSRB/Final_Test/GTSRB/Final_Test/Images/'

# Crear un generador de imágenes
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # 20% para validación

train_generator = datagen.flow_from_directory(
    train_data_directory,
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_data_directory,
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Cargar el modelo existente
model = load_model('C:/DB_Covid19Arg/csv_archivos_limpios/Amazon_test/traffic_sign_classifier.h5')

# Reajustar el modelo
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    epochs=10  # Ajusta según sea necesario
)

# Guardar el modelo reajustado
model.save('C:/DB_Covid19Arg/csv_archivos_limpios/Amazon_test/traffic_sign_classifier_retrained.h5')
