import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Función para mostrar las imágenes y las predicciones
def predict_and_display_image(image_path, model):
    # Cargar la imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen {image_path}. Verifique la ruta o la integridad del archivo.")
        return
    
    # Redimensionar la imagen
    image_resized = cv2.resize(image, (32, 32))
    
    # Normalizar la imagen
    image_normalized = image_resized / 255.0
    
    # Expandir las dimensiones para que coincida con la entrada del modelo
    image_array = np.expand_dims(image_normalized, axis=0)
    
    # Realizar la predicción
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    
    # Mostrar la imagen y el resultado de la predicción
    plt.imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction = {'Stop' if predicted_class == 14 else 'Not Stop'}")
    plt.show()

# Cargar el modelo entrenado desde la ruta correcta
model = load_model('C:/DB_Covid19Arg/csv_archivos_limpios/Amazon_test/traffic_sign_classifier.h5')

# Rutas a las imágenes que serán probadas (asegúrate de que las imágenes existen en estas rutas)
test_images_paths = [
    # Imágenes de prueba del 00000.ppm al 12629.ppm
    *[f'C:/DB_Covid19Arg/csv_archivos_limpios/Amazon_test/GTSRB/Final_Test/GTSRB/Final_Test/Images/{str(i).zfill(5)}.ppm' for i in range(0, 12630)],
    
    # Nuevas imágenes añadidas
    'C:/DB_Covid19Arg/csv_archivos_limpios/Amazon_test/GTSRB/Final_Test/GTSRB/Final_Test/Images/12630.ppm',
    'C:/DB_Covid19Arg/csv_archivos_limpios/Amazon_test/GTSRB/Final_Test/GTSRB/Final_Test/Images/12631.ppm',
    'C:/DB_Covid19Arg/csv_archivos_limpios/Amazon_test/GTSRB/Final_Test/GTSRB/Final_Test/Images/12632.ppm',
    'C:/DB_Covid19Arg/csv_archivos_limpios/Amazon_test/GTSRB/Final_Test/GTSRB/Final_Test/Images/12633.ppm',
    'C:/DB_Covid19Arg/csv_archivos_limpios/Amazon_test/GTSRB/Final_Test/GTSRB/Final_Test/Images/12634.ppm',
    'C:/DB_Covid19Arg/csv_archivos_limpios/Amazon_test/GTSRB/Final_Test/GTSRB/Final_Test/Images/12635.ppm',
    'C:/DB_Covid19Arg/csv_archivos_limpios/Amazon_test/GTSRB/Final_Test/GTSRB/Final_Test/Images/12636.ppm',
    'C:/DB_Covid19Arg/csv_archivos_limpios/Amazon_test/GTSRB/Final_Test/GTSRB/Final_Test/Images/12637.ppm',
    'C:/DB_Covid19Arg/csv_archivos_limpios/Amazon_test/GTSRB/Final_Test/GTSRB/Final_Test/Images/12638.ppm',
    'C:/DB_Covid19Arg/csv_archivos_limpios/Amazon_test/GTSRB/Final_Test/GTSRB/Final_Test/Images/12639.ppm',
    'C:/DB_Covid19Arg/csv_archivos_limpios/Amazon_test/GTSRB/Final_Test/GTSRB/Final_Test/Images/12640.ppm',
]

# Hacer predicciones en las imágenes de prueba y mostrar los resultados
for image_path in test_images_paths:
    predict_and_display_image(image_path, model)
