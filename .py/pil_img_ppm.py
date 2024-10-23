import os
from PIL import Image

# Directorio de las imágenes
image_directory = 'C:/DB_Covid19Arg/csv_archivos_limpios/Amazon_test/GTSRB/Final_test/GTSRB/Final_Test/Images/'

# Nombres base de las imágenes
base_names = [f'12630 ({i})' for i in range(1, 11)]  # del 1 al 10

# Extensiones de formatos que necesitas convertir
formats = ['jpg', 'webp', 'avif']

# Función para convertir imagen a PPM
def convert_to_ppm(image_path, output_path):
    try:
        # Cargar la imagen
        img = Image.open(image_path)
        # Guardar en formato PPM
        img.save(output_path, format='PPM')
        print(f"Convertido: {image_path} a {output_path}")
    except Exception as e:
        print(f"Error al convertir {image_path}: {e}")

# Recorrer cada imagen base
for base_name in base_names:
    for ext in formats:
        image_path = os.path.join(image_directory, f'{base_name}.{ext}')
        output_path = os.path.join(image_directory, f'{base_name}.ppm')
        if os.path.exists(image_path):
            convert_to_ppm(image_path, output_path)
        else:
            print(f"No se encontró la imagen: {image_path}")
