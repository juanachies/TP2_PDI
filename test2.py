import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)


# Leer imagen
original = cv2.imread('monedas.jpg', cv2.IMREAD_GRAYSCALE) 
imshow(original, title='1. Imagen Original')

# Aplicar blur para reducir ruido
img = cv2.GaussianBlur(original, (5,5), 0)
imshow(img, title='2. Imagen con Blur')

# MÉTODO 1: Umbralización Simple
# El fondo es más claro (gris claro), los objetos son más oscuros
_, thresh_simple = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
imshow(thresh_simple, title='3a. Umbral Simple (INV)')

# MÉTODO 2: Umbralización Adaptativa
# Mejor para iluminación no uniforme
thresh_adaptativa = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_MEAN_C, 
                                           cv2.THRESH_BINARY_INV, 21, 5)
imshow(thresh_adaptativa, title='3b. Umbral Adaptativo')

# MÉTODO 3: Otsu (encuentra el umbral óptimo automáticamente)
_, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
imshow(thresh_otsu, title='3c. Umbral Otsu')

# Limpieza con operaciones morfológicas
# Eliminar ruido pequeño
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
cleaned = cv2.morphologyEx(thresh_otsu, cv2.MORPH_OPEN, kernel_open)
imshow(cleaned, title='4. Limpieza con Opening')

# Rellenar huecos
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
final = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
imshow(final, title='5. Relleno con Closing')

print("Prueba los diferentes métodos de umbralización")
print("El método Otsu (3c) suele dar buenos resultados automáticamente")