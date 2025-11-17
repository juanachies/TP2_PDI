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


original = cv2.imread('monedas.jpg', cv2.IMREAD_GRAYSCALE) 
img = cv2.GaussianBlur(original, (5,5), 0)
imshow(img, title='Imagen en escala de grises')

img_canny = cv2.Canny(img, 80, 150)
imshow(img_canny)

L = 3
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (L, L) )
closed = cv2.morphologyEx(img_canny, cv2.MORPH_GRADIENT, kernel)

imshow(closed)

contours, hierarchy = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# 5. Dibujar y analizar
output = img.copy()
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 200:  # descartar ruido pequeño
        continue

    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)

    # Clasificar según circularidad
    if circularity > 0.8:
        color, label = (0,255,0), 'Moneda'
    else:
        color, label = (255,0,0), 'Dado'

    # Dibujar contorno y etiqueta
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.drawContours(output, [cnt], -1, 2)
    cv2.putText(output, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Mostrar
plt.figure(figsize=(12,5))
plt.subplot(1,2,1), plt.imshow(img_canny, cmap='gray'), plt.title('Bordes (Canny)'), plt.axis('off')
plt.subplot(1,2,2), plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)), plt.title('Objetos detectados'), plt.axis('off')
plt.show()