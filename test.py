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


# 1. Leer y preprocesar
img = cv2.imread('monedas.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)

# 2. Detectar bordes
edges = cv2.Canny(blur, 50, 150)

# 3. Cerrar huecos (para que los contornos estén completos)
kernel = np.ones((3,3), np.uint8)
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
imshow(closed)


# 4. Buscar contornos
contours, hierarchy = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# 5. Dibujar y analizar
output = img.copy()
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 500:  # descartar ruido pequeño
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
    cv2.drawContours(output, [cnt], -1, color, 2)
    # cv2.putText(output, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Mostrar
plt.figure(figsize=(12,5))
plt.subplot(1,2,1), plt.imshow(edges, cmap='gray'), plt.title('Bordes (Canny)'), plt.axis('off')
plt.subplot(1,2,2), plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)), plt.title('Objetos detectados'), plt.axis('off')
plt.show()