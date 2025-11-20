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

# Cargar imagen 
img = cv2.imread("monedas.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Filtrar ruido
blur = cv2.GaussianBlur(gray, (9, 9), 0)

#Detección de bordes con Canny
edges = cv2.Canny(blur, 35, 110)

# Cierre morfológico 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=4)

# Buscar contornos
contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

detecciones = {}
conteo_dado = []

# Dibujar y analizar 
output = img.copy()
for cnt in contours:
    area = cv2.contourArea(cnt)
    
    if area < 500:
        continue

    perimeter = cv2.arcLength(cnt, True)
    
    if perimeter < 1e-6:
        continue
        
    circularity = 4 * np.pi * area / (perimeter ** 2)
    
    # Calcular bounding box y aspect ratio
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h > 0 else 0
    # Clasificación refinada basada en datos reales:
    # Monedas: C >= 0.85 (mayoría) o C=0.77 con AR muy cercano a 1.0
    # Dados: C < 0.77 o C=0.77 con AR >= 1.01
    
    if circularity >= 0.85:
        # Alta circularidad -> definitivamente moneda
        color, label = (0,255,0), 'Moneda'
    elif 0.73 <= circularity < 0.85 and aspect_ratio < 1.0:
        # Circularidad media-alta + AR < 1.0 -> moneda problemática
        color, label = (0,255,0), 'Moneda'
    elif circularity > 0.60:
        # Resto con circularidad razonable -> dado
        color, label = (255,0,0), 'Dado'
    else:
        continue
       
    # Clasificación de monedas
    if label == 'Moneda':
        if area < 60000:
            color, label = (0,255,0), 'Moneda 10 centavos'

        elif area < 90000:
             color, label = (0,255,255), 'Moneda 1 peso'
        
        else:
            color, label = (0,0,255), 'Moneda 50 centavos'
    
    if label == 'Dado':
        roi = edges[y:y+h, x:x+w]

        circles = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=9,
            minRadius=15,
            maxRadius=20
        )

        num_puntos = 0
        if circles is not None:
            altura_roi = roi.shape[0]
            for circle in circles[0]:
                if circle[1] < altura_roi * 0.90:  
                    num_puntos += 1
            conteo_dado.append(num_puntos)
        label = f'Dado de {num_puntos} caras'

    if label not in detecciones.keys():
        detecciones[label] = 1
    else:
        detecciones[label] += 1

    cv2.drawContours(output, [cnt], -1, color, 2)
    cv2.putText(output, f'{label}' , (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)


# Mostrar
plt.figure(figsize=(12,5))
plt.subplot(1,3,1), plt.imshow(edges, cmap='gray'), plt.title('Bordes (Canny)'), plt.axis('off')
plt.subplot(1,3,2), plt.imshow(closed, cmap='gray'), plt.title('Morfología'), plt.axis('off')
plt.subplot(1,3,3), plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)), plt.title('Objetos detectados'), plt.axis('off')
plt.show()
