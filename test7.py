import cv2
import numpy as np
import matplotlib.pyplot as plt

def detectar_placa(imagen):
    """Detecta la región de la placa patente"""
    
    # Detección de bordes
    bordes = cv2.Canny(imagen, 30, 200)
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(bordes.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Ordenar contornos por área (los más grandes primero)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:10]
    
    placa_contorno = None
    
    for contorno in contornos:
        # Aproximar el contorno
        perimetro = cv2.arcLength(contorno, True)
        aproximacion = cv2.approxPolyDP(contorno, 0.018 * perimetro, True)
        
        # La placa suele ser un rectángulo (4 vértices)
        if len(aproximacion) == 4:
            x, y, w, h = cv2.boundingRect(aproximacion)
            relacion_aspecto = float(w) / h
            
            # Filtrar por relación de aspecto típica de placas (entre 2:1 y 4:1)
            if 1 <= relacion_aspecto <= 5:
                placa_contorno = aproximacion
                break
    
    return placa_contorno

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

img = cv2.imread('img01.png')

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
placa = detectar_placa(img)
print(placa)