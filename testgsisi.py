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


def detectar_patentes_final(ruta_imagen):
    img = cv2.imread(ruta_imagen)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # MORFOLOGÍA
    # TopHat 
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

    # Sobel 
    sobelX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    sobelX = np.absolute(sobelX)
    (minVal, maxVal) = (np.min(sobelX), np.max(sobelX))
    sobelX = (255 * ((sobelX - minVal) / (maxVal - minVal))).astype("uint8")

    # Limpieza y fusión
    sobelX = cv2.morphologyEx(sobelX, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)) 
    thresh = cv2.morphologyEx(sobelX, cv2.MORPH_CLOSE, close_kernel)
    _, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidatos = []
    
    for c in contours:
        rect = cv2.minAreaRect(c) 
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        (x, y), (h, w), angle = rect
            
        aspect_ratio = w / float(h)
        area = w * h
        
        if (2.2 <= aspect_ratio <= 5.0) and (1000 < area < 8000) and (h < w):
            
            # score
            center_x = rect[0][0]
            dist_center_x = abs(center_x - (width / 2))
            factor_centralidad = 1 - (dist_center_x / (width / 2))
            
            score = area * (factor_centralidad ** 3) * (y**2)
            
            candidatos.append({
                'box': box,
                'score': score,
                'x': x,
                'y': y,
                'h': h,
                'w': w
            })

    # Visualización
    output = img.copy()
    
    # ordenamos por score
    candidatos.sort(key=lambda x: x['score'], reverse=True)

    titulo = "NO DETECTADA"
    if len(candidatos) > 0:
        # El ganador es el Score más alto
        patente = candidatos[0]
        print(patente.values())
        #box, score, x, y, h, w = patente.values()
        cv2.drawContours(output, [patente['box']], 0, (0, 255, 0), 3)
        
        # Escribimos el score para entender por qué ganó
        pos_txt = (patente['box'][1][0], patente['box'][1][1]) # Una esquina
        cv2.putText(output, f"{patente['score']:.0f}", pos_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        titulo = "PATENTE DETECTADA"
        
        # Dibujar perdedores en rojo
        for cand in candidatos[1:]:
            cv2.drawContours(output, [cand['box']], 0, (0, 0, 255), 1)
        
        imshow(img[y: y+h, x : x+w])





    # Mostrar
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(thresh, cmap='gray')
    axes[0].set_title("Morfología")
    axes[0].axis('off')
    
    color = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    axes[1].imshow(color)
    axes[1].set_title(titulo)
    axes[1].axis('off')
    plt.show()

# Ejecutar
for i in range(1, 2, 1):
    if i < 10:
        detectar_patentes_final(f'img0{i}.png')
    else:
        detectar_patentes_final(f'img{i}.png')
