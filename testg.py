import cv2
import numpy as np
import matplotlib.pyplot as plt

def detectar_patentes_final(ruta_imagen):
    img = cv2.imread(ruta_imagen)
    if img is None: return
    
    # 1. Estandarizar tamaño (fundamental para que los Kernels funcionen igual en todas)
    factor = 600 / img.shape[1]
    img_resized = cv2.resize(img, None, fx=factor, fy=factor)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # --- PROCESAMIENTO ---
    
    # CAMBIO 1: Usamos Sobel directo sobre grises.
    # Esto permite detectar tanto patentes blancas (nuevas) como negras (viejas).
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

    # Difuminar para quitar ruido de adoquines/pasto
    gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
    
    # Morfología: "CLOSE" para fusionar las letras en un bloque rectangular
    # Usamos un kernel rectangular ancho (23x3)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 3)) 
    thresh = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, close_kernel)
    
    # Binarización (Otsu calcula el umbral solo)
    _, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Dilatar un poco para rellenar huecos negros dentro de la patente
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Buscar contornos externos
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidatos = []
    
    for c in contours:
        # minAreaRect crea una caja rotada que se ajusta mejor que boundingRect
        rect = cv2.minAreaRect(c) 
        box = cv2.boxPoints(rect)
        box = np.int32(box) 
        
        (x, y), (w, h), angle = rect
        if w < h: w, h = h, w # Aseguramos que w sea el lado largo
            
        aspect_ratio = w / float(h)
        area = w * h
        
        # FILTRO 1: Forma y Tamaño
        # Aspect ratio entre 2.0 y 6.0 cubre patentes normales y Mercosur
        if 2.0 <= aspect_ratio <= 6.0 and 1000 < area < 40000:
            
            # FILTRO 2: Contenido
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [box], 0, 255, -1)
            # Calculamos el brillo promedio de la zona candidata
            mean_val = cv2.mean(img_resized, mask=mask)[0]
            
            # CAMBIO 2: Bajamos exigencia de brillo (>20) para aceptar patentes negras/sucias
            if mean_val > 20: 
                
                # --- SISTEMA DE PUNTUACIÓN (SCORE) ---
                
                # A. Centralidad (Premio si está en el medio horizontal)
                center_x = rect[0][0]
                dist_from_center = abs(center_x - (width / 2))
                factor_centralidad = 1 - (dist_from_center / (width / 2))
                
                # B. Forma (Premio si se parece a 3.1 que es el estándar)
                factor_forma = 1 / (1 + abs(aspect_ratio - 3.1))
                
                # C. Altura (Castigo si está muy arriba, ej: parabrisas)
                center_y = rect[0][1]
                factor_altura = 1.0
                if center_y < (height * 0.35): 
                    factor_altura = 0.2
                
                # Score Final: Tu fórmula exitosa
                score = area * (factor_centralidad ** 2) * factor_forma * factor_altura
                
                candidatos.append({'box': box, 'score': score})

    # Visualización
    output = img_resized.copy()
    
    # Ordenamos por Score descendente
    candidatos.sort(key=lambda x: x['score'], reverse=True)

    titulo = "NO DETECTADA"
    
    if len(candidatos) > 0:
        # El mejor candidato es el ganador
        ganador = candidatos[0]
        cv2.drawContours(output, [ganador['box']], 0, (0, 255, 0), 3)
        titulo = "PATENTE DETECTADA"
        
        # Opcional: Mostrar los descartados en rojo suave
        for cand in candidatos[1:]:
            cv2.drawContours(output, [cand['box']], 0, (0, 0, 255), 1)

    # Mostrar resultado
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title(titulo)
    plt.axis('off')
    plt.show()

# Loop de ejecución
for i in range(1, 13): # Corrección rango para incluir la 12
    nombre = f"img{'0' if i < 10 else ''}{i}.png" # Ajuste para nombres img01...img12
    print(f"Procesando {nombre}...")
    detectar_patentes_final(nombre)