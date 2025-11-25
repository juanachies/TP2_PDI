import cv2
import numpy as np
import matplotlib.pyplot as plt

def detectar_patentes_final(ruta_imagen):
    img = cv2.imread(ruta_imagen)
    if img is None: return
    
    # Estandarizar
    factor = 600 / img.shape[1]
    img_resized = cv2.resize(img, None, fx=factor, fy=factor)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # --- PROCESAMIENTO ---
    # TopHat para resaltar placa blanca
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

    # Sobel X para bordes verticales
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

    # Limpieza y fusión
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)) 
    thresh = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, close_kernel)
    _, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Contornos
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidatos = []
    
    for c in contours:
        rect = cv2.minAreaRect(c) 
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        (x, y), (w, h), angle = rect
        if w < h: w, h = h, w # Corregir rotación
            
        aspect_ratio = w / float(h)
        area = w * h
        
        # --- FILTRO 1: GEOMETRÍA ESTRICTA ---
        # Subimos el mínimo de 2.0 a 2.2 para descartar faros cuadrados
        if 2.2 <= aspect_ratio <= 6.0 and 1000 < area < 40000:
            
            # --- NUEVO: CÁLCULO DE SCORE ---
            
            # A. Factor de Centralidad Horizontal (0 a 1)
            # (1 = en el centro exacto, 0 = en el borde)
            center_x = rect[0][0]
            dist_center_x = abs(center_x - (width / 2))
            factor_centralidad = 1 - (dist_center_x / (width / 2))
            
            # B. Factor de Aspecto Ideal (Patente Arg es ~3.1)
            # Penalizamos si se aleja de 3.0
            diff_aspect = abs(aspect_ratio - 3.1)
            factor_forma = 1 / (1 + diff_aspect)
            
            # C. Posición Vertical (Preferimos mitad inferior)
            center_y = rect[0][1]
            factor_altura = 1.0
            if center_y < (height * 0.3): # Si está muy arriba (cielo/parabrisas)
                factor_altura = 0.2
            
            # SCORE FINAL: Combinamos todo
            # Damos MUCHO peso a que esté en el centro (factor_centralidad)
            score = area * (factor_centralidad ** 2) * factor_forma * factor_altura
            
            candidatos.append({
                'box': box,
                'score': score,
                'debug': f"C:{factor_centralidad:.2f} F:{factor_forma:.2f}"
            })

    # Visualización
    output = img_resized.copy()
    
    # --- CAMBIO CLAVE: ORDENAR POR SCORE (NO POR ÁREA) ---
    candidatos.sort(key=lambda x: x['score'], reverse=True)

    titulo = "NO DETECTADA"
    if len(candidatos) > 0:
        # El ganador es el Score más alto
        ganador = candidatos[0]
        cv2.drawContours(output, [ganador['box']], 0, (0, 255, 0), 3)
        
        # Escribimos el score para entender por qué ganó
        pos_txt = (ganador['box'][1][0], ganador['box'][1][1]) # Una esquina
        cv2.putText(output, f"{ganador['score']:.0f}", pos_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        titulo = "PATENTE DETECTADA"
        
        # Dibujar perdedores en rojo
        for cand in candidatos[1:]:
            cv2.drawContours(output, [cand['box']], 0, (0, 0, 255), 1)

    # Mostrar
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(thresh, cmap='gray')
    ax1.set_title("Morfología")
    ax1.axis('off')
    
    ax2.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    ax2.set_title(titulo)
    ax2.axis('off')
    plt.show()

# Ejecutar
for i in range(1, 13, 1):
    if i < 10:
        detectar_patentes_final(f'img0{i}.png')
    else:
        detectar_patentes_final(f'img{i}.png')