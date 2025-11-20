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


def detectar_caracteres_en_roi(roi_binary):
    """Detecta caracteres en la ROI binarizada"""
    contours_chars, _ = cv2.findContours(roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    caracteres_validos = []
    for cnt in contours_chars:
        x, y, w, h = cv2.boundingRect(cnt)
        
        if h > 0 and w > 0:
            aspect_ratio = h / float(w)
            area = w * h
            
            # Caracteres con relación de aspecto 1.5-3.0 y área razonable
            if 1.5 <= aspect_ratio <= 3.0 and area > 50:
                caracteres_validos.append((x, y, w, h))
    
    # Ordenar por posición X
    caracteres_validos.sort(key=lambda c: c[0])
    
    return caracteres_validos


for i in range(1,2):
    
    # Cargar imagen
    img = cv2.imread(f'img0{i}.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Dimensiones de referencia de la patente real
    patente_ref_y1, patente_ref_y2 = 134, 172
    patente_ref_x1, patente_ref_x2 = 335, 410
    patente_ref_h = patente_ref_y2 - patente_ref_y1  # 38
    patente_ref_w = patente_ref_x2 - patente_ref_x1  # 75
    patente_ref_aspecto = patente_ref_w / float(patente_ref_h)
    patente_ref_area = patente_ref_w * patente_ref_h
    
    print(f"\n=== Dimensiones de referencia de la patente ===")
    print(f"Alto: {patente_ref_h}, Ancho: {patente_ref_w}")
    print(f"Relación de aspecto: {patente_ref_aspecto:.2f}")
    print(f"Área: {patente_ref_area}")
    
    # Tolerancias para filtrado (±30% en dimensiones, ±20% en aspecto)
    aspecto_min = patente_ref_aspecto * 0.8
    aspecto_max = patente_ref_aspecto * 1.2
    area_min = patente_ref_area * 0.5
    area_max = patente_ref_area * 2.0
    
    print(f"Filtros ajustados:")
    print(f"  Relación aspecto: {aspecto_min:.2f} - {aspecto_max:.2f}")
    print(f"  Área: {area_min:.0f} - {area_max:.0f}")

    # Umbrales automáticos (mediana)
    v = np.median(gray)
    low = int(0.66 * v)
    high = int(1.33 * v)

    # Canny
    edges = cv2.Canny(gray, low, high, apertureSize=3, L2gradient=True)
    
    # Binarización con Otsu (esta detecta bien la patente)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Contornos desde la imagen binaria en lugar de Canny
    contours_binary, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # También obtener contornos de Canny (método original)
    output = img.copy()
    contours_canny, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours_canny, -1, (0,255,0), 2)

    # Combinar ambos conjuntos de contornos
    todos_contornos = list(contours_canny) + list(contours_binary)
    
    placa_contorno = None
    mejor_score = 0
    candidatos = []

    for contorno in todos_contornos:
        # Aproximar el contorno
        perimetro = cv2.arcLength(contorno, True)
        aproximacion = cv2.approxPolyDP(contorno, 0.018 * perimetro, True)
        
        # La placa suele ser un rectángulo (4 vértices)
        if len(aproximacion) == 4:
            x, y, w, h = cv2.boundingRect(aproximacion)
            
            # Validar que esté dentro de los límites
            if x < 0 or y < 0 or x+w > binary.shape[1] or y+h > binary.shape[0]:
                continue
                
            relacion_aspecto = float(w) / h
            area = w * h
            
            # Filtrar usando las dimensiones de referencia
            if aspecto_min <= relacion_aspecto <= aspecto_max and area_min < area < area_max:
                # Extraer ROI de la imagen binaria
                roi_binary = binary[y:y+h, x:x+w]
                
                # Detectar caracteres
                caracteres = detectar_caracteres_en_roi(roi_binary)
                num_caracteres = len(caracteres)
                
                candidatos.append({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'area': area,
                    'aspecto': relacion_aspecto,
                    'caracteres': num_caracteres,
                    'contorno': aproximacion
                })
                
                # La patente debe tener entre 4 y 8 caracteres
                if 1 <= num_caracteres <= 8:
                    # Score: priorizar número de caracteres cercano a 6
                    diferencia_6 = abs(num_caracteres - 6)
                    score = (10 - diferencia_6) * 10000 + area
                    
                    if score > mejor_score:
                        placa_contorno = aproximacion
                        mejor_score = score

    # Mostrar candidatos para debug
    print(f"\n=== Candidatos detectados en img0{i} ===")
    
    # Dibujar TODOS los candidatos en la imagen
    img_todos_candidatos = img.copy()
    
    for idx, cand in enumerate(candidatos):
        x, y, w, h = cand['x'], cand['y'], cand['w'], cand['h']
        
        # Dibujar rectángulo de cada candidato con un color diferente
        color = (0, 255, 0) if cand['caracteres'] >= 4 else (0, 0, 255)  # Verde si tiene suficientes chars, rojo si no
        cv2.rectangle(img_todos_candidatos, (x, y), (x+w, y+h), color, 2)
        
        # Agregar número de candidato y cantidad de caracteres
        texto = f"#{idx+1}: {cand['caracteres']}ch"
        cv2.putText(img_todos_candidatos, texto, (x, y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        print(f"Candidato {idx+1}:")
        print(f"  Posición: ({x}, {y}), Tamaño: {w}x{h}")
        print(f"  Área: {cand['area']}, Aspecto: {cand['aspecto']:.2f}")
        print(f"  Caracteres detectados: {cand['caracteres']}")
        print()

    imshow(binary, title=f'Binarización img0{i}')
    imshow(edges, title=f'Canny img0{i}')
    imshow(output, title=f'Contornos img0{i}')
    imshow(img_todos_candidatos, title=f'TODOS los Candidatos img0{i}', color_img=True)
    
    # Si se detectó una patente, mostrar sus coordenadas
    if placa_contorno is not None:
        x, y, w, h = cv2.boundingRect(placa_contorno)
        
        # Indexar la región de la patente
        patente_roi = img[y:y+h, x:x+w]
        patente_gray = gray[y:y+h, x:x+w]
        patente_binary = binary[y:y+h, x:x+w]
        
        # Detectar caracteres para visualización
        caracteres = detectar_caracteres_en_roi(patente_binary)
        
        # Dibujar rectángulo en la imagen
        img_patente = img.copy()
        cv2.rectangle(img_patente, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        # Dibujar caracteres detectados
        patente_con_chars = patente_roi.copy()
        for cx, cy, cw, ch in caracteres:
            cv2.rectangle(patente_con_chars, (cx, cy), (cx+cw, cy+ch), (255, 0, 0), 2)
        
        imshow(img_patente, title=f'Patente Detectada img0{i}', color_img=True)
        imshow(patente_con_chars, title=f'ROI con Caracteres img0{i}', color_img=True)
        imshow(patente_binary, title=f'ROI Binaria img0{i}')
        
        print(f"\n=== PATENTE DETECTADA img0{i} ===")
        print(f"Coordenadas de la patente:")
        print(f"  x: {x}, y: {y}")
        print(f"  Ancho: {w}, Alto: {h}")
        print(f"  Esquina superior izquierda: ({x}, {y})")
        print(f"  Esquina inferior derecha: ({x+w}, {y+h})")
        print(f"  Forma de la ROI: {patente_roi.shape}")
        print(f"  Caracteres detectados: {len(caracteres)}")
    else:
        print(f"\n=== Imagen 0{i} ===")
        print("No se detectó patente")

plt.show()