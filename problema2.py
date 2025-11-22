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
            
            if 1.8 <= aspect_ratio <= 3 and area > 15:
                caracteres_validos.append((x, y, w, h))
    
    # Ordenar por posición X
    caracteres_validos.sort(key=lambda c: c[0])
    
    return caracteres_validos


def detectar_patentes(imagen):
    # Cargar imagen
    img = cv2.imread(imagen)   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Umbrales automáticos (mediana)
    v = np.median(gray)
    low = int(0.001 * v)
    high = int(2.5 * v)

    # Canny
    edges = cv2.Canny(gray, low, high, apertureSize=3, L2gradient=True)

    #En algunos funciona bien
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    #edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    #edges = cv2.dilate(edges, kernel, iterations=1)

    # Binarización con Otsu (para análisis de caracteres)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Contornos desde EDGES (funciona mejor)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidatos = []
    placa_contorno = None
    mejor_score = 0
    
    for contorno in contours:
        perimetro = cv2.arcLength(contorno, True)
        aproximacion = cv2.approxPolyDP(contorno, 0.05 * perimetro, True)
            
        if 2 <= len(aproximacion) <= 6:
            x, y, w, h = cv2.boundingRect(aproximacion)
                
            # Validar límites
            if x < 0 or y < 0 or x+w > binary.shape[1] or y+h > binary.shape[0]:
                continue
                    
            relacion_aspecto = float(w) / h
            area = w * h
                

            if 1.8 <= relacion_aspecto <= 4.5 and 1000 < area < 30000:
                roi_binary = binary[y:y+h, x:x+w]
                caracteres = detectar_caracteres_en_roi(roi_binary)
                num_caracteres = len(caracteres)       
                
                candidatos.append({
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'area': area,
                        'aspecto': relacion_aspecto,
                        'caracteres': num_caracteres,
                        'contorno': aproximacion
                    })     
                    
                if 2 <= num_caracteres <= 8:
                        diferencia_6 = abs(num_caracteres - 6)
                        score = (10 - diferencia_6) * 10000 + num_caracteres * 1000 + area
                        
                        if score > mejor_score:
                            placa_contorno = aproximacion
                            mejor_score = score
                
    
    # Visualización
    output_contornos = img.copy()
    cv2.drawContours(output_contornos, contours, -1, (0,255,0), 1)
    
    output_candidatos = img.copy()
    for idx, cand in enumerate(candidatos):
        x, y, w, h = cand['x'], cand['y'], cand['w'], cand['h']
        color = (0, 255, 0) if cand['caracteres'] >= 4 else (0, 0, 255)
        cv2.rectangle(output_candidatos, (x, y), (x+w, y+h), color, 2)
        texto = f"#{idx+1}:{cand['caracteres']}ch"
        cv2.putText(output_candidatos, texto, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Resultado final
    if placa_contorno is not None:
        x, y, w, h = cv2.boundingRect(placa_contorno)
        output_final = img.copy()
        cv2.rectangle(output_final, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        patente_roi = img[y:y+h, x:x+w]
        
        print(f"\n✓ PATENTE DETECTADA en {imagen}")
        print(f"  Coordenadas: x={x}, y={y}, w={w}, h={h}")
        print(f"  Total candidatos evaluados: {len(candidatos)}")
        
        # Mostrar resultados
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].imshow(cv2.cvtColor(output_contornos, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Todos los contornos")
        axes[0, 0].axis("off")
        
        axes[0, 1].imshow(cv2.cvtColor(output_candidatos, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(f"Candidatos ({len(candidatos)})")
        axes[0, 1].axis("off")
        
        axes[1, 0].imshow(cv2.cvtColor(output_final, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("Patente Detectada")
        axes[1, 0].axis("off")
        
        axes[1, 1].imshow(cv2.cvtColor(patente_roi, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("ROI Patente")
        axes[1, 1].axis("off")
        
        plt.tight_layout()
        plt.show()
        
        return (x, y, w, h)
    else:
        print(f"\n✗ No se detectó patente en {imagen}")
        print(f"  Total candidatos evaluados: {len(candidatos)}")
        
        # Mostrar solo contornos y candidatos
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].imshow(cv2.cvtColor(output_contornos, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Todos los contornos")
        axes[0].axis("off")
        
        axes[1].imshow(cv2.cvtColor(output_candidatos, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Candidatos ({len(candidatos)})")
        axes[1].axis("off")
        
        plt.tight_layout()
        plt.show()
        
        return None

for i in range(1, 13, 1):
    if i < 10:
        detectar_patentes(f'img0{i}.png')
    else:
        detectar_patentes(f'img{i}.png')

