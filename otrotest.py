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


def detectar_patentes(ruta_imagen):

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

    # El ganador es el Score más alto
    patente = candidatos[0]
    box, score, x, y, h, w = patente.values()
    x, y, h, w = int(x), int(y), int(h), int(w)
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    cv2.drawContours(output, [box], 0, (0, 255, 0), 3)
    
    patente = (img[y1:y1+h, x1:x1+w])

    # Mostrar
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # axes[0].imshow(thresh, cmap='gray')
    # axes[0].set_title("Morfología")
    # axes[0].axis('off')
    
    # color = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    # axes[1].imshow(color)
    # axes[1].set_title('Patente detectada')
    # axes[1].axis('off')
    # plt.show()

    return patente

def detectar_letras(img):
    vis = img.copy()
    
    # Escalar imagen para mejor resolución
    h, w = img.shape[:2]
    scale = 3
    img_scaled = cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    vis_scaled = img_scaled.copy()
    
    gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
    
    # Threshold Otsu SIN inversión
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Erosión fuerte para separar caracteres conectados
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    eroded = cv2.erode(thresh, kernel_erode, iterations=3)
    
    # Dilatación para reconstruir los caracteres
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    reconstructed = cv2.dilate(eroded, kernel_dilate, iterations=2)

    # Limpieza de ruido pequeño
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(reconstructed, cv2.MORPH_OPEN, kernel_small, iterations=1)

    imshow(cleaned)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, 8, cv2.CV_32S)

    caracteres = []
    
    # Calcular altura media de componentes para filtros adaptativos
    alturas = []
    anchos = []
    for i in range(1, num_labels):
        h_comp = stats[i, cv2.CC_STAT_HEIGHT]
        w_comp = stats[i, cv2.CC_STAT_WIDTH]
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 50:
            alturas.append(h_comp)
            anchos.append(w_comp)
    
    if len(alturas) > 0:
        altura_media = np.median(alturas)
        ancho_medio = np.median(anchos)
    else:
        altura_media = img_scaled.shape[0] * 0.6
        ancho_medio = img_scaled.shape[1] * 0.1

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Filtros adaptativos
        aspect_ratio = h / float(w) if w > 0 else 0
        
        # Área basada en dimensiones medias
        area_min = altura_media * ancho_medio * 0.15
        area_max = altura_media * ancho_medio * 6
        
        if area < area_min or area > area_max:
            continue
        
        # Caracteres típicamente más altos que anchos
        if aspect_ratio < 0.6 or aspect_ratio > 6.5:
            continue
        
        # Altura y ancho razonables
        if h < altura_media * 0.3 or h > altura_media * 2.8:
            continue
        if w < ancho_medio * 0.2 or w > ancho_medio * 4.5:
            continue

        cv2.rectangle(vis_scaled, (x, y), (x + w, y + h), (255, 0, 0), 2)
        crop = img_scaled[y:y+h, x:x+w]
        caracteres.append({"x": x, "crop": crop})

    # Ordenar left→right
    caracteres.sort(key=lambda c: c["x"])

    # Subplots
    total_plots = 1 + len(caracteres)
    cols = 6
    rows = int(np.ceil(total_plots / cols))

    plt.figure(figsize=(16, 4 * rows))

    plt.subplot(rows, cols, 1)
    plt.imshow(cv2.cvtColor(vis_scaled, cv2.COLOR_BGR2RGB))
    plt.title("Imagen con bounding boxes")
    plt.axis("off")

    for idx, item in enumerate(caracteres):
        plt.subplot(rows, cols, idx + 2)
        plt.imshow(cv2.cvtColor(item["crop"], cv2.COLOR_BGR2RGB))
        plt.title(f"Carácter {idx+1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    return caracteres

# Ejecutar
for i in range(1, 12, 1):
    if i < 10:
        patente = detectar_patentes(f'img0{i}.png')
        detectar_letras(patente)
    else:
        patente = detectar_patentes(f'img{i}.png')
        detectar_letras(patente)



