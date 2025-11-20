import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure(figsize=(8,5))
    if color_img:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title if title else '')
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)

def preprocess(img):
    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # denoise while keeping edges
    den = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    # CLAHE equalization (better than global hist eq for uneven lighting)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    norm = clahe.apply(den)
    return norm

def edges_for_text(gray):
    # Use Sobel-X to emphasize vertical strokes (good for characters)
    sobelx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    abs_sobelx = cv2.convertScaleAbs(sobelx)

    # Normalize and threshold
    _, th = cv2.threshold(abs_sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Small morphology to clear noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened

def candidate_plate_mask(edge_img):
    # Close horizontally to connect characters into a single rectangular region (plate)
    # Use a long horizontal kernel because plates are wide
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,5))
    closed = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, hor_kernel, iterations=2)

    # Optionally dilate a bit to merge fractured pieces
    dil = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_RECT, (5,3)), iterations=1)
    return dil

def find_plate_candidates(mask, min_area=2000):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        ar = w / float(h) if h>0 else 0
        # Typical license plate is wide: adjust range if your plates are different
        if 2.5 < ar < 6.5:
            candidates.append((x,y,w,h,area,ar))
    # sort by area desc
    candidates = sorted(candidates, key=lambda x: x[4], reverse=True)
    return candidates

def extract_and_verify_chars(plate_img, debug=False):
    # Convert to gray+adaptive threshold (characters darker/lighter depending)
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    # increase contrast/denoise a bit
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # Adaptive thresh with inverted to get characters as white on black
    bin_plate = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 19, 8)

    # Morph to clean small holes in characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    clean = cv2.morphologyEx(bin_plate, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours (potential characters)
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_boxes = []
    H, W = clean.shape
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        # filter by size: characters are neither too small nor too large relative to plate
        if h < 0.3*H or h > 0.95*H:
            continue
        if w < 0.03*W or w > 0.3*W:
            continue
        # relation Alto/Ancho ≈ 1.5 - 3.0  -> h/w in that range
        ratio = h / float(w) if w>0 else 0
        if 1.3 <= ratio <= 4.0:   # relax a bit
            area = w*h
            char_boxes.append((x,y,w,h,ratio,area))
    if len(char_boxes) == 0:
        return [], bin_plate, clean

    # sort by x (left to right)
    char_boxes = sorted(char_boxes, key=lambda b: b[0])

    # Try to detect grouping: two groups of 3 characters (6 chars) or at least 5-7 chars contiguous
    # We'll attempt a simple heuristic: cluster x centers into two groups by large gap
    centers = [b[0] + b[2]/2.0 for b in char_boxes]
    # compute gaps between consecutive centers
    gaps = [centers[i+1] - centers[i] for i in range(len(centers)-1)]
    # find largest gap and split there
    if len(gaps) >= 1:
        max_gap_idx = int(np.argmax(gaps))
        left_group = char_boxes[:max_gap_idx+1]
        right_group = char_boxes[max_gap_idx+1:]
    else:
        left_group = char_boxes
        right_group = []

    good_groups = []
    for g in (left_group, right_group):
        if 2 <= len(g) <= 4:  # one group of ~3 chars
            good_groups.append(g)
    # Accept if we found two groups with 2-4 chars each (preferably 3 and 3)
    accepted = (len(good_groups) == 2)
    return char_boxes, bin_plate, clean, accepted, good_groups

# Main loop: read images and try to detect plates
for i in range(1, 3):   # ajusta rango a la cantidad de imágenes que tengas
    img = cv2.imread(f'img{i:02d}.png')
    if img is None:
        print(f"img{i:02d}.png not found")
        continue

    vis = img.copy()
    norm = preprocess(img)
    edges = edges_for_text(norm)
    mask = candidate_plate_mask(edges)
    imshow(img, title=f'Original img{i:02d}', color_img=True)
    imshow(norm, title='Preprocessed gray')
    imshow(edges, title='Sobel-X edges (text candidates)')
    imshow(mask, title='Closed mask for plate candidates')

    candidates = find_plate_candidates(mask, min_area=1500)
    print(f"Found {len(candidates)} plate candidates in img{i:02d}")

    found_any = False
    for (x,y,w,h,area,ar) in candidates[:6]:  # test top candidates
        plate_roi = img[y:y+h, x:x+w]
        char_boxes, bin_plate, clean, accepted, good_groups = extract_and_verify_chars(plate_roi)
        # Draw candidate rectangle and char boxes
        cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)
        # draw char boxes if any
        if len(char_boxes) > 0:
            for (cx,cy,cw,ch,ratio,area) in char_boxes:
                cv2.rectangle(vis, (x+cx, y+cy), (x+cx+cw, y+cy+ch), (0,0,255), 1)

        title = f'cand ar={ar:.2f} area={area}'
        imshow(plate_roi, title='Plate ROI (candidate)')
        imshow(bin_plate, title='Binarized plate (for chars)')
        imshow(clean, title='Cleaned bin plate')

        # check acceptance criterion: two groups of ~3 chars OR at least 5-7 plausible chars
        if accepted or (len(char_boxes) >= 5 and len(char_boxes) <= 8):
            cv2.putText(vis, 'PLATE?', (x, max(0,y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            found_any = True
            # Optionally crop and save or pass to OCR
            plate_crop = img[y:y+h, x:x+w].copy()
            cv2.imwrite(f'plate_candidate_img{i:02d}.png', plate_crop)
            print(f' -> Accepted candidate saved as plate_candidate_img{i:02d}.png')
            break  # stop after first accepted

    if not found_any:
        print("No accepted plates in top candidates, consider relaxing thresholds or reviewing mask.")
    imshow(vis, title=f'Detections img{i:02d}', color_img=True)
