import cv2
import numpy as np
import matplotlib.pyplot as plt

# Defininimos función para mostrar imágenes
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

# DESARROLLAR UN ALGORITMO QUE DETECTE AUTOMÁTICAMENTE CUANDO SE DETIENEN LOS DADOS


# Leemos el video
video_person = {}
video_person["path"] = "tirada_1.mp4"
cap = cv2.VideoCapture(video_person["path"])                 

# Obtengo Meta-Información 
video_person["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))      
video_person["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    
video_person["fps"] = int(cap.get(cv2.CAP_PROP_FPS))                
video_person["n_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   
print(f'{video_person["path"]} --> Ancho={video_person["width"]} | Alto={video_person["height"]} | fps={video_person["fps"]} | Cant. frames = {video_person["n_frames"]}')

# --- Visualización del video ---------------------------------------------
while (cap.isOpened()):                                                 # Itero, siempre y cuando el video esté abierto
    ret, frame = cap.read()                                             # Obtengo el frame
    if ret==True:                                                       # ret indica si la lectura fue exitosa (True) o no (False)
        cv2.imshow('Frame',frame)                                       # Muestro el frame
        if cv2.waitKey(25) & 0xFF == ord('q'):                          # Corto la repoducción si se presiona la tecla "q"
            break
    else:
        break                                       # Corto la reproducción si ret=False, es decir, si hubo un error o no quedán mas frames.


cap.release()                   # Cierro el video
cv2.destroyAllWindows()         # Destruyo todas las ventanas abiertas


# --- Analizo un frame -----------------------------------------------------
cap = cv2.VideoCapture(video_person["path"])        # Abro el video
frame_index = 80                                                           # en 80 estan los dados quietos
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)                              # cambia la escala
ret, frame = cap.read()                             # Obtengo 1 frame
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
imshow(frame)

if ret:
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imshow(frame, title=f"Video - Frame {frame_index}")
    # Analizo un recorte del frame
    bg_crop_RGB = frame[50:1600, 95:1074, :]
    imshow(bg_crop_RGB, title=f"Frame {frame_index} - Crop fondo verde")
else:
    print(f"No se pudo leer el frame {frame_index}.")

cap.release()                                       # Cierro el video

# --- Aplicamos máscara para fitrar el área de juego ------------------------------------
hsv_lower=(30, 50, 50)
hsv_upper=(50, 255, 255)
frame_HSV = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
imshow(frame_HSV, title="Video - Frame 1")
mask = cv2.inRange(frame_HSV, hsv_lower, hsv_upper)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imshow(mask, title="Video - mascara 1")
plt.imshow(frame)
# recorte manual
"""# --- Analizo un recorte del frame ---------------------------------------- (iterar en este recuadro)
bg_crop_RGB = frame[50:1600, 95:1074, :]                     # Cortar una porcion correspondiente al fondo verde con cielab
imshow(bg_crop_RGB, title="Frame 1 - Crop fondo verde")"""

# utilizo cielab para que el fondo sea negro, umbralizo------------------------------ (volver a definir el frame de los dados)
plt.figure()
ax1 = plt.subplot(221); plt.xticks([]), plt.yticks([]), plt.imshow(frame), plt.title('---')
brightLAB = cv2.cvtColor(bg_crop_RGB, cv2.COLOR_RGB2LAB)
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(brightLAB[:,:,1], cmap="gray"), plt.title('cielab-A')
plt.show(block=False)

gray = cv2.cvtColor(brightLAB, cv2.COLOR_LAB2LRGB)
gray = cv2.cvtColor(brightLAB, cv2.COLOR_RGB2GRAY)
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(gray, cmap="gray"), plt.title('gray')
umbral, thresh_img = cv2.threshold(gray, thresh=120, maxval=255, type=cv2.THRESH_BINARY)  # Umbralamos
plt.subplot(224,sharex=ax1,sharey=ax1), plt.imshow(thresh_img, cmap="gray"), plt.title('umbral')
plt.show(block=False)


# Contornos por jerarquía ----------------------------------
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# -- Aproximación de contornos con polinomios ---------------------------------- (encuentra los 5 dados)
contours_area = sorted(contours, key=cv2.contourArea, reverse=True)

# Itero sobre los índices de los contornos  ------------------------------------------ extraemos info de las métricas de los dados
for i in range(5):  # Índices 0, 1, 2, 3, 4
    cnt = contours_area[i]  # Obtener el contorno correspondiente
    area = cv2.contourArea(cnt)  # Calcular el área
    perimeter = cv2.arcLength(cnt, True)  # Calcular el perímetro (True para contornos cerrados)
    
    # Factor de Forma
    Fp = area / (perimeter ** 2) 
    print(f"Contorno {i}: Área = {area:.2f}, Perímetro = {perimeter:.2f}, Factor de Forma = {Fp:.4f}")

output_image = bg_crop_RGB.copy()
for contour in contours_area:
    # Fp: Facto de Forma --------------------------------
    area = cv2.contourArea(contour) 
    perimetro = cv2.arcLength(contour, True)
    # cuadrado --------------------------------
    if perimetro > 0:
        Fp = area / perimetro**2
        if 0.04 < Fp < 0.06:
            cv2.drawContours(output_image, [contour], -1, (0, 255, 0), thickness=2)


plt.imshow(output_image)
plt.show()


		
def detect_dice_stop(video_path, min_contours=5, stability_frames=9):
    """
    Detecta automáticamente el frame en el que los dados se detienen basándose en el factor de forma.
    Args:
        video_path (str): Ruta del video.
        factor_range (tuple): Rango del factor de forma (mínimo, máximo).
        min_contours (int): Número mínimo de contornos que deben cumplir la condición.
    Returns:
        int: Índice del frame donde los dados se detienen, o -1 si no se encuentra.
    """
    # Abrir el video
    cap = cv2.VideoCapture(video_path)        # Abro el video
    if not cap.isOpened():
        print(f"Error al abrir el video: {video_path}")
        return -1
    frame_count = 0  # Contador de frames
    stable_frame_count = 0 
    while cap.isOpened():
        ret, frame = cap.read()                 # Obtengo 1 frame
        if not ret:
            break
        # Convertir a espacio de color LAB y tomar la componente L
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bg_crop_RGB = frame[50:1600, 95:1074, :]                             # delimitar la zona de juego con cielab
        brightLAB = cv2.cvtColor(bg_crop_RGB, cv2.COLOR_RGB2LAB)
        gray = cv2.cvtColor(brightLAB, cv2.COLOR_BGR2GRAY)
        # Aplicar umbral
        umbral, thresh_img = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        # Encontrar contornos
        contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # Filtrar contornos según el factor de forma
        valid_contours = []
        output_image = bg_crop_RGB.copy()
        for contour in contours:
            area = cv2.contourArea(contour)
            perimetro = cv2.arcLength(contour, True)
            if perimetro > 100:
                Fp = area / (perimetro ** 2)
                if 0.04 < Fp < 0.06:
                    cv2.drawContours(output_image, [contour], -1, (0, 255, 0), thickness=2)
                    valid_contours.append(contour)
        # Verificar si se encuentran los contornos deseados
        cv2.imshow('Contornos que cumplen la condición', output_image)
        if len(valid_contours) == min_contours:
            stable_frame_count += 1
            if stable_frame_count >= stability_frames:
                print(f"Dados detectados en el frame {frame_count}.")
                cap.release()
                return frame_count - stability_frames + 1  # Retorna el primer frame del período estable
        else:
            stable_frame_count = 0
        frame_count += 1
    cap.release()
    print("No se encontraron suficientes contornos que cumplan la condición.")
    return -1

video_path = "tirada_1.mp4"
frame_stop = detect_dice_stop(video_path, stability_frames=9)

if frame_stop != -1:
    print(f"Los dados se detuvieron en el frame {frame_stop}.")
else:
    print("No se pudo determinar el momento en que los dados se detienen.")


















img_pixels = bg_crop_RGB.reshape(-1,3)                                 # Re-ordeno los pixels, de manera que cada pixel ocupe una fila.
colours, counts = np.unique(img_pixels, axis=0, return_counts=True) # Obtengo los valores únicos de todos los pixels y su frecuencia de aparición.
N_colours = colours.shape[0]                                        # Cantidad de colores.
N_colours
colours
counts

bg_color_RGB = colours[0]
bg_color_RGB = np.reshape(bg_color_RGB,(1,1,-1))
bg_color_HSV = cv2.cvtColor(bg_color_RGB, cv2.COLOR_RGB2HSV)    # Conversión a HSV.
bg_color_HSV

frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)    

mask_bg = cv2.inRange(frame_hsv, bg_color_HSV[0,0,:], bg_color_HSV[0,0,:])
mask_person = cv2.bitwise_not(mask_bg)
bg = cv2.bitwise_and(frame, frame, mask=mask_bg)
person = cv2.bitwise_and(frame, frame, mask=mask_person)

plt.figure()
ax = plt.subplot(231); imshow(frame, title="Frame", new_fig=False)
plt.subplot(232, sharex=ax, sharey=ax), imshow(mask_bg, title="Máscara Background", new_fig=False)
plt.subplot(233, sharex=ax, sharey=ax), imshow(mask_person, title="Máscara persona", new_fig=False)
plt.subplot(235, sharex=ax, sharey=ax), imshow(bg, title="Background", new_fig=False)
plt.subplot(236, sharex=ax, sharey=ax), imshow(person, title="Persona", new_fig=False)
plt.suptitle(f'bg_RGB = {bg_color_RGB} | bg_HSV = {bg_color_HSV}')
plt.show(block=False)






