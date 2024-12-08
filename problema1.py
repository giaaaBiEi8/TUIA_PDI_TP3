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
video_person["path"] = "tirada_3.mp4"
cap = cv2.VideoCapture(video_person["path"])                 

"""# Obtengo Meta-Información 
video_person["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))      
video_person["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    
video_person["fps"] = int(cap.get(cv2.CAP_PROP_FPS))                
video_person["n_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   
print(f'{video_person["path"]} --> Ancho={video_person["width"]} | Alto={video_person["height"]} | fps={video_person["fps"]} | Cant. frames = {video_person["n_frames"]}')
"""

# --- Visualización del video ---------------------------------------------
while (cap.isOpened()):                                                 # Itero, siempre y cuando el video esté abierto
    ret, frame = cap.read()                                             # Obtengo el frame
    if ret==True:                                                       # ret indica si la lectura fue exitosa (True) o no (False)
        cv2.imshow('Frame',frame)                                       # Muestro el frame
        if cv2.waitKey(25) & 0xFF == ord('q'):                          # Corto la repoducción si se presiona la tecla "q"
            break
    else:
        break                                       # Corto la reproducción si ret=False, es decir, si hubo un error o no quedán mas frames.


# mostramos el procedimiento que se utiliza para identificar el área de juego

# --- Analizo un frame -----------------------------------------------------
cap = cv2.VideoCapture(video_person["path"])        # abro el video
frame_index = 0                                     # extraigo el primer frame
ret, frame = cap.read()                             # obtengo el frame
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# --- Aplicamos máscara para fitrar el área de juego ------------------------------------
hsv_lower=(50, 50, 50)
hsv_upper=(90, 255, 255)
frame_HSV = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
mask = cv2.inRange(frame_HSV, hsv_lower, hsv_upper)                               # genero una máscara
game_area = cv2.bitwise_and(frame, frame, mask=mask)

plt.figure()
ax1 = plt.subplot(221); plt.xticks([]), plt.yticks([]), plt.imshow(frame), plt.title('Frame inicial')
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(frame_HSV, cmap="gray"), plt.title('Frame inicial - HSV')
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(mask, cmap="gray"), plt.title('Máscara')
plt.subplot(224,sharex=ax1,sharey=ax1), plt.imshow(game_area, cmap="gray"), plt.title('Área de juego')
plt.show(block=False)

while (cap.isOpened()):                                                 # Itero, siempre y cuando el video esté abierto
    ret, frame = cap.read()                                             # Obtengo el frame
    if ret==True:                                                       # ret indica si la lectura fue exitosa (True) o no (False)
        game_area = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow('Frame',game_area)                                       # Muestro el frame
        brightLAB = cv2.cvtColor(game_area, cv2.COLOR_RGB2LAB)
        if cv2.waitKey(25) & 0xFF == ord('q'):                          # Corto la repoducción si se presiona la tecla "q"
            break
    else:
        break                                       # Corto la reproducción si ret=False, es decir, si hubo un error o no quedán mas frames.


# mostramos el procedimiento que se utiliza para identificar los dados rojos

# --- Analizo un frame -----------------------------------------------------
cap = cv2.VideoCapture(video_person["path"])        # abro el video
frame_index = 80                                     # extraigo el primer frame
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # Cambia la escala
ret, frame = cap.read()                             # obtengo el frame
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# utilizo cielab para que el fondo sea negro, umbralizo ----------------
plt.figure()
ax1 = plt.subplot(221); plt.xticks([]), plt.yticks([]), plt.imshow(frame), plt.title('Frame dados quietos')
game_area_LAB = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(game_area_LAB, cmap="gray"), plt.title('cielab-A')
plt.show(block=False)

gray = cv2.cvtColor(brightLAB, cv2.COLOR_LAB2LRGB)
gray = cv2.cvtColor(brightLAB, cv2.COLOR_RGB2GRAY)
red_mask = game_area_LAB[:, :, 1] > 150
red_mask_display = (red_mask * 255).astype(np.uint8)
umbral, thresh_imgd = cv2.threshold(red_mask_display, thresh=90, maxval=255, type=cv2.THRESH_BINARY)  # Umbralamos

plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(red_mask, cmap="gray"), plt.title('red_mask')                 # mostrar rojo
plt.subplot(224,sharex=ax1,sharey=ax1), plt.imshow(thresh_imgd, cmap="gray"), plt.title('umbral')                 # quitar
plt.show(block=False)


# mostramos el análisis exploratorio con el cual definimos las metricas,                   falta ver com se pasa a PORCENTAJE!!!

contoursd, _ = cv2.findContours(thresh_imgd, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
areas = []
valid_contours = []
for contour in contoursd:
    area = cv2.contourArea(contour)
    perimetro = cv2.arcLength(contour, True)
    if perimetro > 100:
        Fp = area / (perimetro ** 2)
        if 0.04 < Fp < 0.06:
            valid_contours.append(contour)

print(min(areas))
print(max(areas))

output_image = frame.copy()
output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
cv2.drawContours(output_image, valid_contours, -1, (255, 0, 0), thickness=2)
cv2.imshow('Contornos que cumplen la condicion', output_image)


# 
"""min_contours=5
stability_frames=9
frame_count = 0
stable_frame_count = 0
if len(valid_contours) >= min_contours:
            stable_frame_count += 1
            last_frame_with_dados = frame_count  # Registramos el último frame con los dados detectados
            last_frame_image = output_image.copy()
else:
    stable_frame_count = 0


if stable_frame_count >= stability_frames:
    print(f"Dados detectados desde el frame {frame_count - stability_frames + 1}.")
"""

# 



def funcion(video_path, min_contours=5, stability_frames=9): 
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # Cambia la escala
    ret, frame = cap.read()
    if not ret:
        print("No se pudo abrir el video")
        return
    # Definimos la máscara inicial para el área de juego
    hsv_lower = (50, 50, 50)
    hsv_upper = (90, 255, 255)
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_HSV, hsv_lower, hsv_upper)
    frame_count = 0
    stable_frame_count = 0
    last_frame_with_dados = None
    last_frame_image = None 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        game_area = cv2.bitwise_and(frame, frame, mask=mask) # Aplicamos la máscara del área de juego
        game_area_LAB = cv2.cvtColor(game_area, cv2.COLOR_BGR2LAB)
        red_mask = game_area_LAB[:, :, 1] > 150
        red_mask_display = (red_mask * 255).astype(np.uint8)
        umbral, thresh_imgd = cv2.threshold(red_mask_display, 120, 255, cv2.THRESH_BINARY)
        contoursd, _ = cv2.findContours(thresh_imgd, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # Filtramos contornos según el factor de forma
        valid_contours = []
        for contour in contoursd:
            area = cv2.contourArea(contour)
            perimetro = cv2.arcLength(contour, True)
            if perimetro > 100:
                Fp = area / (perimetro ** 2)
                if 0.04 < Fp < 0.06:                                                    # Valores para dados, pasarlo a porcentaje
                    valid_contours.append(contour)
        # Dibujamos los contornos válidos en el frame actual
        output_image = game_area.copy()
        # busco el indice de los contornos final en game_area
        cv2.drawContours(output_image, valid_contours, -1, (255, 0, 0), thickness=2)
        cv2.imshow('Contornos que cumplen la condicion', output_image)
        if len(valid_contours) >= min_contours:
            stable_frame_count += 1
            last_frame_with_dados = frame_count  # Registramos el último frame con los dados detectados
            last_frame_image = output_image.copy()
        else:
            stable_frame_count = 0
        if stable_frame_count >= stability_frames:
            print(f"Dados detectados desde el frame {frame_count - stability_frames + 1}.")
            break  # (sale si se alcanza la estabilidad)
        frame_count += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # cap.release()
    # cv2.destroyAllWindows()
    if last_frame_with_dados is not None:
        print(f"ultimo frame con dados detectados: {last_frame_with_dados}")
        
        # Extraer regiones de interés a partir de los contornos
        jugada = []
        extracted_regions = []
        for contour in valid_contours:
            # Crear máscara para cada contorno
            mask = np.zeros_like(last_frame_image[:, :, 0], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, color=255, thickness=cv2.FILLED)
            
            # Extraer región usando la máscara
            extracted_region = cv2.bitwise_and(last_frame_image, last_frame_image, mask=mask)
            extracted_regions.append(extracted_region)
            for region in extracted_regions:
                #cv2.imshow("Region extraida", extracted_region)
                hsv_lower = (0, 0, 100)
                hsv_upper = (180, 50, 255)
                region_HSV = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                # cv2.imshow('region_HSV', region_HSV)
                mask = cv2.inRange(region_HSV, hsv_lower, hsv_upper)
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                nro_dado = len(contours)
                jugada.append(nro_dado)
                #imshow(extracted_region)
            resultado = frame.copy()
            cv2.drawContours(resultado, contours, -1, (0, 0, 255), thickness=2)
            # print(nro_dado)
            cv2.imshow('Contornos que cumplen la condicion', resultado)
            #cv2.imshow('mask', mask)
            cv2.waitKey(0)  # Esperar hasta que se presione una tecla para mostrar la siguiente región
        cv2.destroyAllWindows()
        return f'Resultado jugada: {sum(jugada)}'  # Retorna las regiones extraídas como una lista
    else:
        print("No se detectaron dados en el video.")
        return None




funcion('tirada_3.mp4')

# no funciona con jugada 3 :(
# ver de ser menos estricto con el umbral



