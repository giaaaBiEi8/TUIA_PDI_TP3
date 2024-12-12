import cv2
import numpy as np

def load_video(video_path):
    """cargamos video y chequeamos que funcione"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        print("No se pudo abrir el video.")
    return cap

def create_game_area_mask(frame):
    """crea una mascara para los valores verdes de la tela"""
    hsv_lower = (50, 50, 50)
    hsv_upper = (90, 255, 255)
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(frame_HSV, hsv_lower, hsv_upper)

def detect_red_dice(game_area):
    """detecta el rojo de los dados thresholdeando"""
    # usamos lab ya que el canal A usa verde-rojo
    game_area_LAB = cv2.cvtColor(game_area, cv2.COLOR_BGR2LAB)

    red_mask = game_area_LAB[:, :, 1] > 140
    red_mask_display = (red_mask * 255).astype(np.uint8)

    return cv2.findContours(red_mask_display, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

def filter_dice_contours(contours):
    """filtramos los contornos usando factor de forma."""
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 100:
            shape_factor = area / (perimeter ** 2)
            if 0.035 < shape_factor < 0.065:  # masomenos cuadrado
                valid_contours.append(contour)
    return valid_contours

def count_dots_on_die(die_region):
    """contamos los puntos blancos"""
    # detectamos puntos con tonalidades blancas
    hsv_lower = (0, 0, 100)
    hsv_upper = (180, 50, 255)
    region_HSV = cv2.cvtColor(die_region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(region_HSV, hsv_lower, hsv_upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return len(contours)


def process_video(video_path, output_path="output.avi", min_contours=5, stability_frames=20):
    """función main"""

    try:
        cap = load_video(video_path)
    except ValueError as e:
        print(e)
        return
    
    # Initialize video
    ret, first_frame = cap.read()
    if not ret:
        print("Error al leer el primer frame del video.")
        cap.release()
        return
    
    game_area_mask = create_game_area_mask(first_frame)

    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
    fps = int(cap.get(cv2.CAP_PROP_FPS))  

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
    # out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Track stability
    frame_count = 0
    stable_frame_count = 0
    last_frame_with_dice = None
    last_frame_image = None
    last_contours = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        game_area = cv2.bitwise_and(frame, frame, mask=game_area_mask)
        contours, _ = detect_red_dice(game_area)
        valid_contours = filter_dice_contours(contours)
        
        # Draw contours for visualization
        output_image = game_area.copy()
        cv2.drawContours(output_image, valid_contours, -1, (255, 0, 0), thickness=2)
        out.write(output_image)

        # Check stability
        if len(valid_contours) >= min_contours:
            stable_frame_count += 1
            last_frame_with_dice = frame_count
            last_frame_image = output_image.copy()
            last_contours = valid_contours
        else:
            stable_frame_count = 0
            
        if stable_frame_count >= stability_frames:
            print(f"Dados estables desde el frame {frame_count - stability_frames + 1}")
            break
            
        frame_count += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    # Procesamos rtdo final
    if last_frame_with_dice is not None:
        # Detect values on the dice
        dice_values = []
        for contour in last_contours:
            # Extrae cada región de los dados
            mask = np.zeros_like(last_frame_image[:, :, 0], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, color=255, thickness=cv2.FILLED)
            die_region = cv2.bitwise_and(last_frame_image, last_frame_image, mask=mask)
            
            # Cuenta puntos
            dots = count_dots_on_die(die_region)
            dice_values.append(dots)
        
        # Draw BoundingRect and numbers on the last frame
        for i, contour in enumerate(last_contours):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(last_frame_image, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
            cv2.putText(last_frame_image, str(dice_values[i]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(255, 0, 0), thickness=2)
            
        # cv2.destroyAllWindows()
        cv2.imshow('Último frame con BoundingRect', last_frame_image)
        out.write(last_frame_image)                                                              # mostrarlo varias vecesssss

        # Wait for user to press a key
        # print("Presiona cualquier tecla para cerrar...")
        print(f'Suma total: {sum(dice_values)}')
        print("Video guardado en:", output_path)
        cv2.waitKey(0)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()



process_video('tirada_3.mp4', 'output.avi', min_contours=5, stability_frames=20)


