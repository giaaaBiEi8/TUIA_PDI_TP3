import cv2
import numpy as np
import argparse
import os

def load_video(video_path):
    """cargamos video y fijamos que funciones"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("No se pudo abrir el video.")
    return cap, frame

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

def save_processed_video(video_path, output_path, min_contours=5, stability_frames=20):
    """Processes the video and saves the output with bounding boxes and dice numbers, cropped to the playing area."""
    # Initialize video
    cap, first_frame = load_video(video_path)
    if first_frame is None:
        return "No se pudo cargar el video."

    # Create the game area mask
    game_area_mask = create_game_area_mask(first_frame)

    # Determine cropping boundaries based on the binary mask
    y_indices, x_indices = np.where(game_area_mask > 0)
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    # Prepare video writer
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (x_max - x_min, y_max - y_min))

    # Track stability
    stable_frame_count = 0
    last_contours = None
    last_frame_image = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Crop frame to game area
        cropped_frame = frame[y_min:y_max, x_min:x_max]

        # Process cropped frame
        contours, _ = detect_red_dice(cropped_frame)
        valid_contours = filter_dice_contours(contours)

        # Draw contours for visualization
        output_image = cropped_frame.copy()
        cv2.drawContours(output_image, valid_contours, -1, (255, 0, 0), thickness=2)

        # Check stability
        if len(valid_contours) >= min_contours:
            stable_frame_count += 1
            last_frame_image = output_image.copy()
            last_contours = valid_contours
        else:
            stable_frame_count = 0

        # Draw bounding boxes and numbers if stable
        if last_contours and stable_frame_count >= stability_frames:
            dice_values = []
            for contour in last_contours:
                # Extract each dice region
                mask = np.zeros_like(last_frame_image[:, :, 0], dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, color=255, thickness=cv2.FILLED)
                die_region = cv2.bitwise_and(last_frame_image, last_frame_image, mask=mask)

                # Count dots
                dots = count_dots_on_die(die_region)
                dice_values.append(dots)

            # Draw BoundingRect and numbers on the frame
            for i, contour in enumerate(last_contours):
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
                cv2.putText(output_image, str(dice_values[i]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=(255, 0, 0), thickness=2)

        # Write frame to output video
        out.write(output_image)

        # Display for debugging purposes (optional, can be commented out)
        cv2.imshow('Procesando video', output_image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return "Video procesado y guardado exitosamente."

def main():
    parser = argparse.ArgumentParser(description="Process a video to track and annotate dice movements.")
    parser.add_argument('video_path', type=str, help="Path to the input video.")
    args = parser.parse_args()

    video_path = args.video_path
    if not os.path.exists(video_path):
        print("Error: The specified video file does not exist.")
        return

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(os.path.dirname(video_path), f"{base_name}_save.mp4")

    result = save_processed_video(video_path, output_path)
    print(result)

if __name__ == "__main__":
    main()