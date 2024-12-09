# **Detección Automática de Dados en Video**

Este proyecto implementa un sistema para procesar un video, detectar dados rojos en una mesa verde, y contar los puntos visibles en los dados una vez que se detienen. Utiliza técnicas de procesamiento de imágenes con OpenCV y está diseñado para detectar estabilidad antes de realizar el conteo final.

---

## **Características del Script**
- **Carga de video**: Valida que el archivo de video sea accesible y carga los cuadros uno por uno.
- **Detección del área de juego**: Identifica la mesa verde utilizando máscaras en el espacio de color HSV.
- **Detección de dados rojos**: Utiliza el espacio de color LAB para resaltar áreas rojas y detecta contornos.
- **Filtrado de contornos**: Evalúa los contornos según el área y el perímetro para identificar formas similares a dados.
- **Conteo de puntos**: Detecta puntos blancos en los dados utilizando máscaras en el espacio de color HSV.
- **Estabilidad**: Garantiza que los dados estén en reposo antes de realizar el conteo.
- **Salidas visuales**: Permite la visualización de los contornos detectados y los resultados de procesamiento.

---

## **Requisitos**

### **Software**
- Python 3.7 o superior
- OpenCV
- NumPy

### **Librerías necesarias**
Instale las dependencias con el siguiente comando:

```
bash
pip install opencv-python-headless numpy
```

### **Estructura del Script**
#### **Funciones Principales**

##### *load_video(video_path)*
Carga el video y valida su accesibilidad. Devuelve el objeto de captura y el primer cuadro.

##### *create_game_area_mask(frame)*
Genera una máscara que delimita el área de juego (mesa verde) utilizando el espacio HSV.

##### *detect_red_dice(game_area)*
Detecta los dados rojos en el área de juego utilizando el espacio LAB.

##### *filter_dice_contours(contours)*
Filtra contornos para identificar formas similares a dados basándose en el área y el perímetro.

##### *count_dots_on_die(die_region)*
Cuenta los puntos blancos en un dado utilizando el espacio HSV.

##### *process_video(video_path, min_contours=5, stability_frames=20)*
Controla el flujo principal: procesa el video cuadro a cuadro, verifica estabilidad, y cuenta los puntos cuando los dados están en reposo.

### **Uso**

Clonar el repositorio:

```bash
git clone https://github.com/giaaaBiEi8/TUIA_PDI_TP3.git
cd TUIA_PDI_TP3
```
Ejecutar el script: Asegúrese de que el video esté en la misma carpeta que el script o proporcione la ruta completa.

```bash
python programa.py --video <ruta_al_video>
```


### **Autores**

##### **Borgo, Iair**
##### **Nardi, Albano**
##### **Nardi, Gianella**
