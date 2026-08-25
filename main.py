import cv2
import numpy as np
import os
from cvzone.HandTrackingModule import HandDetector

def main():
    # Inicializar la cámara (1 o 2 para Camo, 0 para la integrada)
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Ventana cómoda y redimensionable
    cv2.namedWindow("Gestos con Dos Manos - Memes", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gestos con Dos Manos - Memes", 800, 600)

    # Configurar el detector para que detecte hasta 2 manos simultáneas
    detector = HandDetector(detectionCon=0.8, maxHands=2)

    # Cargar imagen del hámster (busca tanto .jpg como .png)
    img_hamster = None
    for ext in [".jpg", ".jpeg", ".png"]:
        nombre = "hamster" + ext
        if os.path.exists(nombre):
            img_hamster = cv2.imread(nombre)
            if img_hamster is not None:
                print(f"[OK] Cargado con exito: {nombre}")
                break
    if img_hamster is None:
        print("[ADVERTENCIA] No se encontro 'hamster' en la carpeta.")

    # Cargar imagen del gato (busca tanto .jpg como .png)
    img_gato = None
    for ext in [".jpg", ".jpeg", ".png"]:
        nombre = "gato" + ext
        if os.path.exists(nombre):
            img_gato = cv2.imread(nombre)
            if img_gato is not None:
                print(f"[OK] Cargado con exito: {nombre}")
                break
    if img_gato is None:
        print("[ADVERTENCIA] No se encontro 'gato' en la carpeta.")

    print("[INFO] Sistema de doble mano activo. Presiona ESC para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Espejo para que el movimiento sea natural
        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape

        # Buscar manos (hasta 2) y dibujar el esqueleto detallado
        hands, frame = detector.findHands(frame, draw=True)

        estado_texto = "Muestra tus manos a la camara"

        if hands:
            # Recorrer cada mano detectada en pantalla de forma independiente
            for hand in hands:
                dedos = detector.fingersUp(hand)  # [pulgar, índice, medio, anular, meñique]
                
                # Coordenadas de la palma (punto 9) para anclar la imagen sobre esa mano específica
                lmList = hand["lmList"]
                centro_x, centro_y = lmList[9][0], lmList[9][1]

                imagen_a_mostrar = None

                # Gesto 1: Señal de paz ✌️ ([0, 1, 1, 0, 0]) -> Hamster
                if dedos == [0, 1, 1, 0, 0] and img_hamster is not None:
                    imagen_a_mostrar = img_hamster
                    estado_texto = "Gesto detectado: Paz / Dedo medio"
                
                # Gesto 2: Dedo del medio ([0, 0, 1, 0, 0]) -> Gato
                elif dedos == [0, 0, 1, 0, 0] and img_gato is not None:
                    imagen_a_mostrar = img_gato
                    estado_texto = "Gesto detectado: Paz / Dedo medio"

                # Si esta mano en particular hizo un gesto válido, superponemos su imagen encima
                if imagen_a_mostrar is not None:
                    tamano = 130
                    img_resized = cv2.resize(imagen_a_mostrar, (tamano, tamano))

                    # Calcular la posición centrada arriba de esta mano específica
                    y1 = max(0, centro_y - tamano - 15)
                    y2 = min(H, y1 + tamano)
                    x1 = max(0, centro_x - tamano // 2)
                    x2 = min(W, x1 + tamano)

                    ph, pw = y2 - y1, x2 - x1
                    if ph > 0 and pw > 0:
                        img_crop = img_resized[:ph, :pw]
                        
                        # Superposición translúcida sobre el video
                        roi = frame[y1:y2, x1:x2]
                        mezcla = cv2.addWeighted(roi, 0.1, img_crop, 0.9, 0)
                        frame[y1:y2, x1:x2] = mezcla

        # Texto de estado en la parte superior
        cv2.putText(frame, estado_texto, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # Mostrar la ventana final
        cv2.imshow("Gestos con Dos Manos - Memes", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()