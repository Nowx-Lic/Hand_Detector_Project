import cv2
from cvzone.HandTrackingModule import HandDetector

def main():
    # Inicializar la cámara (1 o 2 para Camo, 0 para la integrada)
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Configurar el detector de manos (detecta hasta 1 mano con alta confianza)
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    print("[INFO] Detector de manos iniciado. Presiona ESC para salir.")

    while True:
        ret, cap_frame = cap.read()
        if not ret:
            break

        # Espejo para una interacción más natural
        frame = cv2.flip(cap_frame, 1)

        # Buscar manos y dibujar el esqueleto automáticamente
        hands, frame = detector.findHands(frame, draw=True)

        if hands:
            hand = hands[0]
            # Obtener qué dedos están levantados (lista de 5 elementos [pulgar, índice, medio, anular, meñique])
            dedos = detector.fingersUp(hand)
            
            cv2.putText(frame, f"Dedos arriba: {dedos}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Muestra tu mano a la camara", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # Mostrar ventana
        cv2.imshow("Detector de Manos - GitHub Project", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()