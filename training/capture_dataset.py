import picamera
import time

with picamera.PiCamera() as cam:
    cam.resolution = (640, 480)
    cam.start_preview()
    contador = 0
    while True:
        input(f"Capturar {contador}")
        cam.capture(f'pilha_{contador}.jpg')
        contador += 1
        print("Capturei", contador -1)
