from ultralytics import YOLO
from parinya import LINE
import time
import cv2

line = LINE('js6HpD0JbTXUjNRlvMX4lhpRSgKdEyOsYXMeRvVFq4J')
cap = cv2.VideoCapture(0)
model = YOLO('best.pt')
fps = 24
delay = 1 / fps

while True:
    start_time = time.time()
    _, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictions = model.predict(source=frame_rgb, imgsz=640, conf=0.5, show=False)
    
    if predictions and len(predictions[0]) > 0:
        try:
            line.sendimage(frame_rgb)
            line.sendtext("Detection of watch!")
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการส่งรูปไปที่ LINE: {e}")
    cv2.imshow('Real-time Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elapsed_time = time.time() - start_time
    if elapsed_time < delay:
        time.sleep(delay - elapsed_time)

cv2.destroyAllWindows()
cap.release()
