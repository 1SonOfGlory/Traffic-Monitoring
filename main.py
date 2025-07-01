import cv2
from ultralytics import YOLO
import easyocr

# Load YOLOv8 model (you can use 'yolov8n.pt' or a custom one)
model = YOLO('yolov8n.pt')

# Initialize OCR reader
reader = easyocr.Reader(['en'])

# Load image
img_path = r"C:\Users\Surface_Book\Documents\JabessPortfolio\num plate.png"  # or use a video later
frame = cv2.imread(img_path)

# Run detection
results = model.predict(source=frame, conf=0.3)

# Loop through detected objects
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = frame[y1:y2, x1:x2]

        # OCR on the cropped region
        ocr_result = reader.readtext(cropped)
        if ocr_result:
            text = ocr_result[0][-2]
            print("Detected Plate:", text)

            # Annotate image
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0,255,0), 2)

# Display the result
cv2.imshow("License Plate Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()



"C:\Users\Surface_Book\Documents\JabessPortfolio\Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1).mp4"