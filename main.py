import cv2
import easyocr
import json
import time
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import pika
from plate_utils import extract_best_plate_read, should_send_plate

# ==========================================
# 1. CONFIGURATION
# ==========================================
# ESP32-CAM Stream
ESP32_URL = "http://192.168.1.100:81/stream"

# RabbitMQ Settings
RMQ_HOST = "localhost" # Change to your RabbitMQ server IP
RMQ_PORT = 5672
RMQ_USER = "guest"     # Default username
RMQ_PASS = "guest"     # Default password
RMQ_QUEUE = "alpr_queue"

# ALPR Settings
COOLDOWN_TIME = 10 # Seconds to wait before sending the same plate again

# Hugging Face Model Settings
HF_REPO_ID = "Rattatammanoon/hurricane-od-thai-plate-detector"
HF_MODEL_FILENAME = "HurricaneOD_beta.pt"

# ==========================================
# 2. INITIALIZATION
# ==========================================
print("Loading OCR (Thai/English)...")
reader = easyocr.Reader(['th', 'en'], gpu=True) 

print("Downloading/Loading Thai YOLO model from Hugging Face...")

# Download once, then use local cache on subsequent runs
model_path = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename=HF_MODEL_FILENAME
)

model = YOLO(model_path)

# Initialize RabbitMQ Connection
print("Connecting to RabbitMQ...")
def connect_rabbitmq():
    credentials = pika.PlainCredentials(RMQ_USER, RMQ_PASS)
    parameters = pika.ConnectionParameters(RMQ_HOST, RMQ_PORT, '/', credentials)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    
    # Declare the queue as durable (survives RabbitMQ restarts)
    channel.queue_declare(queue=RMQ_QUEUE, durable=True)
    return connection, channel

try:
    rmq_conn, rmq_channel = connect_rabbitmq()
    print("RabbitMQ Connected successfully!")
except Exception as e:
    print(f"RabbitMQ Connection failed: {e}")
    exit(1)

# ==========================================
# 3. MAIN LOOP
# ==========================================
cap = cv2.VideoCapture(ESP32_URL)
frame_skip = 5 
frame_count = 0

last_seen_plates = {} 

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        time.sleep(1)
        continue
        
    frame_count += 1
    
    # Process every Nth frame
    if frame_count % frame_skip == 0:
        results = model(frame, verbose=False)[0]
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            
            if confidence > 0.5: 
                plate_crop = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                
                # Get OCR result
                ocr_results = reader.readtext(gray, detail=1) 

                best_plate = extract_best_plate_read(ocr_results)
                if best_plate:
                    plate_text, ocr_conf = best_plate
                    current_time = time.time()

                    if should_send_plate(plate_text, last_seen_plates, current_time, COOLDOWN_TIME):
                        print(f"New Plate Detected: {plate_text} (Conf: {ocr_conf:.2f})")

                        last_seen_plates[plate_text] = current_time

                        payload = {
                            "timestamp": int(current_time),
                            "plate_text": plate_text,
                            "yolo_confidence": float(confidence),
                            "ocr_confidence": float(ocr_conf),
                            "camera_id": "esp32_cam_gate_1"
                        }

                        # Convert JSON to bytes using UTF-8 to preserve Thai characters
                        json_payload = json.dumps(payload, ensure_ascii=False).encode('utf-8')

                        try:
                            # Publish to RabbitMQ
                            rmq_channel.basic_publish(
                                exchange='',
                                routing_key=RMQ_QUEUE,
                                body=json_payload,
                                properties=pika.BasicProperties(
                                    delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE # Make message persistent
                                )
                            )
                            print(f"Published to RabbitMQ >> {json.dumps(payload, ensure_ascii=False)}")
                        except pika.exceptions.StreamLostError:
                            # Handle connection drops gracefully
                            print("RabbitMQ connection lost! Reconnecting...")
                            rmq_conn, rmq_channel = connect_rabbitmq()
                            rmq_channel.basic_publish(
                                exchange='',
                                routing_key=RMQ_QUEUE,
                                body=json_payload,
                                properties=pika.BasicProperties(
                                    delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE
                                )
                            )

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("ESP32-CAM ALPR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
if rmq_conn and rmq_conn.is_open:
    rmq_conn.close()