import cv2
import torch
import numpy as np
import os
from tensorflow.keras.models import load_model
from twilio.rest import Client  # Import Twilio Client

# Twilio configuration
TWILIO_SID = 'AC49741252aa0f6b98b5f636aeabdf028e'  # Your Twilio Account SID
TWILIO_AUTH_TOKEN = '07dbdf771b77d4c444d925ff5d818245'  # Your Twilio Auth Token
TWILIO_PHONE_NUMBER = '+12317946349'  # Your Twilio phone number
TO_PHONE_NUMBER = '+918248972523'  # The phone number to send the message to

# Initialize Twilio Client
twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# Load models
vehicle_model = torch.hub.load('ultralytics/yolov5', 'yolov5n')  # Vehicle detection model
trash_model = load_model('trash_classifier_model.h5')  # Trash detection model

output_dir = "trash_detected"
os.makedirs(output_dir, exist_ok=True)

# Open the video feed (replace 'video.mp4' with 0 for webcam)
video_path = 'video4.mp4'  # Change to 0 for real-time webcam
cap = cv2.VideoCapture(video_path)
frame_count = 0
message_sent = False  # Flag to track if message has been sent

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends
    
    frame_count += 1
    results_vehicle = vehicle_model(frame)
    
    vehicle_detected = False
    vehicle_snapshot = None
    
    for result in results_vehicle.xyxy[0]:  # Correct way to access YOLO detections
        x1, y1, x2, y2, conf, cls = result.tolist()
        if conf > 0.5 and int(cls) in [2, 3, 5, 7]:  # Check if detected object is a vehicle
            vehicle_detected = True
            vehicle_snapshot = frame[int(y1):int(y2), int(x1):int(x2)]  # Crop vehicle image
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, "Vehicle", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Define the area below the vehicle to check for trash (increased height to 300 pixels)
            trash_area_height = 300
            trash_area_y1 = int(y2)
            trash_area_y2 = min(int(y2 + trash_area_height), frame.shape[0])  # Ensure within frame bounds
            
            # Make sure the area is valid
            if trash_area_y1 < trash_area_y2:
                trash_area = frame[trash_area_y1:trash_area_y2, int(x1):int(x2)]

                # Check if trash_area is not empty before resizing
                if trash_area.size > 0:
                    # Resize and normalize the trash area for the model
                    resized_trash_area = cv2.resize(trash_area, (128, 128))  # Match the model input size
                    normalized_trash_area = resized_trash_area / 255.0  # Normalize pixel values
                    input_trash_area = np.expand_dims(normalized_trash_area, axis=0)  # Add batch dimension

                    # Predict if trash is present
                    trash_prediction = trash_model.predict(input_trash_area)
                    if trash_prediction[0][0] > 0.5 and not message_sent:  # Adjust threshold and check flag
                        print("Trash Detected from Vehicle")
                        cv2.putText(frame, "Trash Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        snapshot_filename = os.path.join(output_dir, f'trash_detected_{frame_count}.jpg')
                        cv2.imwrite(snapshot_filename, frame)
                        print(f'Snapshot saved: {snapshot_filename}')

                        # Send message through Twilio
                        message = twilio_client.messages.create(
                            body="Trash detected from this vehicle. Rs.100 will be taken as fine.",
                            from_=TWILIO_PHONE_NUMBER,
                            to=TO_PHONE_NUMBER
                        )
                        print(f'Message sent: {message.sid}')
                        message_sent = True  # Set flag to true after sending message
                    else:
                        print("No Trash Detected")
                else:
                    print("No valid trash area detected.")
    
    # Reset the message_sent flag if no vehicle is detected
    if not vehicle_detected:
        message_sent = False
    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
