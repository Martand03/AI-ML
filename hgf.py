import cv2
import pandas as pd
import serial
from ultralytics import YOLO

# Initialize the webcam
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set frame width to 640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set frame height to 480


# Initialize YOLO model
model = YOLO('last.pt')
class_list = model.names  # Get class names from YOLO model

# Initialize serial communication with Arduino
arduino = serial.Serial('COM6', 9600)

while True:
    ret, im = cap.read()  # Capture frame-by-frame from the webcam
    
    if not ret:
        print("Failed to grab frame")
        break

    # Perform object detection using YOLO
    results = model.predict(im, conf=0.6)  # Adjust confidence threshold as needed
    detections = results[0].boxes.data.cpu().numpy()
    px = pd.DataFrame(detections).astype("int")
    
    ball_detected = False  # Flag to track if the ball is detected
    
    for index, row in px.iterrows():
        x1, y1, x2, y2, d = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[5])
        c = class_list[d]
        
        # Filter only for the ball class
        if c.lower() == 'ball':
            ball_detected = True
            
            # Draw bounding box and label around the ball
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(im, (center_x, center_y), 3, (0, 255, 0), -1)
            cv2.putText(im, f'{c}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Calculate horizontal distance between red and green points
            distance = center_x - 320  # Assuming the center of the frame is at x=320 for 2x2 grid
            text_color = (0, 255, 0) if distance < 0 else (0, 0, 255)  # Green for negative, red for positive
            
            # Display the area of ball detection frame
            area = (x2 - x1) * (y2 - y1)
            cv2.putText(im, f'Area: {area}', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Send movement commands to Arduino based on distance
            if area > 24000:
                action_text = "Stop Bot"
                arduino.write(b'S')
            elif abs(distance) < 140:
                action_text = "Move Forward"
                arduino.write(b'F')  # Send 'F' for moving forward to Arduino
            elif distance < -140:
                action_text = "Move Left"
                arduino.write(b'L')  # Send 'L' for moving left to Arduino
            else:
                action_text = "Move Right"
                arduino.write(b'R')  # Send 'R' for moving right to Arduino
            
            cv2.putText(im, f'Distance: {distance}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            cv2.putText(im, action_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw grid lines for 2x2 grid
    cv2.line(im, (320, 0), (320, 480), (255, 255, 255), 2)  # Vertical line
    cv2.line(im, (0, 240), (640, 240), (255, 255, 255), 2)  # Horizontal line

    # Display the resulting frame
    cv2.imshow("Camera", im)
    
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # 'q' key or 'Esc' key
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
