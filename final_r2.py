import cv2
import pandas as pd
import time
from ultralytics import YOLO

# Initialize the internal webcam (index 0) for ball detection
ball_cap = cv2.VideoCapture(0)
ball_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set frame width to 640
ball_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set frame height to 480

# Initialize YOLO model for detecting balls
ball_model = YOLO('last.pt')
ball_class_list = ball_model.names  # Get class names from YOLO model

# Initialize flags
ball_detected = False
silo_started = False

while True:
    # Capture frame-by-frame from the internal webcam for ball detection
    ball_ret, ball_im = ball_cap.read()
    
    if not ball_ret:
        print("Failed to grab frame from internal webcam")
        break

    # Perform object detection using YOLO for balls
    ball_results = ball_model.predict(ball_im, conf=0.6)  # Adjust confidence threshold as needed
    ball_detections = ball_results[0].boxes.data.cpu().numpy()
    px = pd.DataFrame(ball_detections).astype("int")
    
    ball_detected = False  # Reset flag for ball detection
    
    for index, row in px.iterrows():
        x1, y1, x2, y2, d = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[5])
        c = ball_class_list[d]
        
        # Filter only for the ball class
        if c.lower() == 'ball':
            ball_detected = True
            
            # Draw bounding box and label around the ball
            cv2.rectangle(ball_im, (x1, y1), (x2, y2), (0, 0, 255), 2)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(ball_im, (center_x, center_y), 3, (0, 255, 0), -1)
            cv2.putText(ball_im, f'{c}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the resulting frame with ball detection
    cv2.imshow("Ball Detection", ball_im)

    # Check if ball detection is completed and silo detection is not yet started
    if ball_detected and not silo_started:
        # Initialize the external webcam (index 1) for silo detection
        silo_cap = cv2.VideoCapture(1)
        silo_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set frame width to 640
        silo_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set frame height to 480
        
        # Initialize YOLO model for detecting silos
        silo_model = YOLO('silos-100-best.pt')
        silo_class_list = silo_model.names  # Get class names from YOLO model
        
        silo_started = True  # Update flag to indicate silo detection started

    # Perform object detection using YOLO for silos if silo detection is started
    if silo_started:
        # Capture frame-by-frame from the external webcam for silo detection
        silo_ret, silo_im = silo_cap.read()
        
        if not silo_ret:
            print("Failed to grab frame from external webcam")
            break
        
        # Perform object detection using YOLO for silos
        silo_results = silo_model.predict(silo_im, conf=0.6)  # Adjust confidence threshold as needed
        silo_detections = silo_results[0].boxes.data.cpu().numpy()
        px_silo = pd.DataFrame(silo_detections).astype("int")
        
        silo_detected = False  # Flag to track if the silo is detected
        
        for index, row in px_silo.iterrows():
            x1_silo, y1_silo, x2_silo, y2_silo, d_silo = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[5])
            c_silo = silo_class_list[d_silo]
            
            # Filter only for the silo class
            if c_silo.lower() == 'silo':
                silo_detected = True
                
                # Draw bounding box and label around the silo
                cv2.rectangle(silo_im, (x1_silo, y1_silo), (x2_silo, y2_silo), (255, 0, 0), 2)
                cv2.putText(silo_im, f'{c_silo}', (x1_silo, y1_silo - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the resulting frame with silo detection
        cv2.imshow("Silo Detection", silo_im)
    
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # 'q' key or 'Esc' key
        break

# Release the captures and close all OpenCV windows
ball_cap.release()
if silo_started:
    silo_cap.release()
cv2.destroyAllWindows()
