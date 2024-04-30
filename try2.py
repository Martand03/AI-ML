import cv2
import numpy as np
import pandas as pd
import math
import serial
import time

# Load YOLOv4-tiny model
net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Focal length of the camera in pixels and height of a person in meters
focal_length = 110  # Adjust this value based on your camera and setup
average_person_height = 1.7  # meters, adjust as needed

def calculate_distance(box_height):
    # Calculate distance in centimeters using focal length and known average person height
    return (average_person_height * focal_length * 100) / box_height

def calculate_offset(prev_center, center):
    # Calculate the offset in both horizontal (x-axis) and vertical (y-axis) directions
    offset_x = center[0] - prev_center[0]
    offset_y = center[1] - prev_center[1]
    return offset_x, offset_y

def process_frame(frame, outs, prev_centers):
    detected_objects = []
    frame_height, frame_width = frame.shape[:2]
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5 and classes[class_id] == 'person':  # Adjust confidence threshold and class name as needed
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                box_width = int(detection[2] * frame_width)
                box_height = int(detection[3] * frame_height)
                
                x1 = int(center_x - box_width / 2)
                y1 = int(center_y - box_height / 2)
                x2 = x1 + box_width
                y2 = y1 + box_height
                
                distance = calculate_distance(box_height)
                center = (center_x, center_y)
                
                if class_id in prev_centers:
                    # Calculate the offset from the previous center coordinates
                    offset_x, offset_y = calculate_offset(prev_centers[class_id], center)
                    detected_objects.append((classes[class_id], confidence, distance, offset_x, offset_y, center, (x1, y1, x2, y2)))
                
                prev_centers[class_id] = center  # Update previous center coordinates
    
    return detected_objects, prev_centers

def send_signal(arduino, signal):
    arduino.write(signal.encode())

# Arduino communication setup
arduino_port = 'COM12'  # Update with your Arduino port
baud_rate = 9600
arduino = serial.Serial(arduino_port, baud_rate)
time.sleep(2)  # Wait for Arduino to initialize

cap = cv2.VideoCapture(0)  # Change to video file path if needed
prev_centers = {}  # Dictionary to store previous center coordinates for each class
person_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (1020, 500))
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())
    
    detected_objects, prev_centers = process_frame(frame, outs, prev_centers)
    
    for obj, _, distance, offset_x, offset_y, center, bbox in detected_objects:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green rectangle around detected object
        cv2.putText(frame, f'{obj} | Distance: {distance:.2f} cm | OffsetX: {offset_x} px | OffsetY: {offset_y} px', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Display object, distance, and offsets
        
        # Draw center point (circle or dot)
        cv2.circle(frame, center, 5, (255, 0, 0), -1)  # Blue color
        
        person_detected = True
    
    if not detected_objects and person_detected:
        # No person detected, send signal to turn off LED
        send_signal(arduino, '0')
        person_detected = False
    elif detected_objects and not person_detected:
        # Person detected, send signal to turn on LED
        send_signal(arduino, '1')
        person_detected = True
        
    cv2.imshow("YOLOv4-tiny", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
        break

cap.release()
cv2.destroyAllWindows()
