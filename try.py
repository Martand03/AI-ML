from ultralytics import YOLO
import cv2
import pandas as pd
import math

model = YOLO('yolov8s.pt')
my_file = open("coco.txt", "r")  # Update file path as needed
class_list = my_file.read().split("\n")

# Assuming the average height of a person is 1.7 meters and focal length is 1000 pixels
known_person_height = 1.7  # meters
focal_length = 100  # Focal length of the camera in pixels

def calculate_distance(person_center, bounding_box_height):
    distance = (known_person_height * focal_length) / bounding_box_height
    return distance

def process_frame(frame, results):
    detected_people = []
    for result in results:
        boxes = result.boxes
        px = pd.DataFrame(boxes.data).astype("int")
        
        for _, row in px.iterrows():
            x1, y1, x2, y2 = row[:4]
            bounding_box_height = y2 - y1  # Calculate bounding box height for a person
            person_center = ((x1 + x2) // 2, (y1 + y2) // 2)  # Calculate center of person
            conf = row[4]  # Confidence score if available
            d = row[5]  # Class index if available
            c = class_list[d] if d < len(class_list) else 'Unknown'
            
            if c == 'person':  # Check if the detected object is a person
                # Call calculate_distance function with bounding box height
                distance = calculate_distance(person_center, bounding_box_height)
                detected_people.append((c, conf, distance, person_center))  # Include person center coordinates
    
    return detected_people

cv2.namedWindow('RGB')
cap = cv2.VideoCapture(0)  # Change to video file path if needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    detected_people = process_frame(frame, results)
    
    for obj, _, distance, center in detected_people:
        print(f"Detected: {obj} | X: {center[0]} | Y: {center[1]} | Distance: {distance:.2f} meters")
        # Perform actions based on person coordinates (center)
        # For example, adjust camera settings, alert if too close, etc.
    
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
        break

cap.release()
cv2.destroyAllWindows()


# from ultralytics import YOLO
# import cv2
# import pandas as pd
# import math

# model = YOLO('yolov8s.pt')
# my_file = open("coco.txt", "r")  # Update file path as needed
# class_list = my_file.read().split("\n")

# # Focal length of the camera in pixels and height of a person in meters
# focal_length = 1000  # Adjust this value based on your camera and setup
# average_person_height = 1.7  # meters, adjust as needed

# def calculate_distance(box_height):
#     # Calculate distance in centimeters using focal length and known average person height
#     return (average_person_height * focal_length * 100) / box_height

# def process_frame(frame, results):
#     detected_people = []
#     for result in results:
#         boxes = result.boxes
#         px = pd.DataFrame(boxes.data).astype("int")
        
#         for _, row in px.iterrows():
#             x1, y1, x2, y2 = row[:4]
#             box_height = y2 - y1  # Calculate bounding box height
#             center = ((x1 + x2) // 2, (y1 + y2) // 2)  # Calculate center of person
#             conf = row[4]  # Confidence score if available
#             d = row[5]  # Class index if available
#             c = class_list[d] if d < len(class_list) else 'Unknown'
            
#             if c == 'person':  # Check if the detected object is a person
#                 # Call calculate_distance function with bounding box height
#                 distance = calculate_distance(box_height)
#                 detected_people.append((c, conf, distance, center, (x1, y1, x2, y2)))  # Include bounding box coordinates
    
#     return detected_people

# cv2.namedWindow('RGB')
# cap = cv2.VideoCapture(0)  # Change to video file path if needed

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     frame = cv2.resize(frame, (1020, 500))
#     results = model.predict(frame)
#     detected_people = process_frame(frame, results)
    
#     for obj, _, distance, center, bbox in detected_people:
#         x1, y1, x2, y2 = bbox
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green rectangle around detected person
#         cv2.putText(frame, f'{obj} | Distance: {distance:.2f} cm', (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Display object and distance
        
#     cv2.imshow("RGB", frame)
#     if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
#         break

# cap.release()
# cv2.destroyAllWindows()
