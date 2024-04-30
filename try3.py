import cv2
import numpy as np
import RPi.GPIO as GPIO

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Motor Pins
motor1_pin1 = 2
motor1_pin2 = 3
motor2_pin1 = 4
motor2_pin2 = 5

# Set GPIO pins as output
GPIO.setup(motor1_pin1, GPIO.OUT)
GPIO.setup(motor1_pin2, GPIO.OUT)
GPIO.setup(motor2_pin1, GPIO.OUT)
GPIO.setup(motor2_pin2, GPIO.OUT)

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to show on the screen
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            # Ball detection logic
            if label == 'sports ball':
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                center_x = x + w // 2
                center_y = y + h // 2

                # Ball tracking logic (adjust as needed)
                if center_x < 300:
                    # Turn left
                    GPIO.output(motor1_pin1, GPIO.HIGH)
                    GPIO.output(motor1_pin2, GPIO.LOW)
                    GPIO.output(motor2_pin1, GPIO.LOW)
                    GPIO.output(motor2_pin2, GPIO.HIGH)
                elif center_x > 340:
                    # Turn right
                    GPIO.output(motor1_pin1, GPIO.LOW)
                    GPIO.output(motor1_pin2, GPIO.HIGH)
                    GPIO.output(motor2_pin1, GPIO.HIGH)
                    GPIO.output(motor2_pin2, GPIO.LOW)
                else:
                    # Move forward
                    GPIO.output(motor1_pin1, GPIO.HIGH)
                    GPIO.output(motor1_pin2, GPIO.LOW)
                    GPIO.output(motor2_pin1, GPIO.HIGH)
                    GPIO.output(motor2_pin2, GPIO.LOW)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()