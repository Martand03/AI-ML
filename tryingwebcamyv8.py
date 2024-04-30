import cv2
import argparse
from ultralytics import YOLO
import supervision as sv

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Yolov8 live")
    parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.webcam_resolution[1])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.webcam_resolution[0])

    model = YOLO("last.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    while True:
        ret, frame = cap.read()
        cv2.imshow('last.pt', frame)

        result = model(frame)  # Directly call the YOLO model on the frame
        if isinstance(result, list) and len(result) > 0:
            detections = sv.Detections.from_yolov5(result[0])  # Convert YOLOv8 detections to Supervision format

            frame = box_annotator.annotate(scene=frame, detections=detections)

        if cv2.waitKey(30) == 27:  # Break loop on 'Esc' key
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
