from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    results = model.train(
        data='pothole.yaml',
        imgsz=640,
        epochs=50,
        batch=8,
        name='yolov8n_custom'
    )
