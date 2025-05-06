import ultralytics


if __name__ == '__main__':
    model = ultralytics.YOLO("yolo11m.yaml")

    train_results = model.train(
        data="datasets/dataset.yaml",
        epochs=100,
        imgsz=400,
        device="cuda",
        batch=28
    )

    metrics = model.val()

    print(f'Model saved at: {model.export(format="onnx")}')
