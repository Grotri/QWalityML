import psutil
import time
import onnxruntime as ort
from ultralytics import YOLO

model = YOLO(model='runs/detect/train19/weights/best.onnx')

# Замер до инференса
start_time = time.time()
process = psutil.Process()
initial_cpu = process.cpu_percent(interval=None)
initial_mem = process.memory_info().rss / 1024**2  # в МБ

results = model.predict(source="panel_19.jpg", imgsz=416, device='cpu')

# Замер после инференса
end_time = time.time()
final_cpu = process.cpu_percent(interval=None)
final_mem = process.memory_info().rss / 1024**2

# Результаты
print(f"Время выполнения: {end_time - start_time:.2f} сек")
print(f"Использование CPU: {final_cpu - initial_cpu:.1f}%")
print(f"Потребление RAM: {final_mem - initial_mem:.2f} МБ")
print(results)