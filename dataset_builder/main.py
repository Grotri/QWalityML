import pandas as pd
import json
import random
import math
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import gc
import yaml

# Конфигурация
INPUT_CSV = "cells.csv"
OUTPUT_DIR = Path("output")
CELL_SOURCE_DIR = Path("cells")
PANEL_SIZE = (416, 416)
DEFECT_PERCENT = 20
NUM_PANELS = 100
MARGIN = 20
CELL_MARGIN = 5  # Отступ между ячейками

# Порядок колонок в CSV
COLUMN_ORDER = [
    'filename',
    'confidence',
    'panel_type',
    'defect_type'
]

# Категории дефектов
DEFECT_CATEGORIES = [
    {"id": 1, "name": "scratch"},
    {"id": 2, "name": "crack"},
    {"id": 3, "name": "shunt"},
    {"id": 4, "name": "breakdown"},
    {"id": 5, "name": "degradation"},
    {"id": 6, "name": "no_defect"}
]

# Цвета для разных категорий дефектов (кроме no_defect)
COLORS = {
    1: "red",    # scratch
    2: "blue",   # crack
    3: "green",  # shunt
    4: "yellow", # breakdown
    5: "purple"  # degradation
}

coco_data = {
    "info": {"description": "Solar Panel Dataset"},
    "licenses": [],
    "categories": DEFECT_CATEGORIES,
    "images": [],
    "annotations": []
}

# Создание директорий
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "images" / "train").mkdir(exist_ok=True, parents=True)
(OUTPUT_DIR / "images" / "val").mkdir(exist_ok=True, parents=True)
(OUTPUT_DIR / "labels" / "train").mkdir(exist_ok=True, parents=True)
(OUTPUT_DIR / "labels" / "val").mkdir(exist_ok=True, parents=True)
(OUTPUT_DIR / "originals").mkdir(exist_ok=True)


def get_configurations(total=60):
    configs = []
    for i in range(1, total + 1):
        if total % i == 0:
            j = total // i
            if 3 <= i <= 15 and 3 <= j <= 15 and i <= j + 1:
                configs.append((i, j))
    return configs


def load_and_preprocess_cells(df):
    cells = {"mono": [], "poly": []}
    category_map = {cat['name']: cat['id'] for cat in DEFECT_CATEGORIES}

    for _, row in df.iterrows():
        filename = row[0]
        panel_type = row[2].lower()
        defect_type = str(row[3]).lower().strip() if len(row) > 3 else 'none'

        if defect_type == 'none' or defect_type == 'nan':
            category_id = category_map['no_defect']
        else:
            category_id = category_map.get(defect_type, category_map['no_defect'])

        cells[panel_type].append({
            "path": CELL_SOURCE_DIR / filename,
            "category_id": category_id
        })
    return cells


def rotate_point(x, y, cx, cy, angle_rad):
    """Поворачивает точку (x, y) вокруг центра (cx, cy) на угол angle_rad."""
    dx = x - cx
    dy = y - cy
    new_x = cx + dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
    new_y = cy + dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
    return new_x, new_y


def generate_panels(configs, grouped_cells):
    ann_id = 1
    # Выбираем 5 случайных панелей для отображения
    display_indices = random.sample(range(1, NUM_PANELS + 1), min(5, NUM_PANELS))
    display_data = []

    # Разделение на тренировочную и валидационную выборки
    all_panel_ids = list(range(1, NUM_PANELS + 1))
    random.shuffle(all_panel_ids)
    train_size = int(0.8 * NUM_PANELS)
    train_ids = set(all_panel_ids[:train_size])
    val_ids = set(all_panel_ids[train_size:])

    for panel_id in range(1, NUM_PANELS + 1):
        panel_type = random.choice(["mono", "poly"])
        config = random.choice(configs)
        rows, cols = config

        candidates = grouped_cells[panel_type]
        random.shuffle(candidates)

        # Размер ячейки фиксирован: 300x300
        cell_w, cell_h = 300, 300
        # Рассчитываем размер панели с учетом отступов
        panel_width = cols * cell_w + (cols + 1) * CELL_MARGIN
        panel_height = rows * cell_h + (rows + 1) * CELL_MARGIN
        panel = Image.new("RGB", (panel_width, panel_height), "black")

        cell_positions = []

        # Размещаем ячейки и сохраняем позиции
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx >= len(candidates):
                    break

                cell = candidates[idx]
                try:
                    with Image.open(cell["path"]).convert("RGB") as img:
                        paste_x = j * cell_w + (j + 1) * CELL_MARGIN
                        paste_y = i * cell_h + (i + 1) * CELL_MARGIN
                        panel.paste(img, (paste_x, paste_y))

                        # Сохраняем позицию только для дефектных ячеек (category_id != 6)
                        if cell["category_id"] != 6:
                            cell_positions.append({
                                "category_id": cell["category_id"],
                                "corners": [
                                    [paste_x, paste_y],              # Левый верхний
                                    [paste_x + cell_w, paste_y],     # Правый верхний
                                    [paste_x + cell_w, paste_y + cell_h],  # Правый нижний
                                    [paste_x, paste_y + cell_h]      # Левый нижний
                                ]
                            })
                except FileNotFoundError:
                    print(f"Warning: File {cell['path']} not found, skipping")
                    continue

        # Сохраняем оригинальную панель на диск
        original_filename = f"original_panel_{panel_id}.jpg"
        panel.save(OUTPUT_DIR / "originals" / original_filename)

        # Случайное масштабирование (0.6–1.0)
        scale_factor = random.uniform(0.6, 1.0)
        new_width = int(panel_width * scale_factor)
        new_height = int(panel_height * scale_factor)

        # Масштабируем панель (первое масштабирование)
        panel = panel.resize((new_width, new_height), Image.BILINEAR)

        # Случайный угол поворота (-30°–30°)
        angle_deg = random.uniform(-30, 30)
        angle_rad = math.radians(angle_deg)

        # Вычисляем размер повернутого изображения
        cos_val = abs(math.cos(angle_rad))
        sin_val = abs(math.sin(angle_rad))
        rot_width = int(new_width * cos_val + new_height * sin_val)
        rot_height = int(new_width * sin_val + new_height * cos_val)

        # Масштабируем панель дополнительно, чтобы она вписывалась в PANEL_SIZE
        fit_scale = min((PANEL_SIZE[0] - 40) / rot_width, (PANEL_SIZE[1] - 40) / rot_height)
        final_width = int(new_width * fit_scale)
        final_height = int(new_height * fit_scale)
        panel = panel.resize((final_width, final_height), Image.BILINEAR)
        total_scale = scale_factor * fit_scale

        # Поворачиваем панель
        panel = panel.rotate(angle_deg, resample=Image.BICUBIC, expand=True)

        # Создаем финальное изображение
        final_panel = Image.new("RGB", PANEL_SIZE, "black")
        offset_x = (PANEL_SIZE[0] - panel.width) // 2
        offset_y = (PANEL_SIZE[1] - panel.height) // 2
        final_panel.paste(panel, (offset_x, offset_y), panel if panel.mode == 'RGBA' else None)

        # Обновляем позиции ячеек
        transformed_cell_positions = []
        center_x = panel_width / 2
        center_y = panel_height / 2

        for pos in cell_positions:
            # Шаг 1: Масштабирование с scale_factor
            scaled_corners_1 = [
                [corner[0] * scale_factor, corner[1] * scale_factor]
                for corner in pos["corners"]
            ]

            # Шаг 2: Масштабирование с fit_scale
            scaled_corners_2 = [
                [corner[0] * fit_scale, corner[1] * fit_scale]
                for corner in scaled_corners_1
            ]

            # Шаг 3: Поворот вокруг центра масштабированной панели с -angle_rad
            rotated_corners = [
                rotate_point(x, y, center_x * scale_factor * fit_scale, center_y * scale_factor * fit_scale, -angle_rad)
                for x, y in scaled_corners_2
            ]

            # Шаг 4: Смещение для центрирования
            transformed_corners = [
                [x + offset_x + (panel.width / 2 - center_x * total_scale),
                 y + offset_y + (panel.height / 2 - center_y * total_scale)]
                for x, y in rotated_corners
            ]

            # Для COCO аннотаций вычисляем прямоугольный bbox
            new_x = min(corner[0] for corner in transformed_corners)
            new_y = min(corner[1] for corner in transformed_corners)
            new_w = max(corner[0] for corner in transformed_corners) - new_x
            new_h = max(corner[1] for corner in transformed_corners) - new_y

            transformed_cell_positions.append({
                "category_id": pos["category_id"],
                "corners": transformed_corners,
                "bbox": [int(new_x), int(new_y), int(new_w), int(new_h)]  # Для COCO
            })

        # Определяем, тренировочная или валидационная выборка
        if panel_id in train_ids:
            split = "train"
        else:
            split = "val"

        image_dir = OUTPUT_DIR / "images" / split
        label_dir = OUTPUT_DIR / "labels" / split

        image_filename = f"panel_{panel_id}.jpg"
        label_filename = f"panel_{panel_id}.txt"

        # Сохраняем изображение
        final_panel.save(image_dir / image_filename)

        # Создаем файл аннотаций YOLOv8
        with open(label_dir / label_filename, "w") as f:
            for pos in transformed_cell_positions:
                class_id = pos["category_id"] - 1
                x, y, w, h = pos["bbox"]
                center_x = (x + w / 2) / PANEL_SIZE[0]
                center_y = (y + h / 2) / PANEL_SIZE[1]
                norm_w = w / PANEL_SIZE[0]
                norm_h = h / PANEL_SIZE[1]
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        # Добавляем данные в COCO (сохраняем для совместимости)
        coco_data["images"].append({
            "id": panel_id,
            "file_name": f"{split}/{image_filename}",
            "width": PANEL_SIZE[0],
            "height": PANEL_SIZE[1]
        })

        for pos in transformed_cell_positions:
            x, y, w, h = pos["bbox"]
            coco_data["annotations"].append({
                "id": ann_id,
                "image_id": panel_id,
                "category_id": pos["category_id"],
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1

        # Сохраняем данные для отображения
        if panel_id in display_indices:
            display_data.append({
                "original_filename": original_filename,
                "final_filename": image_filename,
                "split": split,
                "cell_positions": transformed_cell_positions
            })

        # Освобождаем память
        panel.close()
        final_panel.close()
        del panel
        del final_panel
        gc.collect()

    # Создаем YAML-файл конфигурации для YOLOv8
    dataset_config = {
        "path": str(OUTPUT_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "scratch",
            1: "crack",
            2: "shunt",
            3: "breakdown",
            4: "degradation"
        }
    }

    with open(OUTPUT_DIR / "dataset.yaml", "w") as f:
        yaml.dump(dataset_config, f)

    # Функция отображения
    def display_before_after_panels(display_data, num_pairs=5):
        num_pairs = min(num_pairs, len(display_data))
        if num_pairs == 0:
            print("Нет панелей для отображения")
            return

        fig, axes = plt.subplots(num_pairs, 2, figsize=(12, 6 * num_pairs))
        if num_pairs == 1:
            axes = [axes]

        category_map = {cat["id"]: cat["name"] for cat in DEFECT_CATEGORIES}

        for i, data in enumerate(display_data[:num_pairs]):
            # Изображение "до"
            with Image.open(OUTPUT_DIR / "originals" / data["original_filename"]).convert("RGB") as orig_img:
                axes[i][0].imshow(orig_img)
                axes[i][0].set_title(f"Panel {i + 1} - Before")
                axes[i][0].axis("off")

            # Изображение "после" с bboxes
            with Image.open(OUTPUT_DIR / "images" / data["split"] / data["final_filename"]).convert("RGB") as after_img:
                after_img_with_bboxes = after_img.copy()
                draw = ImageDraw.Draw(after_img_with_bboxes)
                for pos in data["cell_positions"]:
                    category_id = pos["category_id"]
                    color = COLORS.get(category_id, "white")
                    draw.polygon(
                        [(x, y) for x, y in pos["corners"]],
                        outline=color,
                        width=2
                    )
                    x, y = pos["corners"][0]
                    draw.text(
                        (x + 5, y + 5),
                        category_map[category_id],
                        fill=color
                    )
                axes[i][1].imshow(after_img_with_bboxes)
                axes[i][1].set_title(f"Panel {i + 1} - After (Defect BBoxes)")
                axes[i][1].axis("off")

        legend_patches = [
            plt.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor=color, label=name)
            for cat_id, color in COLORS.items()
            for name in [cat["name"] for cat in DEFECT_CATEGORIES if cat["id"] == cat_id]
        ]
        plt.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(COLORS))

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'before_after_panels.png')
        plt.close()

    # Отображаем панели
    display_before_after_panels(display_data)


if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV, header=None, sep=r'\s+')
    configs = get_configurations()
    grouped_cells = load_and_preprocess_cells(df)

    generate_panels(configs, grouped_cells)

    with (OUTPUT_DIR / "annotations.json").open("w") as f:
        json.dump(coco_data, f, indent=2)