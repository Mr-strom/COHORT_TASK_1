import os
import glob
import shutil
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(base_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

def count_labels(image_dir_path):
    label_dir_path = image_dir_path.replace('images', 'labels')
    counts = Counter()
    label_files = glob.glob(os.path.join(label_dir_path, '**', '*.txt'), recursive=True)
    for lf in label_files:
        with open(lf, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1
    return counts

def generate_class_balance_chart(output_dir=output_dir, yaml_name='construction-ppe.yaml'):
    chart_path = os.path.join(output_dir, 'class_balance_chart.png')
    try:
        dataset_info = check_det_dataset(yaml_name)
        class_names = dataset_info['names']
        train_counts = count_labels(dataset_info['train'])
        val_counts = count_labels(dataset_info['val'])

        data = []
        for idx, name in class_names.items():
            total = train_counts.get(idx, 0) + val_counts.get(idx, 0)
            data.append({'Class': name, 'Total Instances': total})

        df = pd.DataFrame(data).sort_values(by='Total Instances', ascending=True)

        plt.figure(figsize=(10, 6))
        bars = plt.barh(df['Class'], df['Total Instances'], color='#2b5c8f')
        plt.xlabel('Total Instances in Dataset', fontsize=12)
        plt.title('Construction-PPE Class Imbalance', fontsize=14, weight='bold')
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        for bar in bars:
            plt.text(bar.get_width() + 15, bar.get_y() + bar.get_height()/2, 
                     f'{int(bar.get_width())}', va='center', fontsize=10)
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300)
        plt.close()
        print(f"Chart saved to: {chart_path}")
        return chart_path, df
    except Exception as e:
        print(f"Error generating class balance chart: {e}")
        if os.path.exists(chart_path):
            return chart_path, None
        return None, None

def get_available_metrics(output_dir=output_dir):
    metrics = {}
    chart_path = os.path.join(output_dir, 'class_balance_chart.png')
    if os.path.exists(chart_path):
        metrics['class_balance'] = chart_path

    result_folders = sorted(glob.glob(os.path.join(output_dir, 'train_results*')), reverse=True)
    if result_folders:
        latest = result_folders[0]
        for name in ['results.png', 'confusion_matrix.png', 'confusion_matrix_normalized.png', 'BoxF1_curve.png', 'BoxPR_curve.png']:
            p = os.path.join(latest, name)
            if os.path.exists(p):
                metrics[name.replace('.png', '')] = p
    return metrics

def train_model(data_yaml='construction-ppe.yaml', epochs=5, imgsz=320, batch=8, workers=2, device='cpu', output_dir=output_dir):
    weights_path = os.path.join(base_dir, 'code', 'yolov8s.pt')
    if not os.path.exists(weights_path):
        weights_path = 'yolov8s.pt'
    model = YOLO(weights_path)
    results = model.train(
        data=data_yaml,
        epochs=epochs, 
        imgsz=imgsz, 
        batch=batch,
        workers=workers, 
        device=device, 
        project=output_dir, 
        name='train_results' 
    )

    source_weights = os.path.join(output_dir, 'train_results', 'weights', 'best.pt')
    dest_weights = os.path.join(output_dir, 'best.pt')
    if os.path.exists(source_weights):
        shutil.copy(source_weights, dest_weights)
        print("copied to Output folder.")
    return results

if __name__ == '__main__':
    chart_path, df = generate_class_balance_chart()
    print("Class balance chart ready at:", chart_path)
   