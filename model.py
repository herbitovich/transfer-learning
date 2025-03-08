import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.detection import FasterRCNN
import matplotlib.patches as patches
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import resnet18
from PIL import Image
import os
import threading
import pandas as pd
import more_itertools as mit
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

# Аугментация датасета
def augment(image_dir, augment_cnt, output_dir, num_threads):
    augmentation = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.Blur(blur_limit=3, p=0.5),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    files = os.listdir(image_dir)
    def augment_files(image_dir, num, filenames, augment_cnt, output_dir):
        for ind in range(len(filenames)):
            filename = filenames[ind]
            if not filename.endswith(('.png', '.jpg', '.jpeg', 'JPG')): continue
            
            img_path = os.path.join(image_dir, filename)
            image = np.array(Image.open(img_path).convert("RGB"))
            annotation_file = os.path.join(image_dir, f"{os.path.splitext(filename)[0]}.csv")
            df = pd.read_csv(annotation_file)
            bboxes = []
            class_labels = []
            for _, r in df.iterrows():
                xmin, ymin, xmax, ymax = r['xmin'], r['ymin'], r['xmax'], r['ymax']
                cls = r['class']
                bboxes.append([xmin, ymin, xmax, ymax])
                class_labels.append(cls)
                
            for _ in range(augment_cnt):
                try:
                    augmented = augmentation(image=image, bboxes=bboxes, class_labels=class_labels)
                except: continue
                augmented_image = augmented['image']
                augmented_bboxes = augmented['bboxes']
                augmented_filename = f"{os.path.splitext(filename)[0]}_aug_{_}.jpg"
                augmented_path = os.path.join(output_dir, augmented_filename)
                Image.fromarray(augmented_image).save(augmented_path)
                
                annotation_filename = f"{os.path.splitext(filename)[0]}_aug_{_}.csv"
                df = pd.DataFrame.from_dict({'filename':[annotation_filename]*len(augmented_bboxes), 
                                             'width':[augmented_image.shape[1]]*len(augmented_bboxes), 
                                             'height':[augmented_image.shape[0]]*len(augmented_bboxes),
                                             'class':class_labels,
                                            })
                df['xmin'], df['ymin'], df['xmax'], df['ymax'] = zip(*augmented_bboxes)
                
                annotation_path = os.path.join(output_dir, annotation_filename)
                df.to_csv(annotation_path)
    threads = []
    files = [list(chunk) for chunk in mit.divide(num_threads, files)]
    for t in range(num_threads):
        thread = threading.Thread(target=augment_files, args=(image_dir, t, files[t], augment_cnt, output_dir))
        thread.start()
        threads.append(thread)
    for ind in range(len(threads)):
        threads[ind].join()
        print(f"Thread #{ind} completed.")
            
    

dataset_dir = os.path.join(os.getcwd(), 'dataset')
augment_cnt = 1
num_threads = 50
augment(dataset_dir, augment_cnt, dataset_dir, num_threads)
print("Augmentation complete.")

# Парсим датасет
def parse_dataset(image_dir):
    annotations = []
    class_mapping = {}
    class_id = 1
    
    for filename in os.listdir(image_dir):
        if filename.endswith('.csv'):
            csv_path = os.path.join(image_dir, filename)
            df = pd.read_csv(csv_path)
            
            image_name = filename.replace('.csv', '.jpg')
            boxes = []
            labels = []
            
            for _, row in df.iterrows():
                if row['class'] not in class_mapping:
                    class_mapping[row['class']] = class_id
                    class_id += 1
                boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                labels.append(class_mapping[row['class']])
            
            annotations.append({
                'image_id': len(annotations),
                'file_name': image_name,
                'boxes': boxes,
                'labels': labels
            })
    return annotations, class_mapping
    
class CustomDataset(Dataset):
    def __init__(self, image_dir, annotations, transform=None):
        self.image_dir = image_dir
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_info = self.annotations[idx]
        image_id = image_info['image_id']
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")

        boxes = torch.tensor(image_info['boxes'], dtype=torch.float32)
        labels = torch.tensor(image_info['labels'], dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels
        }
        image = T.ToTensor()(image)

        return image_info['file_name'], image, target

image_dir = os.path.join(os.getcwd(), 'dataset')
annotations, class_mapping = parse_dataset(image_dir)
class_id_mapping = {v : k for k,v in class_mapping.items()}
dataset = CustomDataset(image_dir, annotations)

# Разбиваем на выборки
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)
print("Dataset is set up.")

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(
    train_dataset, batch_size=1, shuffle=True, 
    collate_fn=collate_fn, num_workers=8, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=1, 
    collate_fn=collate_fn, num_workers=8
)
test_loader = DataLoader(
    test_dataset, batch_size=1, 
    collate_fn=collate_fn, num_workers=8
)

print("Starting the training process.")

# Инициализируем и тренируем модель
def train(train_loader, val_loader, num_epochs, patience, class_mapping):
    backbone = resnet18(pretrained=True)
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    backbone.out_channels = 512
    
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)*5
    )
    
    model = FasterRCNN(
        backbone,
        num_classes=len(class_mapping) + 1,
        rpn_anchor_generator=anchor_generator,
        box_detections_per_img=50
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    num_bad_epochs = 0
    best_loss = 10e5
    for epoch in range(num_epochs):
        model.train()
        for _, images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            print(losses.item())
            losses.backward()
            optimizer.step()
            
        print(f"Epoch #{epoch}.")
        running_loss = 0
        count = 0
        with torch.no_grad():
            for _, images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                running_loss += sum(loss for loss in loss_dict.values()).item()
                count += 1
        loss_item = running_loss / count
        if loss_item > best_loss:
            num_bad_epochs += 1
        else:
            num_bad_epochs = 0
        if num_bad_epochs >= patience:
            print("Early stopping.")
            break
        print(f"Validation loss: {loss_item}")
    torch.save(model.state_dict(), 'trained.pt')
    return model

model = train(train_loader, val_loader, 15, 2, class_mapping)
model.eval()

# Удаляем явные ошибки детекции - вложенные баунд-боксы, баунд-боксы с почти идентичными координатами (группируем баунд-боксы, указывающие
# на один объект, оставляем только один - с наибольшим скором уверенности)
def process_detections(detections):
    boxes = detections['boxes']
    labels = detections['labels']
    scores = detections['scores']
    
    if len(boxes) == 0:
        return detections
    
    image_width = max(box[2] for box in boxes)
    image_height = max(box[3] for box in boxes)
    epsilon = 0.01
    dx_threshold = epsilon * image_width
    dy_threshold = epsilon * image_height
    image_area = image_width * image_height
    area_threshold = epsilon * image_area

    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    n = len(boxes)
    parent = list(range(n))
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            parent[root_v] = root_u
    
    for i in range(n):
        for j in range(i + 1, n):
            box_i = boxes[i]
            box_j = boxes[j]
            
            # Условие 1: баунд-бокс полностью находится в другом
            i_inside_j = (box_i[0] >= box_j[0] and box_i[1] >= box_j[1] and 
                          box_i[2] <= box_j[2] and box_i[3] <= box_j[3])
            j_inside_i = (box_j[0] >= box_i[0] and box_j[1] >= box_i[1] and 
                          box_j[2] <= box_i[2] and box_j[3] <= box_i[3])
            if i_inside_j or j_inside_i:
                union(i, j)
                continue
            
            # Условие 2: координаты баунд-боксов
            dx_min = abs(box_i[0] - box_j[0])
            dy_min = abs(box_i[1] - box_j[1])
            dx_max = abs(box_i[2] - box_j[2])
            dy_max = abs(box_i[3] - box_j[3])
            
            coord_ok = (dx_min <= dx_threshold and dy_min <= dy_threshold and
                        dx_max <= dx_threshold and dy_max <= dy_threshold)
            area_diff = abs(areas[i] - areas[j])
            area_ok = area_diff <= area_threshold
            
            if coord_ok and area_ok:
                union(i, j)
    
    # Группируем
    groups = {}
    for idx in range(n):
        root = find(idx)
        if root not in groups:
            groups[root] = []
        groups[root].append(idx)
    
    # Выбираем баунд-бокс с наибольшим скором уверенности
    selected_indices = []
    for group_indices in groups.values():
        group_scores = scores[group_indices]
        max_idx_in_group = torch.argmax(group_scores).item()
        original_idx = group_indices[max_idx_in_group]
        selected_indices.append(original_idx)
    
    filtered_boxes = [boxes[i] for i in selected_indices]
    filtered_labels = labels[selected_indices]
    filtered_scores = scores[selected_indices]
    
    return {
        'boxes': filtered_boxes,
        'labels': filtered_labels,
        'scores': filtered_scores
    }

# Визуализируем баунд-боксы
def visualize(image_path, output, save_image_path=None,show_plot = False):
    global class_id_mapping, image_dir
    
    img = Image.open(os.path.join(image_dir, image_path))

    fig, ax = plt.subplots(1, figsize=(12, 8))
    fig.patch.set_facecolor('#101010')
    ax.imshow(img)

    bboxes = output['boxes']
    classes = output['labels'].tolist()

    color_map = plt.get_cmap('tab20')
    num_classes = len(set(classes))
    colors = [color_map(i % 20) for i in range(num_classes)]

    bbox_patches = []
    labels = []

    for cls, bbox in zip(classes, bboxes):
        x_min, y_min, x_max, y_max = bbox

        color = colors[cls % num_classes]

        bbox_patch = patches.Rectangle(
            (x_min, y_min), (x_max - x_min), (y_max - y_min), linewidth=2, edgecolor=color, facecolor='none')
        
        ax.add_patch(bbox_patch)

        if class_id_mapping[cls] not in labels:
            labels.append(class_id_mapping[cls])
            bbox_patches.append(patches.Patch(color=color, label=class_id_mapping[cls]))
            
    ax.legend(handles=bbox_patches, loc='upper left', fontsize=12,frameon=True, facecolor="#101010", labelcolor="white")

    if save_image_path:
        plt.savefig(save_image_path, bbox_inches='tight') 

    if show_plot:
        plt.show()

    plt.close(fig)


# Прогоняемся по тестовой выборке
device = 'cuda' if torch.cuda.is_available() else 'cpu'
y_pred, y_actual = [], []
with torch.no_grad():
    for paths, images, targets in test_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        output = model(images)
        output[0] = process_detections(output[0])
        visualize(paths[0], output[0], save_image_path=os.path.join(os.getcwd(), 'output', paths[0]), show_plot=True)

        y_actual.append(targets[0]['labels'].tolist())
        y_pred.append(output[0]['labels'].tolist())


import numpy as np


# Считаем confusion matrix и метрики по flattened реальным и запредикченным данным 
def compute_confusion_matrix(flat_actual, flat_predicted):
    flat_actual = np.asarray(flat_actual)
    flat_predicted = np.asarray(flat_predicted)

    unique_labels = np.unique(np.concatenate([flat_actual, flat_predicted]))
    num_classes = len(unique_labels)

    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    min_length = min(len(flat_actual), len(flat_predicted))
    for i in range(min_length):
        actual_label = flat_actual[i]
        predicted_label = flat_predicted[i]
        actual_idx = label_to_index[actual_label]
        predicted_idx = label_to_index[predicted_label]
        conf_matrix[actual_idx, predicted_idx] += 1
    
    if len(flat_predicted) > len(flat_actual):
        for i in range(min_length, len(flat_predicted)):
            predicted_label = flat_predicted[i]
            predicted_idx = label_to_index[predicted_label]
            conf_matrix[:, predicted_idx] += 1
    
    if len(flat_actual) > len(flat_predicted):
        for i in range(min_length, len(flat_actual)):
            actual_label = flat_actual[i]
            actual_idx = label_to_index[actual_label]
            conf_matrix[actual_idx, :] += 1
    
    return conf_matrix

flat_actual = [item for sublist in y_actual for item in sublist]
flat_predicted = [item for sublist in y_pred for item in sublist]

conf_matrix = compute_confusion_matrix(flat_actual, flat_predicted)
print("Confusion Matrix:")
print(conf_matrix.tolist())

def compute_metrics(conf_matrix):
    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP
    TN = np.sum(conf_matrix) - (TP + FP + FN)
    
    accuracy = np.sum(TP) / np.sum(conf_matrix)
    precision = np.divide(TP, TP + FP, out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0)
    recall = np.divide(TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)
    
    return accuracy, precision, recall
    
accuracy, precision, recall = compute_metrics(conf_matrix)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)