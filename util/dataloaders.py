import os
import random
from PIL import Image
import torch
import torch.utils.data as data
from util import transforms as tr
from torchvision.transforms import functional as F
import numpy as np

# !!!Vaihingen (使用6类配置)
num_classes = 6
ST_COLORMAP = np.array([
    (255, 255, 255),  # 0: Impervious surfaces (不透水表面)
    (0, 0, 255),      # 1: Building (建筑物)
    (0, 255, 255),    # 2: Low vegetation (低矮植被)
    (0, 255, 0),      # 3: Tree (树木)
    (255, 255, 0),    # 4: Car (汽车)
    (255, 0, 0)       # 5: Clutter/background (杂物)
])
ST_CLASSES = ['Impervious surfaces', 'Building', 'Low vegetation', 'Tree', 'Car', 'Clutter']

# !!!SECOND-CD (7类配置，已注释)
# num_classes = 7
# ST_COLORMAP = np.array([(255, 255, 255), (0, 0, 255), (128, 128, 128),(0, 128, 0), (0, 255, 0), (128, 0, 0), (255, 0, 0)])
# ST_CLASSES = ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']

# !!!Landsat-CD (5类配置，已注释)
# num_classes = 5
# ST_COLORMAP = np.array([(255, 255, 255), (0, 155, 0), (255, 165, 0), (230, 30, 100), (0, 170, 240)])
# ST_CLASSES = ['No change', 'Farmland', 'Desert', 'Building', 'Water']


color_map = ST_COLORMAP


def rgb2label(rgb_label):
    """
    将RGB标签转换为类别索引
    确保所有值都在 [0, num_classes-1] 范围内
    """
    rgb_label = np.array(rgb_label)
    gray_label = np.zeros(
        shape=(rgb_label.shape[0], rgb_label.shape[1]), dtype=np.uint8)
    
    # 对每个类别进行匹配
    for i in range(color_map.shape[0]):
        # 找到所有匹配当前颜色的像素
        mask = np.all(rgb_label == color_map[i], axis=-1)
        gray_label[mask] = i
    
    # 安全检查：确保所有值都在有效范围内
    # 将超出范围的值设为0（背景类）
    gray_label = np.clip(gray_label, 0, num_classes - 1)
    
    # 调试信息：检查标签范围
    unique_labels = np.unique(gray_label)
    if np.any(unique_labels >= num_classes):
        print(f"Warning: Found labels >= num_classes: {unique_labels[unique_labels >= num_classes]}")
        gray_label = np.where(gray_label >= num_classes, 0, gray_label)
    
    return gray_label.astype(np.int64)  # 确保数据类型正确


def gen_changelabel(label1, label2):
    label1=np.array(label1)
    label2=np.array(label2)
    binary_label =np.zeros_like(label1)
    binary_label[label1 != label2] = -1
    binary_label[binary_label !=-1] = 0
    binary_label[binary_label == -1] = 1
    return binary_label


def get_loaders(opt):

    train_dataset = CDDloader(opt, 'train', aug=True)
    val_dataset = CDDloader(opt, 'val', aug=False)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True, drop_last=True,
                                               num_workers=opt.num_workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=False, drop_last=True,
                                             num_workers=opt.num_workers)
    return train_loader, val_loader


def get_eval_loaders(opt):    
    dataset_name = "val"
    print("using dataset: {} set".format(dataset_name))
    eval_dataset = CDDloader(opt, dataset_name, aug=False)
    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=False, drop_last=True,
                                              num_workers=opt.num_workers)
    return eval_loader

def get_infer_loaders(opt):
    infer_datast = CDDloadImageOnly(opt, '', aug=False)
    infer_loader = torch.utils.data.DataLoader(infer_datast,
                                               batch_size=opt.batch_size,
                                               shuffle=False,drop_last=True,
                                               num_workers=opt.num_workers)
    return infer_loader


class CDDloader(data.Dataset):

    def __init__(self, opt, phase, aug=False):
        self.data_dir = str(opt.dataset_dir)
        self.phase = str(phase)
        self.aug = aug
        names = [i for i in os.listdir(
            os.path.join(self.data_dir, phase, 'A'))]
        self.names = []
        for name in names:
            if is_img(name):
                self.names.append(name)

        random.shuffle(self.names)

    def __getitem__(self, index):

        name = str(self.names[index])
        img_A = Image.open(os.path.join(self.data_dir, self.phase, 'A', name))
        img_B = Image.open(os.path.join(self.data_dir, self.phase, 'B', name))
        label_name = name.replace("tif", "png") if name.endswith(
            "tif") else name   # for shengteng

        # 多模态语义分割：使用单标签（沿用原目录中的 labelA）
        label_A = Image.open(os.path.join(
            self.data_dir, self.phase, 'labelA', label_name))
        label_A = rgb2label(label_A)

        # 转换为RGB模式（确保都是3通道）
        if img_A.mode != 'RGB':
            img_A = img_A.convert('RGB')
        if img_B.mode != 'RGB':
            img_B = img_B.convert('RGB')
        
        # 获取目标尺寸（以img_A为准）
        target_size = img_A.size  # (width, height)
        
        # 如果img_B尺寸不同，调整到与img_A相同
        if img_B.size != target_size:
            img_B = img_B.resize(target_size, Image.BILINEAR)
            print(f"Warning: Resized image B from {img_B.size} to {target_size} for {name}")

        img_A = np.array(img_A)
        img_B = np.array(img_B)

        # 返回两模态图像与单标签
        return F.to_tensor(img_A), F.to_tensor(img_B), torch.from_numpy(label_A), label_name

    def __len__(self):
        return len(self.names)

def is_img(name):
    img_format = ["jpg", "png", "jpeg", "bmp", "tif", "tiff", "TIF", "TIFF"]
    if "." not in name:
        return False
    if name.split(".")[-1] in img_format:
        return True
    else:
        return False

class CDDloadImageOnly(data.Dataset):

    def __init__(self, opt, phase, aug=False):
        self.data_dir = str(opt.dataset_dir)
        self.phase = str(phase)
        self.aug = aug
        names = [i for i in os.listdir(os.path.join(self.data_dir, phase, 'A'))]
        self.names = []
        for name in names:
            if is_img(name):
                self.names.append(name)

    def __getitem__(self, index):
        name = str(self.names[index])
        img1 = Image.open(os.path.join(self.data_dir, self.phase, 'A', name))
        img2 = Image.open(os.path.join(self.data_dir, self.phase, 'B', name))

        return  F.to_tensor(img1), F.to_tensor(img2), name

    def __len__(self):
        return len(self.names)
