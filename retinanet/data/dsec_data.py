from pathlib import Path
from typing import Optional, Callable
from torch_geometric.data import Dataset
import numpy as np
import cv2
import torch
from functools import lru_cache
import sys
import os

# 添加所有需要的路径
sys.path.insert(0, '/media/data/hucao/jinkai/FRN/libs/dsec-det/src')
sys.path.insert(0, os.path.dirname(__file__))

# dsec_det 相关导入
from dsec_det.dataset import DSECDet
from dsec_det.io import yaml_file_to_dict
from dsec_det.directory import BaseDirectory

# 本地模块导入
from dsec_utils import filter_tracks, crop_tracks, rescale_tracks, compute_class_mapping, map_classes, filter_small_bboxes
from augment import init_transforms
from utils import to_data

from visualization.bbox_viz import draw_bbox_on_img
from visualization.event_viz import draw_events_on_image


def tracks_to_array(tracks):
    """修复的标注转换函数：将[x,y,w,h,class]转换为[x1,y1,x2,y2,class]"""
    if len(tracks) == 0 or any(len(tracks[k]) == 0 for k in ['x', 'y', 'w', 'h', 'class_id']):
        return np.zeros((0, 5))
    
    # 原始格式：[x, y, w, h, class_id]
    x = tracks['x']
    y = tracks['y'] 
    w = tracks['w']
    h = tracks['h']
    class_id = tracks['class_id']
    
    # 转换为 [x1, y1, x2, y2, class_id] 格式
    x1 = x
    y1 = y
    x2 = x + w  # x2 = x + width
    y2 = y + h  # y2 = y + height
    
    return np.stack([x1, y1, x2, y2, class_id], axis=1)


def interpolate_tracks(detections_0, detections_1, t):
    assert len(detections_1) == len(detections_0)
    if len(detections_0) == 0:
        return detections_1

    t0 = detections_0['t'][0]
    t1 = detections_1['t'][0]

    assert t0 < t1

    # need to sort detections
    detections_0 = detections_0[detections_0['track_id'].argsort()]
    detections_1 = detections_1[detections_1['track_id'].argsort()]

    r = ( t - t0 ) / ( t1 - t0 )
    detections_out = detections_0.copy()
    for k in 'xywh':
        detections_out[k] = detections_0[k] * (1 - r) + detections_1[k] * r

    return detections_out

class EventDirectory(BaseDirectory):
    @property
    @lru_cache
    def event_file(self):
        return self.root / "left/events_2x.h5"


def normalize_event_data(img_event, method='tanh'):
    """归一化事件数据"""
    if method == 'tanh':
        # 使用tanh归一化到[-1, 1]
        img_event = torch.tanh(img_event / 5.0)
    elif method == 'clip':
        # 简单截断到[-2, 2]
        img_event = torch.clamp(img_event, -2, 2)
    elif method == 'minmax':
        # 最小最大归一化
        min_val = img_event.min()
        max_val = img_event.max()
        if max_val > min_val:
            img_event = 2 * (img_event - min_val) / (max_val - min_val) - 1
    
    return img_event


def validate_annotations(annot, img_width=640, img_height=480):
    """验证和修复标注 - 使用标准图像尺寸"""
    if annot.shape[0] == 0:
        return annot
    
    # 提取坐标
    x1, y1, x2, y2, cls = annot[:, 0], annot[:, 1], annot[:, 2], annot[:, 3], annot[:, 4]
    
    # 修复坐标顺序（确保x2>x1, y2>y1）
    x1_new = torch.minimum(x1, x2)
    x2_new = torch.maximum(x1, x2)
    y1_new = torch.minimum(y1, y2)
    y2_new = torch.maximum(y1, y2)
    
    # 确保最小尺寸
    min_size = 2.0
    width = x2_new - x1_new
    height = y2_new - y1_new
    
    # 如果宽度或高度太小，扩展边界框
    small_width_mask = width < min_size
    small_height_mask = height < min_size
    
    if small_width_mask.any():
        expand_w = (min_size - width[small_width_mask]) / 2
        x1_new[small_width_mask] -= expand_w
        x2_new[small_width_mask] += expand_w
    
    if small_height_mask.any():
        expand_h = (min_size - height[small_height_mask]) / 2
        y1_new[small_height_mask] -= expand_h
        y2_new[small_height_mask] += expand_h
    
    # 边界检查
    x1_new = torch.clamp(x1_new, 0, img_width - min_size)
    y1_new = torch.clamp(y1_new, 0, img_height - min_size)
    x2_new = torch.clamp(x2_new, min_size, img_width)
    y2_new = torch.clamp(y2_new, min_size, img_height)
    
    # 最终检查：确保有效的边界框
    valid_mask = (x2_new > x1_new) & (y2_new > y1_new) & (x2_new - x1_new >= 1) & (y2_new - y1_new >= 1)
    
    if not valid_mask.all():
        print(f"[WARNING] Removing {(~valid_mask).sum()} invalid annotations")
    
    # 重构标注
    fixed_annot = torch.stack([x1_new, y1_new, x2_new, y2_new, cls], dim=1)
    
    # 只保留有效的标注
    fixed_annot = fixed_annot[valid_mask]
    
    return fixed_annot


class DSEC(Dataset):
    MAPPING = dict(pedestrian="pedestrian", rider=None, car="car", bus="car", truck="car", bicycle=None,
                   motorcycle=None, train=None)
    
    def __init__(self,
                 root: Path,
                 split: str,
                 transform: Optional[Callable]=None,
                 debug=False,
                 min_bbox_diag=0,
                 min_bbox_height=0,
                 scale=1,  # 修改为1，不进行额外缩放
                 cropped_height=480,  # 标准高度
                 only_perfect_tracks=False,
                 demo=False,
                 no_eval=False):

        Dataset.__init__(self)

        # 修复：DSEC使用配置文件分割，不依赖目录结构
        self.root = root
        self.split = split
        self.debug = debug

        # 加载分割配置
        split_config = None
        if not demo:
            split_config_path = Path(__file__).parent / "dsec_split.yaml"
            if split_config_path.exists():
                split_config = yaml_file_to_dict(split_config_path)
                print(f"[DEBUG] Loaded split config: {list(split_config.keys())}")
            else:
                print(f"[WARNING] Split config not found at {split_config_path}")
                # 使用默认分割
                split_config = self._create_default_split_config()

        # DSEC数据集总是从根目录加载，通过split_config过滤
        try:
            self.dataset = DSECDet(root=root, split=split, sync="back", debug=debug, split_config=split_config)
        except Exception as e:
            print(f"[ERROR] Failed to load DSECDet: {e}")
            # 尝试不同的路径
            if (root / "train").exists():
                print(f"[INFO] Trying train directory as root...")
                self.dataset = DSECDet(root=root / "train", split=split, sync="back", debug=debug, split_config=split_config)
            else:
                raise e

        print(f"[DEBUG] Found directories: {list(self.dataset.directories.keys())}")

        # 验证和设置事件目录
        valid_sequences = []
        for name, directory in self.dataset.directories.items():
            try:
                # Set event directory
                directory.events = EventDirectory(directory.events.root)

                images_valid = False
                events_valid = False

                # Check if images might be valid
                if hasattr(directory, 'images'):
                    for attr_name in ['timestamps', 'image_files', 'filenames', 'paths']:
                        if hasattr(directory.images, attr_name):
                            attr = getattr(directory.images, attr_name)
                            if attr is not None and hasattr(attr, '__len__') and len(attr) > 0:
                                images_valid = True
                                if debug:
                                    print(f"[DEBUG] Directory {name} has valid images via '{attr_name}' attribute")
                                break

                # Check if event file exists
                try:
                    events_valid = directory.events.event_file.exists()
                except Exception as e:
                    if debug:
                        print(f"[WARNING] Error checking event file for {name}: {e}")

                if images_valid:
                    valid_sequences.append(name)
                    if debug:
                        print(f"[INFO] Directory {name} appears valid. Events: {events_valid}")
                else:
                    if debug:
                        print(f"[WARNING] Directory {name} missing necessary files. Images valid: {images_valid}, Events valid: {events_valid}")

            except Exception as e:
                if debug:
                    print(f"[ERROR] Error processing directory {name}: {e}")

        # 过滤有效序列
        if len(valid_sequences) < len(self.dataset.directories) and len(valid_sequences) > 0:
            print(f"[INFO] Found {len(valid_sequences)} valid sequences out of {len(self.dataset.directories)}")
            self.dataset.directories = {name: self.dataset.directories[name] for name in valid_sequences}
            self.dataset.subsequence_directories = [d for d in self.dataset.subsequence_directories if d.name in valid_sequences]

        print(f"[DEBUG] Initialized DSEC with root: {self.root}")
        print(f"[DEBUG] Available directories: {list(self.dataset.directories.keys())}")

        # 设置参数 - 使用标准尺寸，确保FPN兼容
        self.scale = scale
        self.width = 640   # 固定标准宽度，能被32整除
        self.height = 480  # 固定标准高度，能被32整除
        self.classes = ("car", "pedestrian")
        self.time_window = 1000000
        self.min_bbox_height = min_bbox_height
        self.min_bbox_diag = min_bbox_diag
        self.num_us = -1

        self.class_remapping = compute_class_mapping(self.classes, self.dataset.classes, self.MAPPING)

        if transform is not None and hasattr(transform, "transforms"):
            init_transforms(transform.transforms, self.height, self.width)

        self.transform = transform
        self.no_eval = no_eval

        if self.no_eval:
            only_perfect_tracks = False

        # 过滤轨迹
        self.image_index_pairs, self.track_masks = filter_tracks(
            dataset=self.dataset,
            image_width=self.width,
            image_height=self.height,
            class_remapping=self.class_remapping,
            min_bbox_height=min_bbox_height,
            min_bbox_diag=min_bbox_diag,
            only_perfect_tracks=only_perfect_tracks,
            scale=scale
        )

    def _create_default_split_config(self):
        """创建默认的分割配置"""
        print("[INFO] Creating default split configuration...")
        
        # 基于序列名称的简单分割策略
        all_sequences = []
        try:
            # 假设所有序列都在train目录下
            train_dir = self.root / "train" if (self.root / "train").exists() else self.root
            for seq_dir in train_dir.iterdir():
                if seq_dir.is_dir() and not seq_dir.name.startswith('.'):
                    all_sequences.append(seq_dir.name)
        except Exception as e:
            print(f"[ERROR] Failed to scan sequences: {e}")
            return {}

        print(f"[DEBUG] Found {len(all_sequences)} sequences for splitting")
        
        # 简单的分割：80% train, 20% val
        import random
        random.seed(42)  # 确保可重复
        random.shuffle(all_sequences)
        
        split_point = int(0.8 * len(all_sequences))
        train_sequences = all_sequences[:split_point]
        val_sequences = all_sequences[split_point:]
        
        split_config = {
            'train': train_sequences,
            'val': val_sequences
        }
        
        print(f"[INFO] Split: {len(train_sequences)} train, {len(val_sequences)} val sequences")
        return split_config

    def set_num_us(self, num_us):
        self.num_us = num_us

    def visualize_debug(self, index):
        data = self.__getitem__(index)
        image = data.image[0].permute(1,2,0).numpy()
        p = data.x[:,0].numpy()
        x, y = data.pos.t().numpy()
        b_x, b_y, b_w, b_h, b_c = data.bbox.t().numpy()

        image = draw_events_on_image(image, x, y, p)
        image = draw_bbox_on_img(image, b_x, b_y, b_w, b_h,
                                 b_c, np.ones_like(b_c), conf=0.3, nms=0.65)

        cv2.imshow(f"Debug {index}", image)
        cv2.waitKey(0)

    @property
    def num_classes(self):
        return len(self.classes)

    def __len__(self):
        return sum(len(d) for d in self.image_index_pairs.values())

    def preprocess_detections(self, detections):
        detections = rescale_tracks(detections, self.scale)
        detections = crop_tracks(detections, self.width, self.height)
        detections['class_id'], _ = map_classes(detections['class_id'], self.class_remapping)
        return detections

    def preprocess_events(self, events):
        """预处理事件数据 - 使用标准尺寸确保FPN兼容"""
        # 使用标准尺寸，确保能被32整除，FPN兼容
        H, W = 480, 640  
        C = 5  # 通道数

        # 筛选范围内事件
        mask = (events['y'] < H) & (events['x'] < W)
        events = {k: v[mask] for k, v in events.items()}

        if len(events['t']) == 0:
            voxel = np.zeros((C, H, W), dtype=np.float32)
            return {'img': torch.tensor(voxel)}

        # 归一化时间为 [0, 1]
        t_min, t_max = events['t'][0], events['t'][-1]
        t_norm = (events['t'] - t_min) / (t_max - t_min + 1e-6)

        x = events['x'].astype(np.int64)
        y = events['y'].astype(np.int64)
        p = events['p'].astype(np.int64)
        p = (p > 0).astype(np.float32) * 2 - 1  # 255 -> +1, 0 -> -1
        t_bin = (t_norm * (C - 1)).astype(np.int64)  # 修复：确保不会超出索引
        t_bin = np.clip(t_bin, 0, C - 1)

        if self.debug and len(x) > 0:
            print(f"[DEBUG] Event count: {len(x)}")
            print(f"[DEBUG] x range: {x.min()} - {x.max()}, y range: {y.min()} - {y.max()}")
            print(f"[DEBUG] Target size: {H}x{W}")

        voxel = np.zeros((C, H, W), dtype=np.float32)

        # 填充体素网格
        for c, xi, yi, pi in zip(t_bin, x, y, p):
            if 0 <= xi < W and 0 <= yi < H:  # 边界检查
                voxel[c, yi, xi] += pi

        if self.debug:
            print(f"[DEBUG] Voxel shape: {voxel.shape}, sum: {voxel.sum()}, range: [{voxel.min():.3f}, {voxel.max():.3f}]")

        return {'img': torch.tensor(voxel)}

    def preprocess_image(self, image):
        """预处理RGB图像 - 使用标准尺寸确保FPN兼容"""
        # 使用标准尺寸，确保能被32整除，FPN兼容
        target_height, target_width = 480, 640
        
        if image is None:
            return torch.zeros(3, target_height, target_width)
        
        # 直接调整到目标尺寸
        image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        
        # 转换为tensor [C, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # 归一化到 [0, 1] 范围
        if image.max() > 1:
            image = image / 255.0
        
        # 添加batch维度然后移除（为了兼容）
        image = image.unsqueeze(0)
        return image

    def __getitem__(self, idx):
        if self.debug:
            print(f"[DEBUG] __getitem__ called with index {idx}")
        
        try:
            dataset, image_index_pairs, track_masks, idx = self.rel_index(idx)
            image_index_0, image_index_1 = image_index_pairs[idx]
            image_ts_0, image_ts_1 = dataset.images.timestamps[[image_index_0, image_index_1]]

            # 获取检测框
            detections_0 = self.dataset.get_tracks(image_index_0, mask=track_masks, directory_name=dataset.root.name)
            detections_1 = self.dataset.get_tracks(image_index_1, mask=track_masks, directory_name=dataset.root.name)

            if detections_0 is None or detections_1 is None:
                if self.debug:
                    print(f"[WARNING] Track files not found for {dataset.root.name}")
                # 返回空检测
                detections_0 = {'x': np.array([]), 'y': np.array([]), 'w': np.array([]), 'h': np.array([]), 'class_id': np.array([])}
                detections_1 = detections_0.copy()
            else:
                detections_0 = self.preprocess_detections(detections_0)
                detections_1 = self.preprocess_detections(detections_1)

            # 获取 RGB 图像
            image_0 = self.dataset.get_image(image_index_0, directory_name=dataset.root.name)
            if image_0 is None:
                if self.debug:
                    print(f"[WARNING] Image file not found for {dataset.root.name}")
                image_0 = np.zeros((480, 640, 3), dtype=np.uint8)  # 使用标准尺寸
            
            image_0 = self.preprocess_image(image_0)

            # 获取事件数据
            events = self.dataset.get_events(image_index_0, directory_name=dataset.root.name)
            if events is None:
                if self.debug:
                    print(f"[WARNING] Events not found for {dataset.root.name}")
                events = {'x': np.array([]), 'y': np.array([]), 't': np.array([]), 'p': np.array([])}

            if self.num_us >= 0:
                image_ts_1 = image_ts_0 + self.num_us
                events = {k: v[events['t'] < image_ts_1] for k, v in events.items()}
                if not self.no_eval and len(detections_0) > 0 and len(detections_1) > 0:
                    detections_1 = interpolate_tracks(detections_0, detections_1, image_ts_1)

            # 处理事件数据
            events = self.preprocess_events(events)
            img_event = events['img']   # shape [5, 480, 640]
            
            # 归一化事件数据（如果范围过大）
            if img_event.abs().max() > 5:
                img_event = normalize_event_data(img_event, method='tanh')
            
            # 确保事件图像形状正确
            if img_event.shape != (5, self.height, self.width):
                if self.debug:
                    print(f"[WARNING] Resizing event image from {img_event.shape} to (5, {self.height}, {self.width})")
                img_event = torch.nn.functional.interpolate(
                    img_event.unsqueeze(0), 
                    size=(self.height, self.width), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)

            # 处理检测框并修复格式
            if len(detections_1) > 0:
                annot = tracks_to_array(detections_1)
                annot = torch.tensor(annot, dtype=torch.float32)
                # 验证和修复标注 - 使用标准图像尺寸
                annot = validate_annotations(annot, img_width=self.width, img_height=self.height)
            else:
                annot = torch.zeros((0, 5), dtype=torch.float32)

            if self.debug:
                print(f"[DEBUG] Final shapes - Event: {img_event.shape}, RGB: {image_0.shape}, Annot: {annot.shape}")
                print(f"[DEBUG] Final ranges - Event: [{img_event.min():.3f}, {img_event.max():.3f}], RGB: [{image_0.min():.3f}, {image_0.max():.3f}]")

            return {
                'img': img_event,                    # [5, 480, 640] - 标准尺寸
                'img_rgb': image_0,                  # [1, 3, 480, 640] - 标准尺寸
                'annot': annot,                      # [N, 5] - 已修复格式
                'sequence': str(dataset.root.name),
                'timestamp': image_ts_1,
                'image_index': image_index_1
            }

        except Exception as e:
            if self.debug:
                print(f"[ERROR] Error in __getitem__({idx}): {e}")
                import traceback
                traceback.print_exc()
            
            # 返回安全的默认值 - 使用标准尺寸
            return {
                'img': torch.zeros(5, 480, 640, dtype=torch.float32),
                'img_rgb': torch.zeros(1, 3, 480, 640, dtype=torch.float32),
                'annot': torch.zeros((0, 5), dtype=torch.float32),
                'sequence': '',
                'timestamp': 0,
                'image_index': idx
            }

    def rel_index(self, idx):
        for folder in self.dataset.subsequence_directories:
            name = folder.name
            image_index_pairs = self.image_index_pairs[name]
            directory = self.dataset.directories[name]
            track_mask = self.track_masks[name]
            if idx < len(image_index_pairs):
                return directory, image_index_pairs, track_mask, idx
            idx -= len(image_index_pairs)
        raise IndexError(f"Index {idx} out of range")
