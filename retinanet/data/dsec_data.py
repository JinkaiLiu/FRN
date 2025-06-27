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
    return np.stack([tracks['x'], tracks['y'], tracks['w'], tracks['h'], tracks['class_id']], axis=1)



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
                 scale=2,
                 cropped_height=430,
                 only_perfect_tracks=False,
                 demo=False,
                 no_eval=False):

        Dataset.__init__(self)

        split_config = None
        if not demo:
            split_config = yaml_file_to_dict(Path(__file__).parent / "dsec_split.yaml")
            assert split in split_config.keys(), f"'{split}' not in {list(split_config.keys())}"

        self.dataset = DSECDet(root=root, split=split, sync="back", debug=debug, split_config=split_config)

        # Output found directories
        print(f"[DEBUG] Found directories: {list(self.dataset.directories.keys())}")

        valid_sequences = []
        for name, directory in self.dataset.directories.items():
            try:
                # Set event directory
                directory.events = EventDirectory(directory.events.root)

                images_valid = False
                events_valid = False

                # Check if images might be valid
                if hasattr(directory, 'images'):
                    # Try checking some possible image attributes
                    for attr_name in ['timestamps', 'image_files', 'filenames', 'paths']:
                        if hasattr(directory.images, attr_name):
                            attr = getattr(directory.images, attr_name)
                            if attr is not None and hasattr(attr, '__len__') and len(attr) > 0:
                                images_valid = True
                                print(f"[DEBUG] Directory {name} has valid images via '{attr_name}' attribute")
                                break

                # Check if event file exists
                try:
                    events_valid = directory.events.event_file.exists()
                except Exception as e:
                    print(f"[WARNING] Error checking event file for {name}: {e}")

                # Keep this sequence if images and events are valid, or at least images are valid
                if images_valid:
                    valid_sequences.append(name)
                    print(f"[INFO] Directory {name} appears valid. Events: {events_valid}")
                else:
                    print(f"[WARNING] Directory {name} missing necessary files. Images valid: {images_valid}, Events valid: {events_valid}")

            except Exception as e:
                print(f"[ERROR] Error processing directory {name}: {e}")

        # If there are invalid sequences, optionally filter them out
        if len(valid_sequences) < len(self.dataset.directories) and len(valid_sequences) > 0:
            print(f"[INFO] Found {len(valid_sequences)} valid sequences out of {len(self.dataset.directories)}")
            self.dataset.directories = {name: self.dataset.directories[name] for name in valid_sequences}
            self.dataset.subsequence_directories = [d for d in self.dataset.subsequence_directories if d.name in valid_sequences]

        self.debug = debug

        #self.dataset = DSECDet(root=root, split=split, sync="back", debug=debug)
        self.split = split
        self.root = root

        # Automatically detect and use 'transformed_images' if it exists
       # transformed_root = root / split / "transformed_images"
      #  if transformed_root.exists():
     #       print(f"[DEBUG] Using transformed_images folder: {transformed_root}")
    #        self.root = transformed_root
   #     else:
  #          print(f"[DEBUG] Using standard folder: {root / split}")
 #           self.root = root / split

        # Initialize DSECDet with the adjusted root
#        self.dataset = DSECDet(root=self.root, split=split, sync="back", debug=debug)

        print(f"[DEBUG] Initialized DSEC with root: {self.root}")
        print(f"[DEBUG] Available directories: {list(self.dataset.directories.keys())}")

        self.scale = scale
        self.width = self.dataset.width // scale
        self.height = cropped_height // scale
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

        self.image_index_pairs, self.track_masks = filter_tracks(
            dataset=self.dataset,
            image_width=self.width,
            image_height=self.height,
            class_remapping=self.class_remapping,
            min_bbox_height=min_bbox_diag,
            min_bbox_diag=min_bbox_diag,
            only_perfect_tracks=only_perfect_tracks,
            scale=scale
        )

        if debug:
            print(f"\n[DEBUG] Initialized DSEC with root: {root}")
            print(f"[DEBUG] Split: {split}")
            print(f"[DEBUG] Checking directory structure in root: {root / split}")

            if not (root / split).exists():
                print(f"[ERROR] Split directory does not exist: {root / split}")
                raise FileNotFoundError(f"Split directory not found: {root / split}")

#            for sequence in (root / split / 'transformed_images').iterdir():
            for sequence in (root / split).iterdir():
                if sequence.is_dir():
                    print(f"\n[DEBUG] Sequence: {sequence.name}")

                    # check Images folder
                    images_path = sequence / "images" / "left" / "rectified"
                    if images_path.exists():
                        images_files = list(images_path.rglob("*.png"))
                        print(f"  -> Images ({len(images_files)} files):")
                        for img in images_files[:5]:
                            print(f"    - {img.name}")
                        if len(images_files) > 5:
                            print(f"    ... and {len(images_files) - 5} more")
                    else:
                        print(f"  -> Images: Not found")

                    # check Object Detections folder
                    object_detections_path = sequence / "object_detections" / "left"
                    if object_detections_path.exists():
                        detections_files = list(object_detections_path.rglob("*.npy"))
                        print(f"  -> Object Detections ({len(detections_files)} files):")
                        for det in detections_files:
                            print(f"    - {det.name}")
                    else:
                        print(f"  -> Object Detections: Not found")

                    # check Events folder
                    events_path = sequence / "events" / "left"
                    if events_path.exists():
                        events_files = list(events_path.rglob("*.h5"))
                        print(f"  -> Events ({len(events_files)} files):")
                        for evt in events_files:
                            print(f"    - {evt.name}")
                    else:
                        print(f"  -> Events: Not found")

                    # check Timestamps.txt
                    timestamps_path = sequence / "images" / "timestamps.txt"
                    if timestamps_path.exists():
                        print(f"  -> Timestamps: Exists")
                    else:
                        print(f"  -> Timestamps: Not found")
        split_config = None
        if not demo:
            split_config = yaml_file_to_dict(Path(__file__).parent / "dsec_split.yaml")
            assert split in split_config.keys(), f"'{split}' not in {list(split_config.keys())}"

        self.dataset = DSECDet(root=root, split=split, sync="back", debug=debug, split_config=split_config)

        print(f"[DEBUG] Dataset root: {root}")
        print(f"[DEBUG] Split: {split}")
        print(f"[DEBUG] Available directories: {list(self.dataset.directories.keys())}")

        for directory in self.dataset.directories.values():
            directory.events = EventDirectory(directory.events.root)

        self.scale = scale
        self.width = self.dataset.width // scale
        self.height = cropped_height // scale
        self.classes = ("car", "pedestrian")
        self.time_window = 1000000
        self.min_bbox_height = min_bbox_height
        self.min_bbox_diag = min_bbox_diag
        self.debug = debug
        self.num_us = -1

        self.class_remapping = compute_class_mapping(self.classes, self.dataset.classes, self.MAPPING)

        if transform is not None and hasattr(transform, "transforms"):
            init_transforms(transform.transforms, self.height, self.width)

        self.transform = transform
        self.no_eval = no_eval

        if self.no_eval:
            only_perfect_tracks = False

        self.image_index_pairs, self.track_masks = filter_tracks(dataset=self.dataset, image_width=self.width,
                                                                 image_height=self.height,
                                                                 class_remapping=self.class_remapping,
                                                                 min_bbox_height=min_bbox_height,
                                                                 min_bbox_diag=min_bbox_diag,
                                                                 only_perfect_tracks=only_perfect_tracks,
                                                                 scale=scale)

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
        H, W = self.height, self.width
        C = 5  # 通道数

        # 筛选范围内事件
        mask = events['y'] < H
        events = {k: v[mask] for k, v in events.items()}

        if len(events['t']) == 0:
            voxel = np.zeros((C, H, W), dtype=np.float32)
            events['img'] = torch.tensor(voxel)
            return {'img': torch.tensor(voxel)}  # 空图像也要返回

        # 归一化时间为 [0, 1]
        t_min, t_max = events['t'][0], events['t'][-1]
        t_norm = (events['t'] - t_min) / (t_max - t_min + 1e-6)  # 避免除零

        x = events['x'].astype(np.int64)
        y = events['y'].astype(np.int64)
        p = events['p'].astype(np.int64)
        p = (p > 0).astype(np.float32) * 2 - 1  # 255 -> +1, 0 -> -1
        t_bin = (t_norm * C).astype(np.int64)
        t_bin = np.clip(t_bin, 0, C - 1)

        print(f"[DEBUG] Event count: {len(x)}")
        print(f"[DEBUG] x range: {x.min()} - {x.max()}, y range: {y.min()} - {y.max()}")
        print(f"[DEBUG] t_bin range: {t_bin.min()} - {t_bin.max()}, p unique: {np.unique(p)}")

        voxel = np.zeros((C, H, W), dtype=np.float32)

        print(f"[DEBUG] Shapes before voxel fill: x={x.shape}, y={y.shape}, t_bin={t_bin.shape}, p={p.shape}")
        for c, xi, yi, pi in zip(t_bin, x, y, p):
            voxel[c, yi, xi] += pi
        print(f"[DEBUG] Voxel sum: {voxel.sum()}, max: {voxel.max()}, min: {voxel.min()}")
        events['img'] = torch.tensor(voxel)

        return events  # 返回 [5, 480, 640]
        #mask = events['y'] < self.height
        #events = {k: v[mask] for k, v in events.items()}
        #if len(events['t']) > 0:
        #    events['t'] = self.time_window + events['t'] - events['t'][-1]
        #events['p'] = 2 * events['p'].reshape((-1,1)).astype("int8") - 1
        #return events

    def preprocess_image(self, image):
        image = image[:self.scale * self.height]
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = image.unsqueeze(0)
        return image

    def __getitem__(self, idx):
        print(f"[DEBUG] __getitem__ called with index {idx}")
        dataset, image_index_pairs, track_masks, idx = self.rel_index(idx)
        image_index_0, image_index_1 = image_index_pairs[idx]
        image_ts_0, image_ts_1 = dataset.images.timestamps[[image_index_0, image_index_1]]

        # 获取检测框
        detections_0 = self.dataset.get_tracks(image_index_0, mask=track_masks, directory_name=dataset.root.name)
        detections_1 = self.dataset.get_tracks(image_index_1, mask=track_masks, directory_name=dataset.root.name)

        if detections_0 is None or detections_1 is None:
            raise FileNotFoundError("Track files not found in train or test directories.")

        detections_0 = self.preprocess_detections(detections_0)
        detections_1 = self.preprocess_detections(detections_1)

        # 获取 RGB 图像
        image_0 = self.dataset.get_image(image_index_0, directory_name=dataset.root.name)
        if image_0 is None:
            raise FileNotFoundError("Image file not found.")
        image_0 = self.preprocess_image(image_0)

        # 获取事件数据
        events = self.dataset.get_events(image_index_0, directory_name=dataset.root.name)

        if self.num_us >= 0:
            image_ts_1 = image_ts_0 + self.num_us
            events = {k: v[events['t'] < image_ts_1] for k, v in events.items()}
            if not self.no_eval:
                detections_1 = interpolate_tracks(detections_0, detections_1, image_ts_1)

        # 得到 voxel grid 事件图: [5, 480, 640]
        events = self.preprocess_events(events)
        img_event = events['img']   # shape [5, H, W]
        assert img_event.shape == (5, self.height, self.width), f"[ERROR] Event image shape: {img_event.shape}"

        # 处理检测框
        annot = tracks_to_array(detections_1)
        annot = torch.tensor(annot, dtype=torch.float32)  # shape: [N, 5]

        # 返回 retinaNet 格式数据
        return {
            'img': img_event,                    # [5, 480, 640]
            'img_rgb': image_0,                  # [3, 480, 640]
            'annot': annot,                      # [N, 5]
            'sequence': str(dataset.root.name),
            'timestamp': image_ts_1,
            'image_index': image_index_1
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
        raise IndexError
