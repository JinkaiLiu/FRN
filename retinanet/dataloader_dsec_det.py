from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import h5py
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
#import skimage.transform
import cv2
from PIL import Image
import yaml
from pathlib import Path
from torchvision import transforms
try:
    import skimage.transform
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    import cv2


class DSECDetDataset(Dataset):
    """
    DSEC-Det dataset loader for FRN
    structure: train+val in train/, test in test/
    """
    def __init__(self, root_dir, split='train', transform=None, event_representation='time_surface', 
                 dt=50, debug=False, split_config=None, 
                 image_height=480, image_width=640,
                 normalize_events=True, normalize_images=False, 
                 use_downsampled_events=False):
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.event_representation = event_representation
        self.dt = dt  # Time window in milliseconds
        self.debug = debug
        self.image_height = image_height
        self.image_width = image_width
        self.normalize_events = normalize_events
        self.normalize_images = normalize_images
        self.use_downsampled_events = use_downsampled_events
        
        # DSEC-Det 8 classes mapping
        self.classes = {
            'pedestrian': 0, 'rider': 1, 'car': 2, 'bus': 3,
            'truck': 4, 'bicycle': 5, 'motorcycle': 6, 'train': 7
        }
        self.labels = {v: k for k, v in self.classes.items()}
        
        # Load sequences based on split config or auto-discover
        if split_config is None:
            self.sequences = self._get_default_sequences()
        else:
            self.sequences = split_config.get(split, [])
        
        # Load all samples from sequences
        self.samples = self._load_samples()
        print(f"DSEC-Det Dataset: Loaded {len(self.samples)} samples for {split} split")
    
    def _get_default_sequences(self):
        """Auto-discover sequences based on split type"""
        if self.split == 'test':
            # Test sequences in independent test directory
            split_dir = self.root_dir / 'test'
        else:
            # Train and val sequences both in train directory
            split_dir = self.root_dir / 'train'
            
        sequences = []
        if split_dir.exists():
            for seq_path in split_dir.iterdir():
                if seq_path.is_dir():
                    sequences.append(seq_path.name)
        return sequences
    
    def _load_samples(self):
        """Load all sample information from sequences"""
        samples = []
        
        for seq in self.sequences:
            print(f" Processing sequence: {seq}")
            # Decide sequence path based on split type
            if self.split == 'test':
                seq_path = self.root_dir / 'test' / seq
            else:
                # Train and val sequences both in train directory
                seq_path = self.root_dir / 'train' / seq
            
            # Define required file paths
            tracks_file = seq_path / 'object_detections' / 'left' / 'tracks.npy'
            timestamps_file = seq_path / 'images' / 'left' / 'exposure_timestamps.txt'
            
            # Choose event file based on parameter
            if self.use_downsampled_events:
                events_file = seq_path / 'events' / 'left' / 'events_2x.h5'
            else:
                events_file = seq_path / 'events' / 'left' / 'events.h5'
            
            # Image directory (prefer rectified, fallback to distorted)
            image_dir = seq_path / 'images' / 'left' / 'rectified'
            if not image_dir.exists():
                image_dir = seq_path / 'images' / 'left' / 'distorted'
            
            # Check if all required files exist
            if not all([tracks_file.exists(), timestamps_file.exists(), 
                       events_file.exists(), image_dir.exists()]):
                print(f"Warning: Missing files for sequence {seq}, skipping...")
                continue
                
            try:
                # Load image timestamps
                with open(timestamps_file, 'r') as f:
                    image_timestamps = []
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            timestamp = line.split(',')[0].strip()
                            try:
                                image_timestamps.append(int(timestamp))
                            except ValueError:
                                continue
                print(f"Loaded {len(image_timestamps)} timestamps")
                # Load annotation data
                tracks_data = np.load(str(tracks_file))
                print(f"Loaded {len(tracks_data)} tracks")
                # Create sample for each image timestamp
                sample_count = 0
                for i, img_timestamp in enumerate(image_timestamps):
                    img_file = image_dir / f'{i:06d}.png'
                    if img_file.exists():
                        # Find annotations for corresponding timestamp (25ms tolerance)
                        mask = np.abs(tracks_data['t'] - img_timestamp) <= 25000
                        frame_tracks = tracks_data[mask]
                        
                        samples.append({
                            'sequence': seq,
                            'image_path': str(img_file),
                            'events_path': str(events_file),
                            'timestamp': img_timestamp,
                            'tracks': frame_tracks,
                            'image_index': i
                        })
                        sample_count += 1
                print(f"✅ Created {sample_count} samples for {seq}")        
            except Exception as e:
                print(f"Error processing sequence {seq}: {e}")
                continue
        print(f"Total samples loaded: {len(samples)}")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Return FRN expected data format:
        - img: Event data [C, H, W] or [H, W, C] depending on representation
        - img_rgb: RGB image [H, W, 3] (will be converted by transforms)
        - annot: Annotations [N, 5] (x1, y1, x2, y2, class_id)
        """
        sample = self.samples[idx]
        
        try:
            # Load RGB image (returns HWC format for transform compatibility)
            img_rgb = self._load_rgb_image(sample['image_path'])
            
            # Load event data (returns CHW format)
            img_event = self._load_events(sample['events_path'], sample['timestamp'])
            
            # Process annotations to FRN format
            annotations = self._process_tracks(sample['tracks'])
            
            # Organize output (FRN compatible format)
            sample_dict = {
                'img': img_event,           # Event data [C, H, W]
                'img_rgb': img_rgb,         # RGB image [H, W, 3] 
                'annot': annotations,       # Annotations [N, 5]
                'sequence': sample['sequence'],
                'timestamp': sample['timestamp'],
                'image_index': sample['image_index']
            }
            
            # Apply transforms (resize, normalize, augment)
            if self.transform:
                sample_dict = self.transform(sample_dict)
            
            return sample_dict
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return self._get_empty_sample()
    
    def _load_rgb_image(self, image_path):
        """Load and preprocess RGB image, return in HWC format for transforms"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # BGR -> RGB conversion
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size if needed
        if img.shape[:2] != (self.image_height, self.image_width):
            img = cv2.resize(img, (self.image_width, self.image_height))
        
        # Normalize to [0,1] range
        img = img.astype(np.float32) / 255.0
        
        # Keep HWC format for transform compatibility (changed from original)
        return img
    
    def _load_events(self, events_path, timestamp):
        """Load and process event data from h5 file"""
        try:
            with h5py.File(events_path, 'r') as f:
                # Load event data arrays
                events_t = f['events/t'][:].copy()
                events_x = f['events/x'][:].copy()
                events_y = f['events/y'][:].copy()
                events_p = f['events/p'][:].copy()
                
                # Apply time offset if available
                if 't_offset' in f:
                    t_offset = f['t_offset'][()]
                    events_t = events_t + t_offset
                
                # Define time window (dt milliseconds before timestamp)
                dt_us = self.dt * 1000  # Convert to microseconds
                start_time = timestamp - dt_us
                end_time = timestamp
                
                # Filter events within time window
                mask = (events_t >= start_time) & (events_t < end_time)
                x = events_x[mask]
                y = events_y[mask]
                t = events_t[mask]
                p = events_p[mask]
                
        except Exception as e:
            print(f"Error loading events: {e}")
            return self._get_empty_events()
        
        # Create event image based on representation method
        if self.event_representation == 'time_surface':
            event_img = self._create_time_surface(x, y, t, p)
        elif self.event_representation == 'event_count':
            event_img = self._create_event_count_image(x, y, p)
        elif self.event_representation == 'binary':
            event_img = self._create_binary_image(x, y, p)
        else:
            event_img = self._create_time_surface(x, y, t, p)  # Default
        
        return event_img
    
    def _create_time_surface(self, x, y, t, p):
        """Create time surface representation [2, H, W]"""
        time_surface = np.zeros((2, self.image_height, self.image_width), dtype=np.float32)
        
        if len(x) > 0:
            # Normalize timestamps to [0, 1] range
            t_normalized = (t - t.min()) / (t.max() - t.min() + 1e-6)
            
            # Filter valid pixel coordinates
            valid_mask = (x >= 0) & (x < self.image_width) & (y >= 0) & (y < self.image_height)
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            t_valid = t_normalized[valid_mask]
            p_valid = p[valid_mask]
            
            # Fill time surface by polarity
            for i in range(len(x_valid)):
                polarity_idx = 1 if p_valid[i] > 0 else 0  # Positive/negative events
                time_surface[polarity_idx, y_valid[i], x_valid[i]] = t_valid[i]
        
        # Normalize to [-1, 1] range if enabled
        if self.normalize_events:
            time_surface = time_surface * 2.0 - 1.0
        
        return torch.from_numpy(time_surface.copy()).float()
    
    def _create_event_count_image(self, x, y, p):
        """Create event count representation [2, H, W]"""
        event_count = np.zeros((2, self.image_height, self.image_width), dtype=np.float32)
        
        if len(x) > 0:
            # Filter valid coordinates
            valid_mask = (x >= 0) & (x < self.image_width) & (y >= 0) & (y < self.image_height)
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            p_valid = p[valid_mask]
            
            # Count events by polarity
            for i in range(len(x_valid)):
                polarity_idx = 1 if p_valid[i] > 0 else 0
                event_count[polarity_idx, y_valid[i], x_valid[i]] += 1
        
        # Apply log normalization if enabled
        if self.normalize_events:
            event_count = np.log(event_count + 1)
            max_val = event_count.max()
            if max_val > 0:
                event_count = event_count / max_val
        
        return torch.from_numpy(event_count.copy()).float()
    
    def _create_binary_image(self, x, y, p):
        """Create binary event representation [2, H, W]"""
        binary_image = np.zeros((2, self.image_height, self.image_width), dtype=np.float32)
        
        if len(x) > 0:
            # Filter valid coordinates
            valid_mask = (x >= 0) & (x < self.image_width) & (y >= 0) & (y < self.image_height)
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            p_valid = p[valid_mask]
            
            # Set binary values by polarity
            for i in range(len(x_valid)):
                polarity_idx = 1 if p_valid[i] > 0 else 0
                binary_image[polarity_idx, y_valid[i], x_valid[i]] = 1.0
        
        return torch.from_numpy(binary_image.copy()).float()
    
    def _process_tracks(self, tracks):
        """Process annotation data to FRN format [N, 5]"""
        if len(tracks) == 0:
            return torch.zeros((0, 5), dtype=torch.float32)
        
        annotations = []
        for track in tracks:
            # Extract bounding box parameters
            x = float(track['x'])
            y = float(track['y'])
            w = float(track['w'])
            h = float(track['h'])
            class_id = int(track['class_id'])
            
            # Filter invalid annotations
            if w < 1 or h < 1 or class_id >= len(self.classes):
                continue
            
            # Convert to (x1, y1, x2, y2) format and clip to image bounds
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(self.image_width, x + w)
            y2 = min(self.image_height, y + h)
            
            # Skip degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            annotations.append([x1, y1, x2, y2, class_id])
        
        if len(annotations) == 0:
            return torch.zeros((0, 5), dtype=torch.float32)
        
        return torch.tensor(annotations, dtype=torch.float32)
    
    def _get_empty_sample(self):
        """Return empty sample for error cases"""
        return {
            'img': torch.zeros(2, self.image_height, self.image_width),
            'img_rgb': torch.zeros(self.image_height, self.image_width, 3),  # Changed: HWC format
            'annot': torch.zeros(0, 5),
            'sequence': '',
            'timestamp': 0,
            'image_index': 0
        }
    
    def _get_empty_events(self):
        """Return empty event data"""
        return torch.zeros(2, self.image_height, self.image_width)
    
    # FRN compatibility methods
    def name_to_label(self, name):
        """Convert class name to label index"""
        return self.classes[name]
    
    def label_to_name(self, label):
        """Convert label index to class name"""
        return self.labels[label]
    
    def num_classes(self):
        """Return number of classes"""
        return len(self.classes)
    
    def image_aspect_ratio(self, image_index):
        """Return image aspect ratio for sampler"""
        return float(self.image_width) / float(self.image_height)

class Resizer(object):
    """Resize images and annotations to target size"""
    def __init__(self, dataset_name='dsec'):
        self.dataset_name = dataset_name

    def __call__(self, sample):
        # Set target sizes based on dataset
        if self.dataset_name == 'dsec':
            max_side = 640
            min_side = 480
        else:
            max_side = 640
            min_side = 480
            
        image = sample['img_rgb']  # RGB image in HWC format
        annots = sample['annot']

        # Get image dimensions
        if len(image.shape) == 3:
            rows, cols, cns = image.shape
        else:
            rows, cols = image.shape
            cns = 1

        # Calculate scale factor
        smallest_side = min(rows, cols)
        scale = min_side / smallest_side
        largest_side = max(rows, cols)

        # Prevent over-scaling
        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # Resize image
        if len(image.shape) == 3:
            if SKIMAGE_AVAILABLE:
                image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
            else:
                image = cv2.resize(image, (int(round(cols*scale)), int(round(rows*scale))))
        else:
            if SKIMAGE_AVAILABLE:
                image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
            else:
                image = cv2.resize(image, (int(round(cols*scale)), int(round(rows*scale))))

        # Scale annotations accordingly
        if len(annots) > 0:
            annots[:, :4] *= scale  # Scale x1, y1, x2, y2

        return {'img': sample['img'], 'img_rgb': torch.from_numpy(image.astype(np.float32)), 'annot': torch.from_numpy(annots), 'scale': scale}

class Normalizer(object):
    """Normalize RGB images using ImageNet statistics"""
    def __init__(self, dataset_name='dsec'):
        # ImageNet normalization parameters
        if dataset_name == 'dsec':
            self.mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
            self.std = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)
        else:
            self.mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
            self.std = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)

    def __call__(self, sample):
        image = sample['img_rgb']
        annots = sample['annot']
        
        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            image = image.numpy()

        # Apply normalization: (image - mean) / std
        return {'img': sample['img'],'img_rgb': torch.from_numpy(((image.astype(np.float32)-self.mean)/self.std)), 'annot': annots}

class Augmenter(object):
    """Data augmentation with horizontal flipping"""
    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image_rgb = sample['img_rgb']
            image_event = sample['img']
            annots = sample['annot']
            
            # Convert to numpy arrays if needed
            if isinstance(image_rgb, torch.Tensor):
                image_rgb = image_rgb.numpy()
            if isinstance(image_event, torch.Tensor):
                image_event = image_event.numpy()
            if isinstance(annots, torch.Tensor):
                annots = annots.numpy()
            
            # Horizontal flip for RGB image (HWC format)
            image_rgb = image_rgb[:, ::-1, :]
            # Horizontal flip for event image (CHW format) 
            image_event = image_event[:, :, ::-1]

            rows, cols, channels = image_rgb.shape

            # Adjust annotation coordinates for flipping
            if len(annots) > 0:
                x1 = annots[:, 0].copy()
                x2 = annots[:, 2].copy()
                
                x_tmp = x1.copy()

                # Flip x coordinates: new_x1 = width - old_x2, new_x2 = width - old_x1
                annots[:, 0] = cols - x2
                annots[:, 2] = cols - x_tmp

            sample = {'img': torch.from_numpy(image_event), 'img_rgb': torch.from_numpy(image_rgb), 'annot': torch.from_numpy(annots)}

        return sample

def collater(data):
    """
    Collate function for batching data 
    Handles variable-sized images and annotations
    """
    # Extract data from batch
    imgs = [s['img'] for s in data]          # Event images
    imgs_rgb = [s['img_rgb'] for s in data]  # RGB images
    annots = [s['annot'] for s in data]      # Annotations
    scales = [s.get('scale', 1.0) for s in data]  # Scale factors
        
    batch_size = len(imgs)

    # Handle event images (can be tensors or numpy arrays)
    if isinstance(imgs[0], torch.Tensor):
        if len(imgs[0].shape) == 3:  # CHW format
            max_height = max([img.shape[1] for img in imgs])
            max_width = max([img.shape[2] for img in imgs])
            padded_imgs = torch.zeros(batch_size, imgs[0].shape[0], max_height, max_width)
            
            # Pad each image to max size
            for i in range(batch_size):
                img = imgs[i]
                padded_imgs[i, :, :img.shape[1], :img.shape[2]] = img
        else:
            padded_imgs = torch.stack(imgs, dim=0)  # Same size images
    else:
        # Handle numpy arrays
        heights = [int(s.shape[0]) for s in imgs]
        widths = [int(s.shape[1]) for s in imgs]
        max_height = np.array(heights).max()
        max_width = np.array(widths).max()
        
        if len(imgs[0].shape) == 3:  # HWC format
            padded_imgs = torch.zeros(batch_size, max_height, max_width, imgs[0].shape[2])
            
            for i in range(batch_size):
                img = torch.tensor(imgs[i])
                padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img
                
            padded_imgs = padded_imgs.permute(0, 3, 1, 2)  # Convert to BCHW
        else:
            padded_imgs = torch.zeros(batch_size, max_height, max_width)
            
            for i in range(batch_size):
                img = torch.tensor(imgs[i])
                padded_imgs[i, :int(img.shape[0]), :int(img.shape[1])] = img

    # Handle RGB images (similar logic as event images)
    if isinstance(imgs_rgb[0], torch.Tensor):
        if len(imgs_rgb[0].shape) == 3:  # CHW format
            max_height_rgb = max([img.shape[1] for img in imgs_rgb])
            max_width_rgb = max([img.shape[2] for img in imgs_rgb])
            padded_imgs_rgb = torch.zeros(batch_size, imgs_rgb[0].shape[0], max_height_rgb, max_width_rgb)
            
            for i in range(batch_size):
                img = imgs_rgb[i]
                padded_imgs_rgb[i, :, :img.shape[1], :img.shape[2]] = img
        else:
            padded_imgs_rgb = torch.stack(imgs_rgb, dim=0)
    else:
        # Handle numpy arrays
        heights_rgb = [int(s.shape[0]) for s in imgs_rgb]
        widths_rgb = [int(s.shape[1]) for s in imgs_rgb]
        max_height_rgb = np.array(heights_rgb).max()
        max_width_rgb = np.array(widths_rgb).max()
        
        padded_imgs_rgb = torch.zeros(batch_size, max_height_rgb, max_width_rgb, imgs_rgb[0].shape[2])
        
        for i in range(batch_size):
            img_rgb = torch.tensor(imgs_rgb[i]) if not isinstance(imgs_rgb[i], torch.Tensor) else imgs_rgb[i]
            padded_imgs_rgb[i, :int(img_rgb.shape[0]), :int(img_rgb.shape[1]), :] = img_rgb

        padded_imgs_rgb = padded_imgs_rgb.permute(0, 3, 1, 2)  # Convert to BCHW

    # Handle annotations (pad to maximum number of annotations in batch)
    max_num_annots = max(annot.shape[0] for annot in annots) if any(len(annot) > 0 for annot in annots) else 1
    
    # Create padded annotation tensor (-1 padding for empty slots)
    annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

    for idx, annot in enumerate(annots):
        if isinstance(annot, torch.Tensor):
            annot = annot
        else:
            annot = torch.tensor(annot)
            
        if annot.shape[0] > 0:
            annot_padded[idx, :annot.shape[0], :] = annot

    return {'img': padded_imgs, 'img_rgb': padded_imgs_rgb, 'annot': annot_padded, 'scale': scales}

class AspectRatioBasedSampler(Sampler):
    """Sample batches based on aspect ratio to minimize padding"""
    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        # Shuffle groups for each epoch
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        """Group images by aspect ratio for efficient batching"""
        # Sort images by aspect ratio
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))
        
        # Create groups of batch_size
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

def load_split_config(config_path):
    """Load YAML configuration file for dataset splits"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_dsec_det_dataloader(root_dir, split='train', batch_size=8, num_workers=4,
                              shuffle=None, event_representation='time_surface', dt=50,
                              image_height=480, image_width=640, augment=True, debug=False,
                              split_config_path=None, normalize_events=True, normalize_images=True,
                              use_downsampled_events=False, use_aspect_ratio_sampler=False):
    """
    Create DSEC-Det dataloader for FRN training
    
    Args:
        root_dir: DSEC dataset root directory
        split: Dataset split ('train', 'val', 'test')
        batch_size: Batch size for training
        num_workers: Number of data loading processes
        shuffle: Whether to shuffle data (auto-determined if None)
        event_representation: Event representation method ('time_surface', 'event_count', 'binary')
        dt: Event time window in milliseconds
        image_height/width: Target image dimensions
        augment: Whether to apply data augmentation (training only)
        debug: Enable debug mode for visualization
        split_config_path: Path to YAML split configuration file
        normalize_events: Whether to normalize event data
        normalize_images: Whether to apply ImageNet normalization to RGB images
        use_downsampled_events: Whether to use 2x downsampled event data
        use_aspect_ratio_sampler: Whether to use aspect ratio based sampling
    
    Returns:
        tuple: (dataloader, dataset)
    """
    
    # Auto-determine shuffle behavior
    if shuffle is None:
        shuffle = (split == 'train')
    
    # Load split configuration if provided
    split_config = None
    if split_config_path and os.path.exists(split_config_path):
        split_config = load_split_config(split_config_path)
    
    # Build transform pipeline
    transform_list = []
    
    # Add augmentation for training only
    if augment and split == 'train':
        transform_list.append(Augmenter())
    
    # Add resizing (always needed)
    transform_list.append(Resizer(dataset_name='dsec'))
    
    # Add normalization if enabled
    if normalize_images:
        transform_list.append(Normalizer(dataset_name='dsec'))
    
    # Compose transforms
    transform = transforms.Compose(transform_list) if transform_list else None
    
    # Create dataset instance
    dataset = DSECDetDataset(
        root_dir=root_dir,
        split=split,
        transform=transform,
        event_representation=event_representation,
        dt=dt,
        debug=debug,
        split_config=split_config,
        image_height=image_height,
        image_width=image_width,
        normalize_events=normalize_events,
        normalize_images=normalize_images,
        use_downsampled_events=use_downsampled_events
    )
    
    # Create dataloader with appropriate sampler
    if use_aspect_ratio_sampler and split == 'train':
        # Use aspect ratio based sampler for training efficiency
        sampler = AspectRatioBasedSampler(dataset, batch_size, drop_last=True)
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collater,
            num_workers=num_workers
        )
    else:
        # Use standard dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collater,
            num_workers=num_workers,
            drop_last=(split == 'train'),  # Drop last incomplete batch for training
            pin_memory=torch.cuda.is_available()  # Enable memory pinning if GPU available
        )
    
    return dataloader, dataset