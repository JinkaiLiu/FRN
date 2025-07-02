from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from retinanet.data.dsec_data import DSEC
except ImportError:
    print("[WARNING] Could not import from retinanet.data.dsec_data, trying local import")
    from dsec_data import DSEC

class DSECWrapper(Dataset):
    
    def __init__(self, dsec_dataset):
        self.dataset = dsec_dataset
        print(f"[DEBUG] DSECWrapper initialized with {len(dsec_dataset)} samples")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            data = self.dataset[idx]
            
            if isinstance(data, dict):
                return data
            else:
                print(f"[WARNING] Unexpected data format at index {idx}: {type(data)}")
                
                img_event = getattr(data, 'img', torch.zeros(5, 480, 640))
                img_rgb = getattr(data, 'img_rgb', torch.zeros(1, 3, 480, 640))
                annot = getattr(data, 'annot', torch.zeros((0, 5)))
                
                return {
                    'img': img_event,
                    'img_rgb': img_rgb,
                    'annot': annot,
                    'sequence': getattr(data, 'sequence', ''),
                    'timestamp': getattr(data, 'timestamp', 0),
                    'image_index': idx
                }
                
        except Exception as e:
            print(f"[ERROR] Error in DSECWrapper.__getitem__({idx}): {e}")
            
            return {
                'img': torch.zeros(5, 480, 640, dtype=torch.float32),
                'img_rgb': torch.zeros(1, 3, 480, 640, dtype=torch.float32),
                'annot': torch.zeros((0, 5), dtype=torch.float32),
                'sequence': '',
                'timestamp': 0,
                'image_index': idx
            }
    
    def __getattr__(self, name):
        return getattr(self.dataset, name)

def safe_collate_fn(batch):
    try:
        batch_dict = {}
        
        imgs = [b['img'] for b in batch]
        
        expected_shape = (5, 480, 640)
        processed_imgs = []
        
        for i, img in enumerate(imgs):
            if img.dim() == 3 and img.shape == expected_shape:
                processed_imgs.append(img)
            elif img.dim() == 3 and img.shape[0] == 5:
                print(f"[WARNING] Resizing event img at batch index {i}: {img.shape} -> {expected_shape}")
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0), 
                    size=(480, 640), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                processed_imgs.append(img)
            elif img.dim() == 2:
                print(f"[WARNING] Reshaping img at batch index {i}: {img.shape}")
                if img.shape[0] == 5 * 480 * 640:
                    img = img.view(5, 480, 640)
                else:
                    print(f"[ERROR] Cannot reshape img with shape {img.shape}")
                    img = torch.zeros(5, 480, 640)
                processed_imgs.append(img)
            else:
                print(f"[ERROR] Unexpected img shape at batch index {i}: {img.shape}")
                processed_imgs.append(torch.zeros(5, 480, 640))
        
        batch_dict['img'] = torch.stack(processed_imgs)
        
        rgb_imgs = [b['img_rgb'] for b in batch]
        processed_rgb = []
        
        for i, rgb in enumerate(rgb_imgs):
            if rgb.dim() == 4 and rgb.shape[0] == 1:
                rgb = rgb.squeeze(0)
            elif rgb.dim() == 3 and rgb.shape[0] == 3:
                if rgb.shape[1:] != (480, 640):
                    print(f"[WARNING] Resizing RGB at batch index {i}: {rgb.shape}")
                    rgb = torch.nn.functional.interpolate(
                        rgb.unsqueeze(0),
                        size=(480, 640),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
            else:
                print(f"[WARNING] Unexpected RGB shape at batch index {i}: {rgb.shape}")
                rgb = torch.zeros(3, 480, 640)
            processed_rgb.append(rgb)
        
        batch_dict['img_rgb'] = torch.stack(processed_rgb)
        
        annots = [b['annot'] for b in batch]
        
        if len(annots) > 0:
            max_annots = max([a.shape[0] for a in annots]) if any(a.shape[0] > 0 for a in annots) else 1
            
            padded_annots = []
            for a in annots:
                if a.shape[0] == 0:
                    padded = torch.ones((max_annots, 5), dtype=a.dtype) * -1
                elif a.shape[0] < max_annots:
                    pad_size = max_annots - a.shape[0]
                    pad = torch.ones((pad_size, 5), dtype=a.dtype) * -1
                    padded = torch.cat([a, pad], dim=0)
                else:
                    padded = a[:max_annots]
                
                padded_annots.append(padded)
            
            batch_dict['annot'] = torch.stack(padded_annots)
        else:
            batch_dict['annot'] = torch.ones((len(batch), 1, 5), dtype=torch.float32) * -1
        
        for key in ['sequence', 'timestamp', 'image_index']:
            if key in batch[0]:
                batch_dict[key] = [b[key] for b in batch]
        
        return batch_dict
        
    except Exception as e:
        print(f"[ERROR] Error in collate_fn: {e}")
        import traceback
        traceback.print_exc()
        
        batch_size = len(batch)
        return {
            'img': torch.zeros(batch_size, 5, 480, 640, dtype=torch.float32),
            'img_rgb': torch.zeros(batch_size, 3, 480, 640, dtype=torch.float32),
            'annot': torch.ones(batch_size, 1, 5, dtype=torch.float32) * -1,
            'sequence': [''] * batch_size,
            'timestamp': [0] * batch_size,
            'image_index': list(range(batch_size))
        }

def create_fast_dataloader(root_dir, split='train', batch_size=4, num_workers=4, transform=None, collate_fn=None):
    
    if collate_fn is None:
        collate_fn = safe_collate_fn
    
    print(f"[DEBUG] Creating DSEC dataset from {root_dir} for split '{split}'")
    
    try:
        dsec_dataset = DSEC(
            root=Path(root_dir),
            split=split,
            transform=transform,
            #debug=(batch_size == 1),
            debug=False,
            no_eval=False,
            scale=1,
            cropped_height=480
        )
        
        print(f"[DEBUG] DSEC dataset created successfully")
        print(f"[DEBUG] Dataset length: {len(dsec_dataset)}")
        print(f"[DEBUG] Dataset classes: {dsec_dataset.classes}")
        print(f"[DEBUG] Image size: {dsec_dataset.height}x{dsec_dataset.width}")
        
    except Exception as e:
        print(f"[ERROR] Failed to create DSEC dataset: {e}")
        
        alternative_paths = [
            Path(root_dir) / "train",
            Path(root_dir) / split,
            Path(root_dir).parent / split
        ]
        
        for alt_path in alternative_paths:
            if alt_path.exists():
                print(f"[INFO] Trying alternative path: {alt_path}")
                try:
                    dsec_dataset = DSEC(
                        root=alt_path,
                        split=split,
                        transform=transform,
                        debug=(batch_size == 1),
                        no_eval=False,
                        scale=1,
                        cropped_height=480
                    )
                    print(f"[SUCCESS] Dataset loaded from {alt_path}")
                    break
                except Exception as e2:
                    print(f"[ERROR] Alternative path {alt_path} also failed: {e2}")
                    continue
        else:
            raise RuntimeError(f"Could not load dataset from any path. Original error: {e}")
    
    dataset = DSECWrapper(dsec_dataset)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=0 if batch_size == 1 else min(num_workers, 4),
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == 'train'),
        persistent_workers=False
    )
    
    print(f"[DEBUG] DataLoader created successfully")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Num workers: {dataloader.num_workers}")
    print(f"  - Expected tensor sizes: Event[B,5,480,640], RGB[B,3,480,640]")
    
    return dataloader, dataset
