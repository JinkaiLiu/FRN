from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from retinanet.data.dsec_data import DSEC

# --- Wrapper: converts DAGR DSEC Data object to dict sample ---
class DSECWrapper(Dataset):
    def __init__(self, dsec_dataset):
        self.dataset = dsec_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        # Event image (C, H, W)
        img_event = data.x if hasattr(data, 'x') else torch.zeros(5, 480, 640)

        # RGB image (C, H, W)
        img_rgb = data.image[0] if hasattr(data, 'image') else torch.zeros(3, 480, 640)
        img_rgb = img_rgb.float() / 255.0

        # Annotations [x1, y1, x2, y2, class_id]
        if hasattr(data, 'bbox') and data.bbox.shape[0] > 0:
            bboxes = data.bbox.clone()
            bboxes[:, 2] += bboxes[:, 0]
            bboxes[:, 3] += bboxes[:, 1]
            annot = bboxes
        else:
            annot = torch.zeros((0, 5), dtype=torch.float32)

        return {
            'img': img_event.float(),
            'img_rgb': img_rgb.float(),
            'annot': annot.float(),
            'sequence': getattr(data, 'sequence', ''),
            'timestamp': getattr(data, 't1', torch.tensor(0)).item(),
            'image_index': idx
        }

    def __getattr__(self, name):
        return getattr(self.dataset, name)

def my_collate_fn(batch):
    batch_dict = {}
    # 1. img: list of [N_i, 1] tensors → pad to max_len then stack
    imgs = [b['img'] for b in batch]
    max_len = max([i.shape[0] for i in imgs])
    padded_imgs = []
    for img in imgs:
        pad_len = max_len - img.shape[0]
        if pad_len > 0:
            pad = torch.zeros((pad_len, 1), dtype=img.dtype)
            img = torch.cat([img, pad], dim=0)
        padded_imgs.append(img)
    batch_dict['img'] = torch.stack(padded_imgs)  # [B, max_len, 1]

    # 2. img_rgb: already batchable
    batch_dict['img_rgb'] = torch.stack([b['img_rgb'] for b in batch])

    # 3. annot: list of [N, 5] → pad to max_annots then stack
    annots = [b['annot'] for b in batch]
    max_annots = max([a.shape[0] for a in annots])
    padded_annots = []
    for a in annots:
        pad = torch.ones((max_annots - a.shape[0], 5), dtype=a.dtype) * -1  # -1 填充
        a_padded = torch.cat([a, pad], dim=0)
        padded_annots.append(a_padded)
    batch_dict['annot'] = torch.stack(padded_annots)  # [B, max_annots, 5]

    # 4. 保留其他字段
    for key in batch[0]:
        if key in ['img', 'img_rgb', 'annot']:
            continue
        batch_dict[key] = [b[key] for b in batch]

    return batch_dict

def create_fast_dataloader(root_dir, split='train', batch_size=4, num_workers=4, transform=None, collate_fn=my_collate_fn):
    dsec_dataset = DSEC(
        root=Path(root_dir),
        split=split,
        transform=transform,
        debug=False,
        no_eval=False
    )

    dataset = DSECWrapper(dsec_dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == 'train')
    )

    return dataloader, dataset
