import argparse
import collections
import numpy as np
import time
import math
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import os
from retinanet.dataloader_fast_combined import create_fast_dataloader as create_dsec_det_dataloader
from retinanet import model
from retinanet.csv_eval_dsec_det import evaluate, evaluate_coco_map

assert torch.__version__.split('.')[0] == '1'
print('CUDA available: {}'.format(torch.cuda.is_available()))


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m}m {s:.0f}s'


def normalize_event_data(img_event, method='tanh'):
    if method == 'tanh':
        img_event = torch.tanh(img_event / 5.0)
    elif method == 'clip':
        img_event = torch.clamp(img_event, -2, 2)
    elif method == 'minmax':
        min_val = img_event.min()
        max_val = img_event.max()
        if max_val > min_val:
            img_event = 2 * (img_event - min_val) / (max_val - min_val) - 1
    
    return img_event


def validate_annotations(annot, img_width=320, img_height=215):
    if annot.shape[0] == 0:
        return annot
    
    x1, y1, x2, y2, cls = annot[:, 0], annot[:, 1], annot[:, 2], annot[:, 3], annot[:, 4]
    
    x1_new = torch.minimum(x1, x2)
    x2_new = torch.maximum(x1, x2)
    y1_new = torch.minimum(y1, y2)
    y2_new = torch.maximum(y1, y2)
    
    min_size = 2.0
    width = x2_new - x1_new
    height = y2_new - y1_new
    
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
    
    x1_new = torch.clamp(x1_new, 0, img_width - min_size)
    y1_new = torch.clamp(y1_new, 0, img_height - min_size)
    x2_new = torch.clamp(x2_new, min_size, img_width)
    y2_new = torch.clamp(y2_new, min_size, img_height)
    
    valid_mask = (x2_new > x1_new) & (y2_new > y1_new) & (x2_new - x1_new >= 1) & (y2_new - y1_new >= 1)
    
    if not valid_mask.all():
        print(f"[WARNING] Removing {(~valid_mask).sum()} invalid annotations")
    
    fixed_annot = torch.stack([x1_new, y1_new, x2_new, y2_new, cls], dim=1)
    
    fixed_annot = fixed_annot[valid_mask]
    
    return fixed_annot


def debug_batch_data(data_batch, batch_idx):
    print(f"\n=== Debug Batch {batch_idx} (FIXED) ===")
    
    img_event = data_batch['img']
    img_rgb = data_batch['img_rgb']
    annot = data_batch['annot']
    
    batch_size = img_event.shape[0]
    print(f"Batch size: {batch_size}")
    print(f"Event data shape: {img_event.shape}, type: {img_event.dtype}")
    print(f"RGB data shape: {img_rgb.shape}, type: {img_rgb.dtype}")
    print(f"Annotation shape: {annot.shape}, type: {annot.dtype}")
    
    sample_event = img_event[0]
    sample_rgb = img_rgb[0]
    sample_annot = annot[0]
    
    print(f"First sample - event: {sample_event.shape}, RGB: {sample_rgb.shape}, annotation: {sample_annot.shape}")
    
    if sample_event.abs().max() > 5:
        print(f"Event data range too large: [{sample_event.min():.3f}, {sample_event.max():.3f}], normalizing...")
        sample_event_norm = normalize_event_data(sample_event, method='tanh')
        print(f"Normalized event range: [{sample_event_norm.min():.3f}, {sample_event_norm.max():.3f}]")
    else:
        print(f"Event data range: [{sample_event.min():.3f}, {sample_event.max():.3f}]")
    
    print(f"RGB data range: [{sample_rgb.min():.3f}, {sample_rgb.max():.3f}]")
    
    if sample_annot.shape[0] > 0:
        print(f"\nAnnotations before fixing:")
        valid_indices = torch.where(sample_annot[:, 0] != -1)[0]
        for i, ann_idx in enumerate(valid_indices[:3]):
            ann = sample_annot[ann_idx]
            x1, y1, x2, y2, cls = ann[0].item(), ann[1].item(), ann[2].item(), ann[3].item(), ann[4].item()
            w, h = x2 - x1, y2 - y1
            print(f"  Annotation {i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] w={w:.1f} h={h:.1f} cls={int(cls)}")
        
        fixed_annot = validate_annotations(sample_annot)
        
        print(f"\nAnnotations after fixing ({len(fixed_annot)} valid):")
        for i, ann in enumerate(fixed_annot[:3]):
            x1, y1, x2, y2, cls = ann[0].item(), ann[1].item(), ann[2].item(), ann[3].item(), ann[4].item()
            w, h = x2 - x1, y2 - y1
            print(f"  Annotation {i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] w={w:.1f} h={h:.1f} cls={int(cls)}")
            
            if x2 <= x1 or y2 <= y1 or w < 1 or h < 1:
                print(f"    Still has issues!")
            else:
                print(f"    Fixed")
    
    issues = []
    
    if sample_event.abs().max() > 5:
        issues.append("Event data range too large (auto-fixable)")
    if sample_rgb.min() < 0 or sample_rgb.max() > 1.5:
        issues.append("RGB data range abnormal")
    
    if sample_annot.shape[0] > 0:
        valid_annot = validate_annotations(sample_annot)
        if len(valid_annot) == 0:
            issues.append("No valid annotations")
        elif len(valid_annot) < sample_annot.shape[0]:
            issues.append(f"Some annotations invalid (fixed: {len(valid_annot)}/{sample_annot.shape[0]})")
    
    if issues:
        print(f"\n  Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
        return True
    else:
        print("\ Batch data normal")
        return True


def safe_training_step(retinanet, data, optimizer, iter_num, loss_threshold=50.0,
                      scaler=None, use_amp=False):
    try:
        if isinstance(data['img_rgb'], list):
            img_rgb = torch.stack(data['img_rgb']).cuda(non_blocking=True).float()
        else:
            img_rgb = data['img_rgb'].cuda(non_blocking=True).float()
    
        if isinstance(data['img'], list):
            img_event = torch.stack(data['img']).cuda(non_blocking=True).float()
        else:
            img_event = data['img'].cuda(non_blocking=True).float()
        
        if img_event.abs().max() > 5:
            img_event = normalize_event_data(img_event, method='tanh')
        
        if isinstance(data['annot'], list):
            fixed_annots = []
            max_annots = 0
            
            for a in data['annot']:
                fixed_a = validate_annotations(a)
                fixed_annots.append(fixed_a)
                max_annots = max(max_annots, len(fixed_a))
            
            if max_annots == 0:
                max_annots = 1
            
            padded_annots = []
            for fixed_a in fixed_annots:
                if len(fixed_a) == 0:
                    padded = torch.ones((max_annots, 5), dtype=torch.float32) * -1
                elif len(fixed_a) < max_annots:
                    pad_len = max_annots - len(fixed_a)
                    pad = torch.ones((pad_len, 5), dtype=fixed_a.dtype) * -1
                    padded = torch.cat([fixed_a, pad], dim=0)
                else:
                    padded = fixed_a[:max_annots]
                padded_annots.append(padded)
            
            annot = torch.stack(padded_annots).cuda(non_blocking=True).float()
        else:
            batch_size = data['annot'].shape[0]
            fixed_batch_annots = []
            
            for i in range(batch_size):
                fixed_annot = validate_annotations(data['annot'][i])
                fixed_batch_annots.append(fixed_annot)
            
            max_annots = max([len(a) for a in fixed_batch_annots]) if fixed_batch_annots else 1
            if max_annots == 0:
                max_annots = 1
            
            padded_annots = []
            for fixed_a in fixed_batch_annots:
                if len(fixed_a) == 0:
                    padded = torch.ones((max_annots, 5), dtype=torch.float32) * -1
                elif len(fixed_a) < max_annots:
                    pad_len = max_annots - len(fixed_a)
                    pad = torch.ones((pad_len, 5), dtype=fixed_a.dtype) * -1
                    padded = torch.cat([fixed_a, pad], dim=0)
                else:
                    padded = fixed_a[:max_annots]
                padded_annots.append(padded)
            
            annot = torch.stack(padded_annots).cuda(non_blocking=True).float()
        
        if torch.isnan(img_event).any() or torch.isinf(img_event).any():
            print(f"Iteration {iter_num}: NaN/Inf in event data")
            return None, None, None
        if torch.isnan(img_rgb).any() or torch.isinf(img_rgb).any():
            print(f"Iteration {iter_num}: NaN/Inf in RGB data")
            return None, None, None
        
        valid_annot_mask = annot[:, :, 0] != -1
        if not valid_annot_mask.any():
            print(f"Iteration {iter_num}: No valid annotations, skipping")
            return None, None, None
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast():
                classification_loss, regression_loss = retinanet([img_rgb, img_event, annot])
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                total_loss = classification_loss + regression_loss
        else:
            classification_loss, regression_loss = retinanet([img_rgb, img_event, annot])
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            total_loss = classification_loss + regression_loss
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Iteration {iter_num}: Loss NaN/Inf - classification:{classification_loss.item():.6f}, regression:{regression_loss.item():.6f}")
            return None, None, None
            
        if total_loss.item() > loss_threshold:
            print(f"Iteration {iter_num}: Loss too large {total_loss.item():.1f} > {loss_threshold}")
            return None, None, None
        
        if use_amp and scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 1.0)
            optimizer.step()
        
        if iter_num % 50 == 0:
            print(f"Iteration {iter_num}: Success - loss={total_loss.item():.4f}")
            valid_annots = valid_annot_mask.sum().item()
            print(f"  Valid annotations: {valid_annots}, event range: [{img_event.min():.3f}, {img_event.max():.3f}]")
        
        return classification_loss.item(), regression_loss.item(), total_loss.item()
        
    except Exception as e:
        print(f"Iteration {iter_num}: Training error - {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def train_epoch(dataloader, retinanet, optimizer, epoch_num, start_time, loss_threshold=50.0,
               scaler=None, use_amp=False):
    retinanet.train()
    if hasattr(retinanet.module, 'freeze_bn'):
        retinanet.module.freeze_bn()
    
    loss_hist = collections.deque(maxlen=100)
    epoch_losses = []
    valid_iterations = 0
    total_iterations = 0
    
    for iter_num, data in enumerate(dataloader):
        total_iterations += 1
        
        cls_loss, reg_loss, total_loss = safe_training_step(
            retinanet, data, optimizer, iter_num, loss_threshold, scaler, use_amp
        )
        
        if total_loss is not None:
            valid_iterations += 1
            loss_hist.append(total_loss)
            
            if iter_num % 10 == 0:
                avg_loss = np.mean(loss_hist) if loss_hist else 0
                valid_rate = 100 * valid_iterations / total_iterations
                print(f'[{time_since(start_time)}] Epoch {epoch_num} | Iter {iter_num} | '
                      f'Loss: cls={cls_loss:.4f} reg={reg_loss:.4f} total={total_loss:.4f} | '
                      f'Avg: {avg_loss:.4f} | Valid: {valid_iterations}/{total_iterations} ({valid_rate:.1f}%)')
                epoch_losses.append(avg_loss)
        
        if iter_num > 0 and iter_num % 100 == 0:
            torch.cuda.empty_cache()
    
    if epoch_losses:
        avg_epoch_loss = np.mean(epoch_losses)
        valid_rate = 100 * valid_iterations / total_iterations
        print(f"Epoch {epoch_num} summary: {valid_iterations}/{total_iterations} valid ({valid_rate:.1f}%), avg loss: {avg_epoch_loss:.4f}")
        return avg_epoch_loss
    else:
        print(f"Epoch {epoch_num}: No valid iterations!")
        return float('inf')


def evaluate_epoch(retinanet, dataset_val, epoch_num, save_folder):
    print(f"\n{'='*20} Validating Epoch {epoch_num} {'='*20}")
    
    eval_save_folder = os.path.join(save_folder, f'eval_epoch_{epoch_num}')
    os.makedirs(eval_save_folder, exist_ok=True)
    
    mean_ap = evaluate(
        generator=dataset_val,
        retinanet=retinanet,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        save_detection=True,
        save_folder=eval_save_folder,
        load_detection=False,
        save_path=eval_save_folder
    )
    
    print(f"\nEpoch {epoch_num} validation results:")
    print(f"mAP@0.5: {mean_ap:.4f}")
    
    return mean_ap


def main():
    parser = argparse.ArgumentParser(description='DSEC Detection Training Script (Fixed Version)')
    
    parser.add_argument('--root_dir', default='/media/data/hucao/zhenwu/hucao/DSEC/DSEC_Det', 
                       help='DSEC dataset root directory')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--loss_threshold', type=float, default=50.0, help='Loss threshold')
    
    parser.add_argument('--fusion', default='fpn_fusion', choices=['fpn_fusion', 'rgb', 'event'],
                       help='Fusion model type')
    parser.add_argument('--depth', type=int, default=50, choices=[18, 34, 50], 
                       help='ResNet depth')
    
    parser.add_argument('--eval_interval', type=int, default=5, help='Evaluation interval (epochs)')
    parser.add_argument('--eval_coco', action='store_true', help='Also evaluate COCO-style mAP')
    
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    
    parser.add_argument('--debug_data', action='store_true', help='Enable data debugging')
    parser.add_argument('--continue_training', action='store_true', help='Continue training from checkpoint')
    parser.add_argument('--checkpoint', default='', help='Checkpoint path')
    parser.add_argument('--save_dir', default='/media/data/hucao/zehua/results_dsec/fixed_version',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*60)
    print("DSEC Detection Training Script (Fixed Version)")
    print("="*60)
    print(f"Root directory: {args.root_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Loss threshold: {args.loss_threshold}")
    print(f"Training epochs: {args.epochs}")
    print(f"Fusion model: {args.fusion}")
    print(f"Mixed precision: {args.use_amp}")
    print(f"Evaluation interval: {args.eval_interval}")
    print(f"COCO evaluation: {args.eval_coco}")
    
    print("\nLoading training dataset...")
    dataloader_train, dataset_train = create_dsec_det_dataloader(
        root_dir=args.root_dir,
        split='train',
        batch_size=args.batch_size,
        num_workers=0 if args.debug_data else args.num_workers
    )
    
    print(f"Training dataset loaded: {len(dataset_train)} samples")
    
    print("Loading validation dataset...")
    _, dataset_val = create_dsec_det_dataloader(
        root_dir=args.root_dir,
        split='val',  
        batch_size=1,
        num_workers=1
    )
    
    print(f"Validation dataset loaded: {len(dataset_val)} samples")
    
    if args.debug_data:
        print("\n" + "="*40)
        print("Data Debug Mode (Fixed Version)")
        print("="*40)
        
        for batch_idx, data in enumerate(dataloader_train):
            success = debug_batch_data(data, batch_idx)
            if not success:
                print(f" Batch {batch_idx} has unfixable issues!")
            if batch_idx >= 4:
                break
        
        print("\nDebugging complete. Use command without --debug_data to start training.")
        return
    
    print("\nCreating model...")
    retinanet = model.resnet50(
        dataset_name='dsec',
        num_classes=dataset_train.num_classes,
        fusion_model=args.fusion,
        pretrained=True
    )
    print(f"Model created: ResNet{args.depth} with {args.fusion} fusion")
    print(f"Number of classes: {dataset_train.num_classes}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        retinanet = retinanet.cuda()
        retinanet = torch.nn.DataParallel(retinanet).cuda()
        print("Model moved to GPU")
    else:
        retinanet = torch.nn.DataParallel(retinanet)
        print("Using CPU")
    
    optimizer = optim.Adam(retinanet.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.5)
    
    scaler = GradScaler() if args.use_amp else None
    
    best_map = 0.0
    best_epoch = 0
    
    train_log = []
    
    start_epoch = 0
    if args.continue_training and args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        try:
            checkpoint = torch.load(args.checkpoint)
            retinanet.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scaler and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_map = checkpoint.get('best_map', 0.0)
            best_epoch = checkpoint.get('best_epoch', 0)
            print(f"Resumed from epoch {start_epoch}, best mAP: {best_map:.4f} (epoch {best_epoch})")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            start_epoch = 0
    
    print(f"\n" + "="*60)
    print("Starting Training (Fixed Version)")
    print("="*60)
    
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        avg_epoch_loss = train_epoch(
            dataloader_train, retinanet, optimizer, epoch, 
            start_time, args.loss_threshold, scaler, args.use_amp
        )
        
        current_map = 0.0
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            current_map = evaluate_epoch(retinanet, dataset_val, epoch, args.save_dir)
            
            if args.eval_coco:
                print("Computing COCO-style mAP...")
                coco_save_folder = os.path.join(args.save_dir, f'coco_eval_epoch_{epoch}')
                os.makedirs(coco_save_folder, exist_ok=True)
                
                coco_aps = evaluate_coco_map(
                    generator=dataset_val,
                    retinanet=retinanet,
                    iou_threshold=0.5,
                    score_threshold=0.05,
                    max_detections=100,
                    save_detection=True,
                    save_folder=coco_save_folder,
                    load_detection=False,
                    save_path=coco_save_folder
                )
                
                coco_map = np.mean([np.mean(aps) for aps in coco_aps.values()])
                print(f"COCO-style mAP: {coco_map:.4f}")
            
            if current_map > best_map:
                best_map = current_map
                best_epoch = epoch
                print(f"🎉 New best mAP: {best_map:.4f} (epoch {best_epoch})")
                
                best_model_path = f'{args.save_dir}/best_model_fixed.pt'
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': retinanet.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_epoch_loss,
                    'map': current_map,
                    'best_map': best_map,
                    'best_epoch': best_epoch,
                    'config': {
                        'fusion': args.fusion,
                        'depth': args.depth,
                        'learning_rate': args.learning_rate,
                        'batch_size': args.batch_size,
                        'loss_threshold': args.loss_threshold
                    }
                }
                if scaler:
                    save_dict['scaler_state_dict'] = scaler.state_dict()
                
                torch.save(save_dict, best_model_path)
                print(f"Best model saved: {best_model_path}")
        
        train_log.append({
            'epoch': epoch,
            'loss': avg_epoch_loss,
            'map': current_map,
            'best_map': best_map,
            'best_epoch': best_epoch
        })
        
        if avg_epoch_loss != float('inf'):
            scheduler.step(avg_epoch_loss)
        
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            save_path = f'{args.save_dir}/dsec_retinanet_fixed_epoch_{epoch}.pt'
            save_dict = {
                'epoch': epoch,
                'model_state_dict': retinanet.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'map': current_map,
                'best_map': best_map,
                'best_epoch': best_epoch,
                'train_log': train_log,
                'config': {
                    'fusion': args.fusion,
                    'depth': args.depth,
                    'learning_rate': args.learning_rate,
                    'batch_size': args.batch_size,
                    'loss_threshold': args.loss_threshold
                }
            }
            if scaler:
                save_dict['scaler_state_dict'] = scaler.state_dict()
            
            torch.save(save_dict, save_path)
            print(f"Model saved: {save_path}")
    
    final_path = f'{args.save_dir}/dsec_retinanet_fixed_final.pt'
    final_dict = {
        'epoch': args.epochs,
        'model_state_dict': retinanet.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_log': train_log,
        'best_map': best_map,
        'best_epoch': best_epoch,
        'config': {
            'fusion': args.fusion,
            'depth': args.depth,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'loss_threshold': args.loss_threshold
        }
    }
    if scaler:
        final_dict['scaler_state_dict'] = scaler.state_dict()
    
    torch.save(final_dict, final_path)
    
    total_time = time_since(start_time)
    print(f"\n" + "="*60)
    print(f"Training completed! Time taken: {total_time}")
    print(f"Final model saved: {final_path}")
    print(f"Best mAP: {best_map:.4f} (epoch {best_epoch})")
    print("="*60)


if __name__ == '__main__':
    main()
