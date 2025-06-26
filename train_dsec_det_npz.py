import argparse
import collections
import numpy as np
import time
import math
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import os

from retinanet.dataloader_dsec_det_npz import create_dsec_det_dataloader
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


def debug_batch_data(data_batch, batch_idx):
    """
    Debug batch data format and content
    """
    print(f"\n=== Debug Batch {batch_idx} ===")
    
    img_event = data_batch['img']      # [batch_size, channels, H, W]
    img_rgb = data_batch['img_rgb']    # [batch_size, H, W, channels] or [batch_size, channels, H, W]
    annot = data_batch['annot']        # [batch_size, max_annots_in_batch, 5]
    
    batch_size = img_event.shape[0]
    print(f"Batch size: {batch_size}")
    print(f"Event shape: {img_event.shape}, dtype: {img_event.dtype}")
    print(f"RGB shape: {img_rgb.shape}, dtype: {img_rgb.dtype}")
    print(f"Annot shape: {annot.shape}, dtype: {annot.dtype}")
    
    # 检查第一个样本
    sample_event = img_event[0]
    sample_rgb = img_rgb[0]
    sample_annot = annot[0]
    
    print(f"First sample - Event: {sample_event.shape}, RGB: {sample_rgb.shape}, Annot: {sample_annot.shape}")
    print(f"Event range: [{sample_event.min():.3f}, {sample_event.max():.3f}]")
    print(f"RGB range: [{sample_rgb.min():.3f}, {sample_rgb.max():.3f}]")
    
    # 数据质量检查
    issues = []
    
    # 检查全0数据
    if sample_event.max() == 0 and sample_event.min() == 0:
        issues.append("Event data all zeros")
    if sample_rgb.max() == 0 and sample_rgb.min() == 0:
        issues.append("RGB data all zeros")
    
    # 检查NaN/Inf
    if torch.isnan(sample_event).any() or torch.isinf(sample_event).any():
        issues.append("Event data has NaN/Inf")
    if torch.isnan(sample_rgb).any() or torch.isinf(sample_rgb).any():
        issues.append("RGB data has NaN/Inf")
    
    # 检查数据范围
    if sample_event.min() < -2 or sample_event.max() > 2:
        issues.append(f"Event range unusual: [{sample_event.min():.3f}, {sample_event.max():.3f}]")
    if sample_rgb.min() < -10 or sample_rgb.max() > 10:
        issues.append(f"RGB range unusual: [{sample_rgb.min():.3f}, {sample_rgb.max():.3f}]")
    
    # 检查标注
    max_annots = sample_annot.shape[0]
    
    # 用torch.where避免boolean indexing问题
    valid_indices = torch.where(sample_annot[:, 0] != -1)[0]
    num_valid = len(valid_indices)
    
    print(f"Annotations: {num_valid}/{max_annots} valid")
    
    if num_valid > 0:
        print("Valid annotation examples:")
        for i, ann_idx in enumerate(valid_indices[:3]):  # 显示前3个
            ann = sample_annot[ann_idx]
            x1, y1, x2, y2, cls = ann[0].item(), ann[1].item(), ann[2].item(), ann[3].item(), ann[4].item()
            w, h = x2 - x1, y2 - y1
            print(f"  Ann {i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] w={w:.1f} h={h:.1f} cls={int(cls)}")
            
            # 基本有效性检查
            ann_issues = []
            if x2 <= x1: ann_issues.append("x2<=x1")
            if y2 <= y1: ann_issues.append("y2<=y1")
            if x1 < -5 or y1 < -5: ann_issues.append("negative_coords")
            if x2 > 650 or y2 > 490: ann_issues.append("out_of_bounds")
            if w < 0.5 or h < 0.5: ann_issues.append("too_small")
            if cls < 0 or cls >= 8: ann_issues.append("invalid_class")
            
            if ann_issues:
                print(f"    Issues: {', '.join(ann_issues)}")
                issues.append(f"Annotation {i} has issues")
    
    # 总结
    if issues:
        print(f"❌ Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Batch data looks good")
        return True


def safe_training_step(retinanet, data, optimizer, iter_num, loss_threshold=50.0, 
                      scaler=None, use_amp=False):
    """
    Training step with error handling
    """
    try:
        # non_blocking GPU transfer
        img_rgb = data['img_rgb'].cuda(non_blocking=True).float()
        img_event = data['img'].cuda(non_blocking=True).float()
        annot = data['annot'].cuda(non_blocking=True).float()
        
        # data validation
        if torch.isnan(img_event).any() or torch.isinf(img_event).any():
            print(f"Iter {iter_num}: NaN/Inf in event data")
            return None, None, None
        if torch.isnan(img_rgb).any() or torch.isinf(img_rgb).any():
            print(f"Iter {iter_num}: NaN/Inf in RGB data")
            return None, None, None
        
        # check for empty data
        if img_event.max() == 0 or img_rgb.max() == 0:
            print(f"Iter {iter_num}: Zero data detected")
            return None, None, None
        
        optimizer.zero_grad()
        
        # mixed precision forward pass
        if use_amp and scaler is not None:
            with autocast():
                classification_loss, regression_loss = retinanet([img_rgb, img_event, annot])
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                total_loss = classification_loss + regression_loss
        else:
            # standard forward pass
            classification_loss, regression_loss = retinanet([img_rgb, img_event, annot])
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            total_loss = classification_loss + regression_loss
        
        # loss validation
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Iter {iter_num}: NaN/Inf loss")
            return None, None, None
            
        if total_loss.item() > loss_threshold:
            print(f"Iter {iter_num}: Loss {total_loss.item():.1f} > threshold {loss_threshold}")
        
        # mixed precision backward pass
        if use_amp and scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # standard backward pass
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 1.0)
            optimizer.step()
        
        return classification_loss.item(), regression_loss.item(), total_loss.item()
        
    except Exception as e:
        print(f"Iter {iter_num}: Training error - {e}")
        return None, None, None


def train_epoch(dataloader, retinanet, optimizer, epoch_num, start_time, loss_threshold=50.0,
               scaler=None, use_amp=False):
    """
    Train for one epoch
    """
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
            
            # periodic logging
            if iter_num % 10 == 0:
                avg_loss = np.mean(loss_hist) if loss_hist else 0
                valid_rate = 100 * valid_iterations / total_iterations
                print(f'[{time_since(start_time)}] Epoch {epoch_num} | Iter {iter_num} | '
                      f'Loss: cls={cls_loss:.4f} reg={reg_loss:.4f} total={total_loss:.4f} | '
                      f'Avg: {avg_loss:.4f} | Valid: {valid_iterations}/{total_iterations} ({valid_rate:.1f}%)')
                epoch_losses.append(avg_loss)
        
        # memory cleanup
        if iter_num > 0 and iter_num % 100 == 0:
            torch.cuda.empty_cache()
    
    # epoch summary
    if epoch_losses:
        avg_epoch_loss = np.mean(epoch_losses)
        valid_rate = 100 * valid_iterations / total_iterations
        print(f"Epoch {epoch_num} summary: {valid_iterations}/{total_iterations} valid ({valid_rate:.1f}%), avg loss: {avg_epoch_loss:.4f}")
        return avg_epoch_loss
    else:
        print(f"Epoch {epoch_num}: No valid iterations!")
        return float('inf')


def evaluate_epoch(retinanet, dataset_val, epoch_num, save_folder):
    """
    Evaluate model performance using mAP
    """
    print(f"\n{'='*20} Validation Epoch {epoch_num} {'='*20}")
    
    eval_save_folder = os.path.join(save_folder, f'eval_epoch_{epoch_num}')
    os.makedirs(eval_save_folder, exist_ok=True)
    
    # use evaluation function
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
    parser = argparse.ArgumentParser(description='DSEC Detection Training Script with mAP Evaluation')
    
    # 数据参数
    parser.add_argument('--root_dir', default='/media/data/hucao/zhenwu/hucao/DSEC/DSEC_Det', 
                       help='DSEC dataset root directory')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Data loading workers')
    
    # 训练参数  
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--loss_threshold', type=float, default=50.0, help='Loss threshold for filtering')
    
    # 模型参数
    parser.add_argument('--fusion', default='fpn_fusion', choices=['fpn_fusion', 'rgb', 'event'],
                       help='Fusion model type')
    parser.add_argument('--depth', type=int, default=50, choices=[18, 34, 50], 
                       help='ResNet depth')
    
    # 评估参数
    parser.add_argument('--eval_interval', type=int, default=5, help='Evaluation interval (epochs)')
    parser.add_argument('--eval_coco', action='store_true', help='Also evaluate COCO-style mAP')
    
    # 优化参数
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    
    # 其他参数
    parser.add_argument('--debug_data', action='store_true', help='Enable data debugging')
    parser.add_argument('--continue_training', action='store_true', help='Continue from checkpoint')
    parser.add_argument('--checkpoint', default='', help='Checkpoint path')
    parser.add_argument('--save_dir', default='/media/data/hucao/zehua/results_dsec/cross_4layer',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*60)
    print("DSEC Detection Training with mAP Evaluation")
    print("="*60)
    print(f"Root directory: {args.root_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num workers: {args.num_workers}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Loss threshold: {args.loss_threshold}")
    print(f"Epochs: {args.epochs}")
    print(f"Fusion model: {args.fusion}")
    print(f"Mixed precision: {args.use_amp}")
    print(f"Evaluation interval: {args.eval_interval}")
    print(f"COCO evaluation: {args.eval_coco}")
    
    # load training data
    print("\nLoading training dataset...")
    dataloader_train, dataset_train = create_dsec_det_dataloader(
        root_dir=args.root_dir,
        split='train',
        batch_size=args.batch_size,
        num_workers=0 if args.debug_data else args.num_workers,
        shuffle=True,
        event_representation='time_surface',
        dt=50,
        image_height=480,
        image_width=640,
        augment=True,
        normalize_events=True,
        normalize_images=True,
        use_downsampled_events=True,
        cache_events=True
    )
    
    print(f"Training dataset loaded: {len(dataset_train)} samples")
    
    # load validation data (single samples for evaluation function)
    print("Loading validation dataset...")
    _, dataset_val = create_dsec_det_dataloader(
        root_dir=args.root_dir,
        split='val',  
        batch_size=1,  # validation uses batch_size=1
        num_workers=1,
        shuffle=False,
        event_representation='time_surface',
        dt=50,
        image_height=480,
        image_width=640,
        augment=False,  # no data augmentation for validation
        normalize_events=True,
        normalize_images=True,
        use_downsampled_events=True,
        cache_events=True
    )
    
    print(f"Validation dataset loaded: {len(dataset_val)} samples")
    
    # Debug模式
    if args.debug_data:
        print("\n" + "="*40)
        print("DATA DEBUGGING MODE")
        print("="*40)
        
        for batch_idx, data in enumerate(dataloader_train):
            success = debug_batch_data(data, batch_idx)
            if not success:
                print(f"❌ Batch {batch_idx} has issues!")
            if batch_idx >= 4:  # 检查前5个batch
                break
        
        print("\nDebug completed. Use without --debug_data to start training.")
        return
    
    # create model
    print("\nCreating model...")
    retinanet = model.resnet50(
        dataset_name='dsec',
        num_classes=dataset_train.num_classes(),
        fusion_model=args.fusion,
        pretrained=False
    )
    print(f"Model created: ResNet{args.depth} with {args.fusion} fusion")
    print(f"Number of classes: {dataset_train.num_classes()}")
    
    # GPU setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        retinanet = retinanet.cuda()
        retinanet = torch.nn.DataParallel(retinanet).cuda()
        print("Model moved to GPU")
    else:
        retinanet = torch.nn.DataParallel(retinanet)
        print("Using CPU")
    
    # optimizer and scheduler
    optimizer = optim.Adam(retinanet.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.5)
    
    # mixed precision scaler
    scaler = GradScaler() if args.use_amp else None
    
    # track best performance
    best_map = 0.0
    best_epoch = 0
    
    # training log
    train_log = []
    
    # load checkpoint
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
    
    # start training
    print(f"\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # training
        avg_epoch_loss = train_epoch(
            dataloader_train, retinanet, optimizer, epoch, 
            start_time, args.loss_threshold, scaler, args.use_amp
        )
        
        # validation
        current_map = 0.0
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            current_map = evaluate_epoch(retinanet, dataset_val, epoch, args.save_dir)
            
            # optional COCO evaluation
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
                
                # compute COCO-style average mAP
                coco_map = np.mean([np.mean(aps) for aps in coco_aps.values()])
                print(f"COCO-style mAP: {coco_map:.4f}")
            
            # update best results
            if current_map > best_map:
                best_map = current_map
                best_epoch = epoch
                print(f"🎉 New best mAP: {best_map:.4f} (Epoch {best_epoch})")
                
                # save best model
                best_model_path = f'{args.save_dir}/best_model.pt'
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
        
        # log training progress
        train_log.append({
            'epoch': epoch,
            'loss': avg_epoch_loss,
            'map': current_map,
            'best_map': best_map,
            'best_epoch': best_epoch
        })
        
        # adjust learning rate
        if avg_epoch_loss != float('inf'):
            scheduler.step(avg_epoch_loss)
        
        # save model periodically
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            save_path = f'{args.save_dir}/dsec_retinanet_epoch_{epoch}.pt'
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
    
    # final save
    final_path = f'{args.save_dir}/dsec_retinanet_final.pt'
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
    print(f"TRAINING COMPLETED IN {total_time}")
    print(f"Final model saved: {final_path}")
    print(f"Best mAP: {best_map:.4f} (Epoch {best_epoch})")
    print("="*60)


if __name__ == '__main__':
    main()
