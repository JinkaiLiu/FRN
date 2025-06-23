import argparse
import collections
import numpy as np
import time
import math
import torch
import torch.optim as optim

from dsec_dataloader import create_dsec_det_dataloader
from retinanet import model

assert torch.__version__.split('.')[0] == '1'
print('CUDA available: {}'.format(torch.cuda.is_available()))


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m}m {s:.0f}s'


def debug_sample_data(sample_dict, sample_idx, verbose=False):
    if verbose:
        print(f"\n=== Debug Sample {sample_idx} ===")
    
    img_event = sample_dict['img']
    img_rgb = sample_dict['img_rgb'] 
    annot = sample_dict['annot']
    
    if verbose:
        print(f"Event shape: {img_event.shape}, RGB shape: {img_rgb.shape}")
        print(f"Event range: [{img_event.min():.3f}, {img_event.max():.3f}]")
        print(f"RGB range: [{img_rgb.min():.3f}, {img_rgb.max():.3f}]")
    
    if torch.isnan(img_event).any() or torch.isinf(img_event).any():
        print(f"❌ Sample {sample_idx}: Event data contains NaN/Inf!")
        return False
        
    if torch.isnan(img_rgb).any() or torch.isinf(img_rgb).any():
        print(f"❌ Sample {sample_idx}: RGB data contains NaN/Inf!")
        return False
    
    valid_annots = annot[annot[:, 0] != -1]
    if verbose:
        print(f"Valid annotations: {len(valid_annots)}")
    
    if len(valid_annots) > 0:
        img_h, img_w = img_rgb.shape[-2], img_rgb.shape[-1]
        
        invalid_boxes = (
            (valid_annots[:, 2] <= valid_annots[:, 0]) |  
            (valid_annots[:, 3] <= valid_annots[:, 1]) |  
            (valid_annots[:, 0] < 0) | (valid_annots[:, 1] < 0) |  
            (valid_annots[:, 2] > img_w) | (valid_annots[:, 3] > img_h) |  
            ((valid_annots[:, 2] - valid_annots[:, 0]) < 1) |  
            ((valid_annots[:, 3] - valid_annots[:, 1]) < 1)   
        )
        
        if invalid_boxes.any():
            print(f"❌ Sample {sample_idx}: Found {invalid_boxes.sum()} invalid bounding boxes!")
            if verbose:
                invalid_annots = valid_annots[invalid_boxes]
                for i, box in enumerate(invalid_annots):
                    print(f"  Box {i}: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}] class={box[4]}")
            return False
    
    if verbose:
        print("✅ Sample data looks valid")
    return True


def safe_training_step(retinanet, data, optimizer, iter_num, loss_threshold=50.0):
    try:
        img_rgb = data['img_rgb'].cuda().float()
        img_event = data['img'].cuda().float()
        annot = data['annot'].cuda().float()
        
        if not debug_sample_data({'img': img_event[0], 'img_rgb': img_rgb[0], 'annot': annot[0]}, iter_num, verbose=False):
            return None, None, None
        
        classification_loss, regression_loss = retinanet([img_rgb, img_event, annot])
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()
        total_loss = classification_loss + regression_loss
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: NaN/Inf loss at iteration {iter_num}")
            return None, None, None
            
        if total_loss.item() > loss_threshold:
            print(f"Warning: Loss is {total_loss.item()}, exceeds threshold {loss_threshold}, skipping iteration")
            return None, None, None
        
        optimizer.zero_grad()
        total_loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 1.0)
        if grad_norm > 10.0:
            print(f"Warning: Large gradient norm {grad_norm:.3f} at iteration {iter_num}")
            return None, None, None
        
        optimizer.step()
        
        return classification_loss.item(), regression_loss.item(), total_loss.item()
        
    except Exception as e:
        print(f"Error in training step {iter_num}: {e}")
        return None, None, None


def robust_training_loop(dataloader_train, retinanet, optimizer, scheduler, epoch_num, start_time, loss_threshold=50.0):
    loss_hist = collections.deque(maxlen=100)
    epoch_losses = []
    valid_iterations = 0
    total_iterations = 0
    
    retinanet.train()
    if hasattr(retinanet.module, 'freeze_bn'):
        retinanet.module.freeze_bn()
    
    for iter_num, data in enumerate(dataloader_train):
        total_iterations += 1
        
        cls_loss, reg_loss, total_loss = safe_training_step(
            retinanet, data, optimizer, iter_num, loss_threshold=loss_threshold
        )
        
        if total_loss is not None:
            valid_iterations += 1
            loss_hist.append(total_loss)
            
            if iter_num % 20 == 0:
                avg_loss = np.mean(loss_hist) if loss_hist else 0
                print(f'[DSEC-Det fpn_fusion] [{time_since(start_time)}], Epoch: {epoch_num} | '
                      f'Iteration: {iter_num} | Classification loss: {cls_loss:.5f} | '
                      f'Regression loss: {reg_loss:.5f} | Running loss: {avg_loss:.5f} | '
                      f'Valid rate: {valid_iterations}/{total_iterations} ({100*valid_iterations/total_iterations:.1f}%)')
                epoch_losses.append(avg_loss)
        
        if iter_num > 0 and iter_num % 100 == 0:
            torch.cuda.empty_cache()
    
    if len(epoch_losses) > 0:
        avg_epoch_loss = np.mean(epoch_losses)
        scheduler.step(avg_epoch_loss)
        print(f"Epoch {epoch_num} completed. Valid iterations: {valid_iterations}/{total_iterations}, Average loss: {avg_epoch_loss:.5f}")
        return avg_epoch_loss
    else:
        print(f"Epoch {epoch_num} completed with no valid losses!")
        return float('inf')


def main(args=None):
    base_dir = '/media/data/hucao/zhenwu/hucao/DSEC/DSEC_Det'
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    
    parser.add_argument('--dataset_name', default='dsec', help='dsec or ddd17')
    parser.add_argument('--dataset', default='csv', help='Dataset type, must be one of csv or coco.') 
    parser.add_argument('--root_dir', default=base_dir, help='Root directory of DSEC dataset')
    parser.add_argument('--fusion', help='fpn_fusion, rgb, event', type=str, default='fpn_fusion')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50', type=int, default=50) 
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=60) 
    parser.add_argument('--continue_training', help='load a pretrained file', action='store_true') 
    parser.add_argument('--checkpoint', help='location of pretrained file', default='') 
    parser.add_argument('--batch_size', help='batch size for training', type=int, default=1)
    parser.add_argument('--learning_rate', help='learning rate', type=float, default=1e-5)
    parser.add_argument('--debug_data', help='enable data debugging', action='store_true')
    parser.add_argument('--loss_threshold', help='loss threshold for filtering', type=float, default=50.0)
    parser.add_argument('--save_dir', help='directory to save checkpoints', 
                       default='/media/data/hucao/zehua/results_dsec/cross_4layer')
    parser.add_argument('--num_workers', help='number of data loading workers', type=int, default=6)

    parser = parser.parse_args(args)

    print(f"Starting DSEC training with parameters:")
    print(f"  Root directory: {parser.root_dir}")
    print(f"  Fusion model: {parser.fusion}")
    print(f"  Batch size: {parser.batch_size}")
    print(f"  Learning rate: {parser.learning_rate}")
    print(f"  Loss threshold: {parser.loss_threshold}")
    print(f"  Epochs: {parser.epochs}")

    dataloader_train, dataset_train = create_dsec_det_dataloader(
        root_dir=parser.root_dir,
        split='train',
        batch_size=parser.batch_size,
        num_workers=0 if parser.debug_data else parser.num_workers,
        shuffle=True,
        event_representation='time_surface',
        dt=50,
        image_height=480,
        image_width=640,
        augment=True,
        normalize_events=True,
        normalize_images=True,
        use_downsampled_events=True
    )
    
    if parser.debug_data:
        print("=== Data Debugging Mode ===")
        for batch_idx, data in enumerate(dataloader_train):
            debug_sample_data(data, batch_idx, verbose=True)
            if batch_idx >= 2:
                break
        return
    
    dataloader_val, dataset_val = create_dsec_det_dataloader(
        root_dir=parser.root_dir,
        split='test',
        batch_size=1,
        num_workers=parser.num_workers,
        shuffle=False,
        event_representation='time_surface',
        dt=50,
        image_height=480,
        image_width=640,
        augment=False,
        normalize_events=True,
        normalize_images=True,
        use_downsampled_events=True
    )
        
    list_models = ['fpn_fusion', 'event', 'rgb']
    if parser.fusion in list_models:
        if parser.depth == 50:
            retinanet = model.resnet50(
                dataset_name=parser.dataset_name, 
                num_classes=dataset_train.num_classes(),
                fusion_model=parser.fusion,
                pretrained=False
            )
    else:
        raise ValueError('Unsupported model fusion')

    use_gpu = True
    epoch_loss_all = []
    epoch_total = 0
    
    if parser.continue_training and parser.checkpoint:
        print(f"Loading checkpoint from: {parser.checkpoint}")
        checkpoint = torch.load(parser.checkpoint)
        retinanet.load_state_dict(checkpoint['model_state_dict'])
        epoch_loss_all = checkpoint.get('loss', [])
        epoch_total = checkpoint.get('epoch', 0)
        print(f'Loaded pretrained model from epoch {epoch_total}')
        
    if use_gpu and torch.cuda.is_available():
        retinanet = retinanet.cuda()
        retinanet = torch.nn.DataParallel(retinanet).cuda() 
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True 

    optimizer = optim.Adam(retinanet.parameters(), lr=parser.learning_rate)
    
    if parser.continue_training and parser.checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state")
        except:
            print("Could not load optimizer state, using fresh optimizer")
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.5) 

    print('Num training images: {}'.format(len(dataset_train)))
    start = time.time()

    for epoch_num in range(parser.epochs):
        epoch_total += 1 
        
        print(f"\n{'='*60}")
        print(f"Starting Epoch {epoch_num+1}/{parser.epochs} (Total: {epoch_total})")
        print(f"{'='*60}")
        
        avg_epoch_loss = robust_training_loop(
            dataloader_train, retinanet, optimizer, scheduler, 
            epoch_num, start, loss_threshold=parser.loss_threshold
        )

        if epoch_num % 5 == 0 or epoch_num == parser.epochs - 1:
            save_path = f'{parser.save_dir}/dsec_fpn_retinanet_{epoch_total}.pt'
            torch.save({
                'epoch': epoch_total, 
                'model_state_dict': retinanet.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss_all + [avg_epoch_loss],
                'config': {
                    'dataset_name': parser.dataset_name,
                    'fusion': parser.fusion,
                    'depth': parser.depth,
                    'learning_rate': parser.learning_rate,
                    'batch_size': parser.batch_size,
                    'loss_threshold': parser.loss_threshold
                }
            }, save_path)
            print(f"Model saved to {save_path}")

    final_save_path = f'{parser.save_dir}/dsec_fpn_retinanet_final_{epoch_total}.pt'
    torch.save({
        'epoch': epoch_total, 
        'model_state_dict': retinanet.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss_all,
        'config': {
            'dataset_name': parser.dataset_name,
            'fusion': parser.fusion,
            'depth': parser.depth,
            'learning_rate': parser.learning_rate,
            'batch_size': parser.batch_size,
            'loss_threshold': parser.loss_threshold
        }
    }, final_save_path)
    print(f"Final model saved to {final_save_path}")
    print(f"Training completed in {time_since(start)}")


if __name__ == '__main__':
    main()
