import argparse
import collections
import numpy as np
import time
import math
import torch
import torch.optim as optim

from retinanet import model
from dsec_dataloader import create_dsec_det_dataloader
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1' 

print('CUDA available: {}'.format(torch.cuda.is_available()))

def time_since(since):
    now = time.time() 
    s = now - since 
    m = math.floor(s / 60)  
    s -= m * 60
    return '%dm %ds' % (m, s)

def main(args=None):
    base_dir = '/media/data/hucao/zhenwu/hucao/DSEC/DSEC_Det'
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    
    parser.add_argument('--dataset_name', default='dsec', help='dsec dataset name')
    parser.add_argument('--dataset_root', default=base_dir, help='Path to DSEC-Det dataset root directory')
    parser.add_argument('--split_config', default='./retinanet/dsec_split.yaml', help='Path to split configuration file')
    parser.add_argument('--event_representation', default='time_surface', help='Event representation: time_surface, event_count, binary')
    parser.add_argument('--dt', type=int, default=50, help='Event time window in milliseconds')
    parser.add_argument('--image_height', type=int, default=480, help='Target image height')
    parser.add_argument('--image_width', type=int, default=640, help='Target image width')
    parser.add_argument('--use_downsampled_events', action='store_true', help='Use downsampled events (events_2x.h5)')
    parser.add_argument('--fusion', help='fpn_fusion, rgb, event', type=str, default='fpn_fusion')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50', type=int, default=50) 
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=60) 
    parser.add_argument('--batch_size', help='Batch size', type=int, default=2)
    parser.add_argument('--num_workers', help='Number of workers', type=int, default=4)
    parser.add_argument('--continue_training', help='load a pretrained file', default=True) 
    parser.add_argument('--checkpoint', help='location of pretrained file', 
                       default='/media/data/hucao/zehua/results_dsec/cross_4layer/csv_fpn_homographic_retinanet_retinanet101_53.pt') 
    parser.add_argument('--save_dir', help='Directory to save checkpoints', 
                       default='/media/data/hucao/zehua/results_dsec/cross_4layer')

    parser = parser.parse_args(args)

    print("Creating DSEC-Det datasets...")
    
    dataloader_train, dataset_train = create_dsec_det_dataloader(
        root_dir=parser.dataset_root,
        split='train',
        batch_size=parser.batch_size,
        num_workers=parser.num_workers,
        shuffle=True,
        event_representation=parser.event_representation,
        dt=parser.dt,
        image_height=parser.image_height,
        image_width=parser.image_width,
        augment=True,
        debug=False,
        split_config_path=parser.split_config,
        normalize_events=True,
        normalize_images=True,
        use_downsampled_events=parser.use_downsampled_events,
        use_aspect_ratio_sampler=True
    )
    
    try:
        dataloader_val, dataset_val = create_dsec_det_dataloader(
            root_dir=parser.dataset_root,
            split='val',
            batch_size=1,
            num_workers=parser.num_workers,
            shuffle=False,
            event_representation=parser.event_representation,
            dt=parser.dt,
            image_height=parser.image_height,
            image_width=parser.image_width,
            augment=False,
            debug=False,
            split_config_path=parser.split_config,
            normalize_events=True,
            normalize_images=True,
            use_downsampled_events=parser.use_downsampled_events
        )
        print(f"Validation dataset: {len(dataset_val)} samples")
    except Exception as e:
        print(f"Warning: Could not create validation dataset: {e}")
        dataloader_val, dataset_val = None, None
    
    dataloader_test, dataset_test = create_dsec_det_dataloader(
        root_dir=parser.dataset_root,
        split='test',
        batch_size=1,
        num_workers=parser.num_workers,
        shuffle=False,
        event_representation=parser.event_representation,
        dt=parser.dt,
        image_height=parser.image_height,
        image_width=parser.image_width,
        augment=False,
        debug=False,
        split_config_path=parser.split_config,
        normalize_events=True,
        normalize_images=True,
        use_downsampled_events=parser.use_downsampled_events
    )

    print(f"Training dataset: {len(dataset_train)} samples")
    print(f"Test dataset: {len(dataset_test)} samples")
    print(f"Number of classes: {dataset_train.num_classes()}")
    print(f"Classes: {list(dataset_train.classes.keys())}")

    list_models = ['fpn_fusion', 'event', 'rgb']
    if parser.fusion in list_models:
        if parser.depth == 50:
            retinanet = model.resnet50(dataset_name=parser.dataset_name, 
                                     num_classes=dataset_train.num_classes(),
                                     fusion_model=parser.fusion,
                                     pretrained=False)
    else:
        raise ValueError('Unsupported model fusion')

    use_gpu = True
    if parser.continue_training:
        checkpoint = torch.load(parser.checkpoint)
        retinanet.load_state_dict(checkpoint['model_state_dict'])
        epoch_loss_all = checkpoint['loss']
        epoch_total = checkpoint['epoch']
        print(f'Continuing training from epoch {epoch_total}')
        retinanet.eval()
    else:
        epoch_total = 0
        epoch_loss_all = []
        
    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda() 
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True 

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True) 
    loss_hist = collections.deque(maxlen=100) 

    retinanet.train()
    retinanet.module.freeze_bn() 

    num_batches = 0
    start = time.time()
    
    print(f'Starting training with {parser.fusion} fusion...')
    print(time_since(start))
    
    epoch_loss = []
    for epoch_num in range(parser.epochs):
        retinanet.train()
        retinanet.module.freeze_bn()
        epoch_total += 1 
        
        for iter_num, data in enumerate(dataloader_train):
            try:
                classification_loss, regression_loss = retinanet([
                    data['img_rgb'].cuda().float(),
                    data['img'].cuda().float(),
                    data['annot'].cuda()
                ])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue 

                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1) 

                num_batches += 1
                if num_batches == 2:
                    optimizer.step() 
                    optimizer.zero_grad() 
                    num_batches = 0

                loss_hist.append(float(loss)) 

                if iter_num % 50 == 0:
                    print(
                        '[DSEC-Det {}] [{}], Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            parser.fusion, time_since(start), epoch_num, iter_num, 
                            float(classification_loss), float(regression_loss), np.mean(loss_hist)))
                    epoch_loss.append(np.mean(loss_hist))

                del classification_loss
                del regression_loss
                
            except Exception as e: 
                print(f"Error in training iteration: {e}")
                continue

        if dataset_val is not None:
            print('Evaluating on validation dataset...')
            try:
                mAP = csv_eval.evaluate(dataset_val, retinanet)
                print(f'Validation mAP: {mAP}')
            except Exception as e:
                print(f"Validation evaluation failed: {e}")

        scheduler.step(np.mean(epoch_loss))

        if epoch_num % 5 == 0:
            checkpoint_path = f'{parser.save_dir}/{parser.dataset_name}_fpn_{parser.fusion}_retinanet_{epoch_total}.pt'
            torch.save({
                'epoch': epoch_total, 
                'model_state_dict': retinanet.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': np.append(epoch_loss_all, epoch_loss)
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')

    final_checkpoint_path = f'{parser.save_dir}/{parser.dataset_name}_fpn_{parser.fusion}_retinanet_final_{epoch_total}.pt'
    torch.save({
        'epoch': epoch_total, 
        'model_state_dict': retinanet.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': np.append(epoch_loss_all, epoch_loss)
    }, final_checkpoint_path)
    print(f'Final checkpoint saved: {final_checkpoint_path}')

if __name__ == '__main__':
    main()