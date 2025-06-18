import argparse
import collections
import numpy as np
import time
import math
import os
import torch
import torch.optim as optim
import pickle

from retinanet import model
from retinanet.dataloader_dsec_det import create_dsec_det_dataloader
from retinanet import csv_eval_dsec_det

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def load_model(fusion_type, num_classes, checkpoint_path):
    """Load model from checkpoint"""
    list_models = ['fpn_fusion', 'event', 'rgb']
    if fusion_type in list_models:
        retinanet = model.resnet50(num_classes=num_classes, fusion_model=fusion_type, pretrained=False)
    else:
        raise ValueError('Unsupported model fusion')

    checkpoint = torch.load(checkpoint_path)
    retinanet.load_state_dict(checkpoint['model_state_dict'])
    return retinanet, checkpoint['epoch'], checkpoint['loss']

def setup_arg_parser():
    base_dir = '/media/data/hucao/zhenwu/hucao/DSEC/DSEC_Det'
    parser = argparse.ArgumentParser(description='Test script for DSEC-Det RetinaNet.')
    
    parser.add_argument('--dataset_name', default='dsec', help='dsec dataset name')
    parser.add_argument('--dataset_root', default=base_dir, help='Path to DSEC-Det dataset root directory')
    parser.add_argument('--split_config', default='./retinanet/dsec_split.yaml', help='Path to split configuration file')
    parser.add_argument('--event_representation', default='time_surface', help='Event representation: time_surface, event_count, binary')
    parser.add_argument('--dt', type=int, default=50, help='Event time window in milliseconds')
    parser.add_argument('--image_height', type=int, default=480, help='Target image height')
    parser.add_argument('--image_width', type=int, default=640, help='Target image width')
    parser.add_argument('--use_downsampled_events', action='store_true', help='Use downsampled events (events_2x.h5)')
    parser.add_argument('--fusion', help='fpn_fusion, rgb, event', type=str, default='fpn_fusion')
    parser.add_argument('--checkpoint', help='location of pretrained file', default='../best_epoch.pt')
    parser.add_argument('--batch_size', help='Batch size for evaluation', type=int, default=1)
    parser.add_argument('--num_workers', help='Number of workers', type=int, default=4)
    parser.add_argument('--eval_corruption', help='evaluate on corrupted images', action='store_true')
    parser.add_argument('--corruption_group', help='corruption group number', type=int, default=0)
    parser.add_argument('--corruption_root', help='root directory for corrupted images', 
                       default='/media/data/hucao/zhenwu/hucao/DSEC/corruptions')
    parser.add_argument('--save_results', help='save detection results', action='store_true')
    parser.add_argument('--results_dir', help='directory to save results', 
                       default='/media/data/hucao/zehua/results_dsec/cross_4layer')

    return parser

def create_test_dataloader(args, corruption_type=None, severity=None):
    """Create test dataloader, optionally with corruption"""
    
    if corruption_type and severity:
        dataset_root = os.path.join(args.corruption_root, corruption_type, f'severity_{severity}')
        print(f"Testing on corruption: {corruption_type}, severity: {severity}")
    else:
        dataset_root = args.dataset_root
    
    dataloader, dataset = create_dsec_det_dataloader(
        root_dir=dataset_root,
        split='test',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        event_representation=args.event_representation,
        dt=args.dt,
        image_height=args.image_height,
        image_width=args.image_width,
        augment=False,
        debug=False,
        split_config_path=args.split_config,
        normalize_events=True,
        normalize_images=True,
        use_downsampled_events=args.use_downsampled_events
    )
    
    return dataloader, dataset

def main(args=None):
    parser = setup_arg_parser()
    args = parser.parse_args(args)
    
    print(f"Testing {args.fusion} model on DSEC-Det dataset")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Event representation: {args.event_representation}")
    print(f"Time window: {args.dt}ms")
    
    try:
        train_loader, train_dataset = create_dsec_det_dataloader(
            root_dir=args.dataset_root,
            split='train',
            batch_size=1,
            num_workers=0,
            shuffle=False,
            split_config_path=args.split_config,
            event_representation=args.event_representation,
            dt=args.dt,
            image_height=args.image_height,
            image_width=args.image_width,
            use_downsampled_events=args.use_downsampled_events
        )
        num_classes = train_dataset.num_classes()
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {list(train_dataset.classes.keys())}")
    except Exception as e:
        print(f"Warning: Could not load train dataset for class info: {e}")
        num_classes = 8  # Default for DSEC-Det
    
    retinanet, epoch_total, epoch_loss_all = load_model(args.fusion, num_classes, args.checkpoint)
    
    print(f'Testing {args.fusion} model from epoch {epoch_total}')
    retinanet.eval()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.module.freeze_bn()
    
    corruption_types = [
        ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur'],
        ['motion_blur', 'zoom_blur', 'fog', 'snow', 'frost'],
        ['brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    ]

    if args.eval_corruption:
        corruption_list = corruption_types[args.corruption_group]
        severity_list = [1, 2, 3, 4, 5]
        print(f"Evaluating corruptions: {corruption_list}")
        
        for corruption in corruption_list:
            Average_precisions = {'pedestrian': [], 'rider': [], 'car': [], 'bus': [], 
                                'truck': [], 'bicycle': [], 'motorcycle': [], 'train': []}
            start_c = time.time()
            
            for severity in severity_list:
                save_detect_folder = os.path.join(
                    args.results_dir, f'{args.fusion}_{args.event_representation}', 
                    corruption, f'severity_{severity}'
                )
                os.makedirs(save_detect_folder, exist_ok=True)
                
                try:
                    test_loader, test_dataset = create_test_dataloader(args, corruption, severity)
                    
                    start = time.time()
                    mAP = csv_eval_dsec_det.evaluate_coco_map(
                        test_dataset, retinanet, 
                        save_detection=args.save_results,
                        save_folder=save_detect_folder,
                        load_detection=False
                    )
                    
                    class_names = list(test_dataset.classes.keys())
                    for i, class_name in enumerate(class_names):
                        if i < len(mAP):
                            Average_precisions[class_name].append(mAP[i])
                    
                    print(f"Corruption: {corruption}, Severity: {severity}, Time: {time_since(start)}")
                    
                except Exception as e:
                    print(f"Error evaluating {corruption} severity {severity}: {e}")
                    continue
            
            print(f'{args.fusion}, {corruption}')
            for class_name in Average_precisions:
                if Average_precisions[class_name]:
                    avg_ap = np.mean(Average_precisions[class_name])
                    print(f'{class_name}: {avg_ap:.3f}')
            
            print(f'Time for corruption: {time_since(start_c)}')
            
            ap_file = os.path.join(save_detect_folder, f'{corruption}_ap.pkl')
            with open(ap_file, "wb") as fp:
                pickle.dump(Average_precisions, fp)

    else:
        Average_precisions = {'pedestrian': [], 'rider': [], 'car': [], 'bus': [], 
                            'truck': [], 'bicycle': [], 'motorcycle': [], 'train': []}
        
        test_loader, test_dataset = create_test_dataloader(args)
        print(f"Test dataset: {len(test_dataset)} samples")
        
        start = time.time()
        save_detect_folder = os.path.join(
            args.results_dir, f'{args.fusion}_{args.event_representation}', 'evaluation'
        )
        os.makedirs(save_detect_folder, exist_ok=True)
        
        try:
            mAP = csv_eval_dsec_det.evaluate_coco_map(
                test_dataset, retinanet,
                save_detection=args.save_results,
                save_folder=save_detect_folder,
                load_detection=False
            )
            
            class_names = list(test_dataset.classes.keys())
            for i, class_name in enumerate(class_names):
                if i < len(mAP):
                    Average_precisions[class_name].append(mAP[i])
            
            test_time = time.time() - start
            fps = len(test_dataset) / test_time
            print(f'FPS: {fps:.2f}')
            print(f'Test time: {time_since(start)}')
            
            print('mAP per class:')
            overall_map = []
            for class_name in class_names:
                if Average_precisions[class_name]:
                    class_map = Average_precisions[class_name][0]
                    print(f'{class_name}: {class_map:.3f}')
                    overall_map.append(class_map)
            
            if overall_map:
                print(f'Overall mAP: {np.mean(overall_map):.3f}')
            
            ap_file = os.path.join(save_detect_folder, f'evaluation_{args.fusion}.pkl')
            with open(ap_file, "wb") as fp:
                pickle.dump(Average_precisions, fp)
                
        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()