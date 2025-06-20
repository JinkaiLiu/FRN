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

def load_model(fusion_type, num_classes, checkpoint_path, dataset_name='dsec'):
    """Load model from checkpoint"""
    list_models = ['fpn_fusion', 'event', 'rgb']
    if fusion_type in list_models:
        retinanet = model.resnet50(
            dataset_name=dataset_name,
            num_classes=num_classes, 
            fusion_model=fusion_type, 
            pretrained=False
        )
    else:
        raise ValueError('Unsupported model fusion')

    # Load checkpoint with proper error handling
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        retinanet.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', [])
    else:
        # Assume checkpoint contains model state directly
        retinanet.load_state_dict(checkpoint)
        epoch = 0
        loss = []
    
    return retinanet, epoch, loss

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
    parser.add_argument('--checkpoint', help='location of pretrained file', default='./checkpoints/best_model.pt')
    parser.add_argument('--batch_size', help='Batch size for evaluation', type=int, default=1)
    parser.add_argument('--num_workers', help='Number of workers', type=int, default=4)
    parser.add_argument('--eval_corruption', help='evaluate on corrupted images', action='store_true')
    parser.add_argument('--corruption_group', help='corruption group number', type=int, default=0)
    parser.add_argument('--corruption_root', help='root directory for corrupted images', 
                       default='/media/data/hucao/zhenwu/hucao/DSEC/corruptions')
    parser.add_argument('--save_results', help='save detection results', action='store_true')
    parser.add_argument('--results_dir', help='directory to save results', 
                       default='./results')
    parser.add_argument('--score_threshold', help='detection score threshold', type=float, default=0.05)
    parser.add_argument('--iou_threshold', help='IoU threshold for evaluation', type=float, default=0.5)
    parser.add_argument('--use_coco_eval', help='use COCO-style evaluation', action='store_true')

    return parser

def create_test_dataloader(args, corruption_type=None, severity=None):
    """Create test dataloader, optionally with corruption"""
    
    if corruption_type and severity:
        dataset_root = os.path.join(args.corruption_root, corruption_type, f'severity_{severity}')
        print(f"Testing on corruption: {corruption_type}, severity: {severity}")
        if not os.path.exists(dataset_root):
            raise FileNotFoundError(f"Corruption dataset not found: {dataset_root}")
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
        split_config_path=args.split_config if os.path.exists(args.split_config) else None,
        normalize_events=True,
        normalize_images=True,
        use_downsampled_events=args.use_downsampled_events
    )
    
    return dataloader, dataset

def evaluate_model(retinanet, dataset, args, save_folder):
    """Evaluate model on dataset with proper error handling"""
    
    # Ensure model is in eval mode
    retinanet.eval()
    
    try:
        if args.use_coco_eval:
            # Use COCO-style evaluation
            mAP_results = csv_eval_dsec_det.evaluate_coco_map(
                dataset, retinanet,
                score_threshold=args.score_threshold,
                save_detection=args.save_results,
                save_folder=save_folder,
                load_detection=False
            )
            
            # Convert COCO mAP results to class-wise results
            class_names = list(dataset.classes.keys())
            results = {}
            for i, class_name in enumerate(class_names):
                if i < len(mAP_results) and len(mAP_results[i]) > 0:
                    # Average over different IoU thresholds for COCO-style mAP
                    results[class_name] = np.mean(mAP_results[i])
                else:
                    results[class_name] = 0.0
                    
        else:
            # Use standard Pascal VOC evaluation
            mAP_results = csv_eval_dsec_det.evaluate(
                dataset, retinanet,
                iou_threshold=args.iou_threshold,
                score_threshold=args.score_threshold,
                save_detection=args.save_results,
                save_folder=save_folder,
                load_detection=False
            )
            
            # Convert evaluation results to class-wise format
            class_names = list(dataset.classes.keys())
            results = {}
            for i, class_name in enumerate(class_names):
                if i in mAP_results:
                    results[class_name] = mAP_results[i][0]  # AP value
                else:
                    results[class_name] = 0.0
                    
        return results
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main(args=None):
    parser = setup_arg_parser()
    args = parser.parse_args(args)
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    print(f"Testing {args.fusion} model on DSEC-Det dataset")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Event representation: {args.event_representation}")
    print(f"Time window: {args.dt}ms")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load training dataset to get class information
    try:
        train_loader, train_dataset = create_dsec_det_dataloader(
            root_dir=args.dataset_root,
            split='train',
            batch_size=1,
            num_workers=0,
            shuffle=False,
            split_config_path=args.split_config if os.path.exists(args.split_config) else None,
            event_representation=args.event_representation,
            dt=args.dt,
            image_height=args.image_height,
            image_width=args.image_width,
            use_downsampled_events=args.use_downsampled_events
        )
        num_classes = train_dataset.num_classes()
        class_names = list(train_dataset.classes.keys())
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {class_names}")
    except Exception as e:
        print(f"Warning: Could not load train dataset for class info: {e}")
        num_classes = 8  # Default for DSEC-Det
        class_names = ['pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train']
    
    # Load model from checkpoint
    try:
        retinanet, epoch_total, epoch_loss_all = load_model(
            args.fusion, num_classes, args.checkpoint, args.dataset_name
        )
        print(f'Loaded {args.fusion} model from epoch {epoch_total}')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Setup model for evaluation
    retinanet.eval()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
        print("Using GPU for evaluation")
    else:
        retinanet = torch.nn.DataParallel(retinanet)
        print("Using CPU for evaluation")

    retinanet.training = False
    if hasattr(retinanet.module, 'freeze_bn'):
        retinanet.module.freeze_bn()
    
    # Define corruption types for robustness evaluation
    corruption_types = [
        ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur'],
        ['motion_blur', 'zoom_blur', 'fog', 'snow', 'frost'],
        ['brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    ]

    if args.eval_corruption:
        print(f"Evaluating on corruption group {args.corruption_group}")
        corruption_list = corruption_types[args.corruption_group]
        severity_list = [1, 2, 3, 4, 5]
        print(f"Evaluating corruptions: {corruption_list}")
        
        # Initialize results storage
        all_corruption_results = {}
        
        for corruption in corruption_list:
            print(f"\n=== Evaluating corruption: {corruption} ===")
            corruption_results = {class_name: [] for class_name in class_names}
            start_corruption = time.time()
            
            for severity in severity_list:
                print(f"Processing severity {severity}...")
                save_detect_folder = os.path.join(
                    args.results_dir, f'{args.fusion}_{args.event_representation}', 
                    corruption, f'severity_{severity}'
                )
                os.makedirs(save_detect_folder, exist_ok=True)
                
                try:
                    test_loader, test_dataset = create_test_dataloader(args, corruption, severity)
                    print(f"Test dataset: {len(test_dataset)} samples")
                    
                    start_eval = time.time()
                    results = evaluate_model(retinanet, test_dataset, args, save_detect_folder)
                    
                    if results is not None:
                        for class_name in class_names:
                            if class_name in results:
                                corruption_results[class_name].append(results[class_name])
                        
                        eval_time = time.time() - start_eval
                        fps = len(test_dataset) / eval_time if eval_time > 0 else 0
                        print(f"Severity {severity} completed - Time: {time_since(start_eval)}, FPS: {fps:.2f}")
                    
                except Exception as e:
                    print(f"Error evaluating {corruption} severity {severity}: {e}")
                    # Add zero values for failed evaluations
                    for class_name in class_names:
                        corruption_results[class_name].append(0.0)
                    continue
            
            # Print results for this corruption
            print(f'\n{args.fusion} - {corruption} Results:')
            corruption_means = {}
            for class_name in class_names:
                if corruption_results[class_name]:
                    mean_ap = np.mean(corruption_results[class_name])
                    corruption_means[class_name] = mean_ap
                    print(f'{class_name}: {mean_ap:.3f}')
                else:
                    corruption_means[class_name] = 0.0
                    print(f'{class_name}: 0.000')
            
            overall_mean = np.mean(list(corruption_means.values()))
            print(f'Overall mAP for {corruption}: {overall_mean:.3f}')
            print(f'Time for {corruption}: {time_since(start_corruption)}')
            
            # Save results
            all_corruption_results[corruption] = corruption_results
            
            # Save individual corruption results
            results_file = os.path.join(
                args.results_dir, f'{args.fusion}_{args.event_representation}', 
                f'{corruption}_results.pkl'
            )
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            with open(results_file, "wb") as fp:
                pickle.dump(corruption_results, fp)
        
        # Save all corruption results
        all_results_file = os.path.join(
            args.results_dir, f'{args.fusion}_{args.event_representation}', 
            f'all_corruptions_group_{args.corruption_group}.pkl'
        )
        with open(all_results_file, "wb") as fp:
            pickle.dump(all_corruption_results, fp)
        
        print(f"\nAll corruption results saved to: {all_results_file}")

    else:
        # Standard evaluation on clean test set
        print("\n=== Standard Evaluation ===")
        
        try:
            test_loader, test_dataset = create_test_dataloader(args)
            print(f"Test dataset: {len(test_dataset)} samples")
        except Exception as e:
            print(f"Failed to create test dataloader: {e}")
            return
        
        save_detect_folder = os.path.join(
            args.results_dir, f'{args.fusion}_{args.event_representation}', 'clean_evaluation'
        )
        os.makedirs(save_detect_folder, exist_ok=True)
        
        start_eval = time.time()
        results = evaluate_model(retinanet, test_dataset, args, save_detect_folder)
        
        if results is not None:
            eval_time = time.time() - start_eval
            fps = len(test_dataset) / eval_time if eval_time > 0 else 0
            
            print(f'\nEvaluation Results:')
            print(f'Evaluation time: {time_since(start_eval)}')
            print(f'FPS: {fps:.2f}')
            print(f'Score threshold: {args.score_threshold}')
            print(f'IoU threshold: {args.iou_threshold}')
            
            print('\nmAP per class:')
            class_aps = []
            for class_name in class_names:
                if class_name in results:
                    class_ap = results[class_name]
                    print(f'{class_name}: {class_ap:.3f}')
                    class_aps.append(class_ap)
                else:
                    print(f'{class_name}: 0.000')
                    class_aps.append(0.0)
            
            if class_aps:
                overall_map = np.mean(class_aps)
                print(f'\nOverall mAP: {overall_map:.3f}')
            
            # Save results
            results_summary = {
                'class_results': results,
                'overall_mAP': overall_map if class_aps else 0.0,
                'fps': fps,
                'eval_time': eval_time,
                'num_samples': len(test_dataset),
                'args': vars(args)
            }
            
            results_file = os.path.join(save_detect_folder, f'evaluation_results_{args.fusion}.pkl')
            with open(results_file, "wb") as fp:
                pickle.dump(results_summary, fp)
            
            print(f"Results saved to: {results_file}")
        
        else:
            print("Evaluation failed!")

if __name__ == '__main__':
    main()