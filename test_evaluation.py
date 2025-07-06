import argparse
import torch
import os
import time
from retinanet.dataloader_fast_combined import create_fast_dataloader as create_dsec_det_dataloader
from retinanet import model
from retinanet.csv_eval_dsec_det import evaluate, evaluate_coco_map

def test_model_loading(checkpoint_path, fusion='fpn_fusion', depth=50, use_cpu=False):
    print(f"Testing model loading from: {checkpoint_path}")
    
    try:
        if torch.cuda.is_available() and not use_cpu:
            print(f"GPU memory before loading: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            torch.cuda.empty_cache()
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("✓ Checkpoint loaded successfully")
        
        print("Checkpoint keys:", list(checkpoint.keys()))
        
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"Model config: {config}")
            fusion = config.get('fusion', fusion)
            depth = config.get('depth', depth)
        
        dummy_dataset_classes = 8
        
        print(f"Creating model with fusion={fusion}, classes={dummy_dataset_classes}")
        retinanet = model.resnet50(
            dataset_name='dsec',
            num_classes=dummy_dataset_classes,
            fusion_model=fusion,
            pretrained=False
        )
        print("✓ Model created successfully")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        missing_keys, unexpected_keys = retinanet.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys}")
        
        print("✓ Model weights loaded successfully")
        
        if torch.cuda.is_available() and not use_cpu:
            try:
                retinanet = retinanet.cuda()
                print("✓ Model moved to GPU")
                print(f"GPU memory after loading: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("⚠ GPU memory insufficient, falling back to CPU")
                    retinanet = retinanet.cpu()
                    use_cpu = True
                else:
                    raise e
        else:
            print("✓ Using CPU")
        
        retinanet.eval()
        print("✓ Model set to evaluation mode")
        
        return retinanet, True, use_cpu
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False, use_cpu

def test_dataset_loading(root_dir, split='val', batch_size=1):
    print(f"\nTesting dataset loading: {root_dir}")
    
    try:
        _, dataset = create_dsec_det_dataloader(
            root_dir=root_dir,
            split=split,
            batch_size=batch_size,
            num_workers=0
        )
        
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        print(f"✓ Number of classes: {dataset.num_classes()}")
        
        sample = dataset[0]
        print(f"✓ First sample loaded")
        print(f"  - RGB image shape: {sample['img_rgb'].shape}")
        print(f"  - Event image shape: {sample['img'].shape}")
        print(f"  - Annotations shape: {sample['annot'].shape}")
        
        return dataset, True
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False

def test_model_inference(retinanet, dataset, num_samples=3, use_cpu=False):
    print(f"\nTesting model inference on {num_samples} samples...")
    device = 'cpu' if use_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        retinanet.eval()
        successful_inferences = 0
        
        with torch.no_grad():
            for i in range(min(num_samples, len(dataset))):
                print(f"Testing sample {i+1}/{num_samples}")
                
                try:
                    sample = dataset[i]
                    
                    img_rgb = sample['img_rgb']
                    img_event = sample['img']
                    
                    if len(img_rgb.shape) == 3:
                        img_rgb = img_rgb.unsqueeze(0)
                    if len(img_event.shape) == 3:
                        img_event = img_event.unsqueeze(0)
                    
                    if device == 'cuda':
                        img_rgb = img_rgb.cuda().float()
                        img_event = img_event.cuda().float()
                    else:
                        img_rgb = img_rgb.float()
                        img_event = img_event.float()
                    
                    if device == 'cuda':
                        print(f"    GPU memory before inference: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    
                    scores, labels, boxes = retinanet([img_rgb, img_event])
                    
                    if device == 'cuda':
                        print(f"    GPU memory after inference: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    
                    print(f"  ✓ Sample {i+1}: {len(scores)} detections")
                    print(f"    Scores shape: {scores.shape}")
                    print(f"    Labels shape: {labels.shape}")
                    print(f"    Boxes shape: {boxes.shape}")
                    
                    if len(scores) > 0:
                        max_score = scores.max().item()
                        print(f"    Max detection score: {max_score:.4f}")
                    
                    successful_inferences += 1
                    
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"  ✗ Sample {i+1} failed: {e}")
                    if device == 'cuda' and "out of memory" in str(e):
                        print("    GPU memory issue detected, clearing cache...")
                        torch.cuda.empty_cache()
                    continue
        
        success_rate = successful_inferences / num_samples
        print(f"\nInference test results: {successful_inferences}/{num_samples} successful ({success_rate*100:.1f}%)")
        
        return success_rate > 0.5
        
    except Exception as e:
        print(f"✗ Model inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation_functions(retinanet, dataset, save_folder):
    print(f"\nTesting evaluation functions...")
    
    test_folder = os.path.join(save_folder, 'evaluation_test')
    os.makedirs(test_folder, exist_ok=True)
    
    subset_size = min(10, len(dataset))
    print(f"Using subset of {subset_size} samples for testing")
    
    class SubsetDataset:
        def __init__(self, original_dataset, size):
            self.original_dataset = original_dataset
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return self.original_dataset[idx]
        
        def num_classes(self):
            return self.original_dataset.num_classes()
        
        def label_to_name(self, label):
            return self.original_dataset.label_to_name(label)
    
    subset_dataset = SubsetDataset(dataset, subset_size)
    
    try:
        print("Testing standard evaluation...")
        start_time = time.time()
        
        mean_ap = evaluate(
            generator=subset_dataset,
            retinanet=retinanet,
            iou_threshold=0.5,
            score_threshold=0.05,
            max_detections=100,
            save_detection=True,
            save_folder=test_folder,
            load_detection=False,
            save_path=test_folder
        )
        
        eval_time = time.time() - start_time
        print(f"✓ Standard evaluation completed in {eval_time:.2f}s")
        print(f"  mAP@0.5: {mean_ap:.4f}")
        
        print("\nTesting COCO-style evaluation...")
        start_time = time.time()
        
        coco_aps = evaluate_coco_map(
            generator=subset_dataset,
            retinanet=retinanet,
            iou_threshold=0.5,
            score_threshold=0.05,
            max_detections=100,
            save_detection=True,
            save_folder=test_folder,
            load_detection=True,
            save_path=test_folder
        )
        
        eval_time = time.time() - start_time
        print(f"✓ COCO evaluation completed in {eval_time:.2f}s")
        
        if coco_aps:
            coco_map = sum([sum(aps) for aps in coco_aps.values()]) / sum([len(aps) for aps in coco_aps.values()])
            print(f"  COCO mAP: {coco_map:.4f}")
        
        generated_files = os.listdir(test_folder)
        print(f"✓ Generated files: {generated_files}")
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Test Evaluation Script')
    
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--root_dir', default='/media/data/hucao/zhenwu/hucao/DSEC/DSEC_Det', 
                       help='DSEC dataset root directory')
    parser.add_argument('--save_folder', default='./evaluation_test_results', 
                       help='Folder to save test results')
    parser.add_argument('--fusion', default='fpn_fusion', help='Fusion model type')
    parser.add_argument('--depth', type=int, default=50, help='ResNet depth')
    parser.add_argument('--inference_samples', type=int, default=3, 
                       help='Number of samples for inference testing')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DSEC Detection Evaluation Script Test")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset root: {args.root_dir}")
    print(f"Save folder: {args.save_folder}")
    print(f"Force CPU: {args.force_cpu}")
    
    if torch.cuda.is_available() and not args.force_cpu:
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        torch.cuda.empty_cache()
    
    os.makedirs(args.save_folder, exist_ok=True)
    
    all_tests_passed = True
    use_cpu = args.force_cpu
    
    print("\n" + "="*40)
    print("TEST 1: Model Loading")
    print("="*40)
    retinanet, model_ok, use_cpu = test_model_loading(args.checkpoint, args.fusion, args.depth, use_cpu)
    all_tests_passed &= model_ok
    
    if not model_ok:
        print("Model loading failed, stopping tests")
        return
    
    print("\n" + "="*40)
    print("TEST 2: Dataset Loading")
    print("="*40)
    dataset, dataset_ok = test_dataset_loading(args.root_dir, 'val', 1)
    all_tests_passed &= dataset_ok
    
    if not dataset_ok:
        print("Dataset loading failed, stopping tests")
        return
    
    print("\n" + "="*40)
    print("TEST 3: Model Inference")
    print("="*40)
    inference_ok = test_model_inference(retinanet, dataset, args.inference_samples, use_cpu)
    all_tests_passed &= inference_ok
    
    if not inference_ok:
        print("Model inference failed, stopping tests")
        return
    
    print("\n" + "="*40)
    print("TEST 4: Evaluation Functions")
    print("="*40)
    eval_ok = test_evaluation_functions(retinanet, dataset, args.save_folder)
    all_tests_passed &= eval_ok
    
    print("\n" + "="*60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! Evaluation script is working correctly.")
        if use_cpu:
            print("⚠ Note: Tests ran on CPU due to GPU memory constraints")
    else:
        print("❌ SOME TESTS FAILED! Check the error messages above.")
    print("="*60)
    
    print(f"\nTest results saved to: {args.save_folder}")

if __name__ == '__main__':
    main()
