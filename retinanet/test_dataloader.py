#!/usr/bin/env python3
import os
import sys

sys.path.append('.')

from retinanet.dataloader_dsec_det import create_dsec_det_dataloader

def test_dataloader():
    """Test DSEC-Det dataloader functionality"""
    dataset_root = "/media/data/hucao/zhenwu/hucao/DSEC/DSEC_Det"
    split_config_path = "./retinanet/dsec_split.yaml"
    
    print("Testing DSEC-Det dataloader...")
    print(f"Dataset root: {dataset_root}")
    print(f"Split config: {split_config_path}")
    
    if not os.path.exists(dataset_root):
        print(f"❌ Dataset root does not exist: {dataset_root}")
        return False
    
    if not os.path.exists(split_config_path):
        print(f"⚠️  Split config not found: {split_config_path}")
        print("Will use auto-discovery instead")
        split_config_path = None
    
    try:
        train_loader, train_dataset = create_dsec_det_dataloader(
            root_dir=dataset_root,
            split='train',
            batch_size=2,
            num_workers=0,
            event_representation='time_surface',
            dt=50,
            image_height=480,
            image_width=640,
            split_config_path=split_config_path,
            debug=True,
            normalize_images=True,
            normalize_events=True,
            augment=False
        )
        
        print(f"✅ Train dataset: {len(train_dataset)} samples")
        print(f"✅ Classes: {train_dataset.num_classes()}")
        print(f"✅ Class names: {list(train_dataset.classes.keys())}")
        
        print("\nTesting data loading...")
        print("Starting first batch loading (before DataLoader iteration)...")
        for i, batch in enumerate(train_loader):
            print(f"✅ Batch {i}:")
            print(f"   Event images: {batch['img'].shape}")
            print(f"   RGB images: {batch['img_rgb'].shape}")
            print(f"   Annotations: {batch['annot'].shape}")
            print(f"   Event range: [{batch['img'].min():.3f}, {batch['img'].max():.3f}]")
            print(f"   RGB range: [{batch['img_rgb'].min():.3f}, {batch['img_rgb'].max():.3f}]")
            print(f"   Scale factors: {batch['scale']}")
            
            # Check annotation validity 
            annot_tensor = batch['annot']
            # annot_tensor shape: [batch_size, max_annotations, 5]
            
            # 统计有效标注
            valid_count = 0
            sample_annot = None
            
            for b in range(annot_tensor.shape[0]):  
                batch_annots = annot_tensor[b]  # [max_annotations, 5]
                
                valid_mask = batch_annots[:, 0] != -1
                valid_batch_annots = batch_annots[valid_mask]
                
                if len(valid_batch_annots) > 0:
                    valid_count += len(valid_batch_annots)
                    if sample_annot is None:
                        sample_annot = valid_batch_annots[0]
            
            if valid_count > 0:
                print(f"   Valid annotations: {valid_count} boxes")
                print(f"   Annotation sample: {sample_annot}")
            else:
                print(f"   No valid annotations in this batch")
            
            if i >= 2:
                break
        
        # Test validation split if available
        try:
            val_loader, val_dataset = create_dsec_det_dataloader(
                root_dir=dataset_root,
                split='val',
                batch_size=2,
                num_workers=0,
                event_representation='time_surface',
                dt=50,
                split_config_path=split_config_path,
                debug=False,
                augment=False
            )
            print(f"✅ Val dataset: {len(val_dataset)} samples")
        except Exception as e:
            print(f"⚠️  Val dataset not available: {e}")
            
        print("✅ Dataloader test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_sample():
    """Test loading a single sample to check data format"""
    dataset_root = "/media/data/hucao/zhenwu/hucao/DSEC/DSEC_Det"
    
    print("\n" + "="*50)
    print("Testing single sample loading...")
    
    try:
        _, dataset = create_dsec_det_dataloader(
            root_dir=dataset_root,
            split='train',
            batch_size=1,
            num_workers=0,
            shuffle=False
        )
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✅ Sample loaded successfully")
            print(f"   Event data shape: {sample['img'].shape}")
            print(f"   RGB data shape: {sample['img_rgb'].shape}")
            print(f"   Annotations shape: {sample['annot'].shape}")
            print(f"   Sequence: {sample.get('sequence', 'N/A')}")
            print(f"   Timestamp: {sample.get('timestamp', 'N/A')}")
            
            # Check data types and ranges
            print(f"   Event data type: {sample['img'].dtype}")
            print(f"   RGB data type: {sample['img_rgb'].dtype}")
            print(f"   Annotations data type: {sample['annot'].dtype}")
            
        else:
            print("❌ No samples found in dataset")
            
    except Exception as e:
        print(f"❌ Single sample test failed: {e}")
        import traceback
        traceback.print_exc()

def test_different_configurations():
    """Test different dataloader configurations"""
    dataset_root = "/media/data/hucao/zhenwu/hucao/DSEC/DSEC_Det"
    split_config_path = "./retinanet/dsec_split.yaml" if os.path.exists("./retinanet/dsec_split.yaml") else None
    
    configurations = [
        {
            'name': 'Time Surface + Normalization',
            'event_representation': 'time_surface',
            'normalize_events': True,
            'normalize_images': True
        },
        {
            'name': 'Event Count + No Normalization', 
            'event_representation': 'event_count',
            'normalize_events': False,
            'normalize_images': False
        },
        {
            'name': 'Binary + Event Normalization',
            'event_representation': 'binary',
            'normalize_events': True,
            'normalize_images': True
        }
    ]
    
    print("\n" + "="*50)
    print("Testing different configurations...")
    
    for config in configurations:
        print(f"\n🧪 Testing: {config['name']}")
        try:
            test_config = {k: v for k, v in config.items() if k != 'name'}
            
            loader, dataset = create_dsec_det_dataloader(
                root_dir=dataset_root,
                split='train',
                batch_size=1,
                num_workers=0,
                split_config_path=split_config_path,
                **test_config
            )
            
            batch = next(iter(loader))
            print(f"   ✅ Event shape: {batch['img'].shape}")
            print(f"   ✅ Event range: [{batch['img'].min():.3f}, {batch['img'].max():.3f}]")
            print(f"   ✅ RGB shape: {batch['img_rgb'].shape}")
            print(f"   ✅ RGB range: [{batch['img_rgb'].min():.3f}, {batch['img_rgb'].max():.3f}]")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")

def test_file_paths():
    """Test if required files exist in the dataset"""
    dataset_root = "/media/data/hucao/zhenwu/hucao/DSEC/DSEC_Det"
    
    print("\n" + "="*50)
    print("Testing file paths...")
    
    # Check train directory
    train_dir = os.path.join(dataset_root, 'train')
    if os.path.exists(train_dir):
        print(f"✅ Train directory found: {train_dir}")
        
        # List first few sequences
        sequences = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        print(f"✅ Found {len(sequences)} training sequences")
        
        if sequences:
            first_seq = sequences[0]
            seq_path = os.path.join(train_dir, first_seq)
            print(f"   Checking first sequence: {first_seq}")
            
            # Check required files
            events_file = os.path.join(seq_path, 'events', 'left', 'events.h5')
            tracks_file = os.path.join(seq_path, 'object_detections', 'left', 'tracks.npy')
            timestamps_file = os.path.join(seq_path, 'images', 'left', 'exposure_timestamps.txt')  
            image_dir = os.path.join(seq_path, 'images', 'left', 'rectified')
            
            print(f"   Events file: {'✅' if os.path.exists(events_file) else '❌'} {events_file}")
            print(f"   Tracks file: {'✅' if os.path.exists(tracks_file) else '❌'} {tracks_file}")
            print(f"   Timestamps file: {'✅' if os.path.exists(timestamps_file) else '❌'} {timestamps_file}")
            print(f"   Image directory: {'✅' if os.path.exists(image_dir) else '❌'} {image_dir}")
            
            if os.path.exists(image_dir):
                images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
                print(f"   Found {len(images)} images in rectified directory")
    else:
        print(f"❌ Train directory not found: {train_dir}")

if __name__ == "__main__":
    print("DSEC-Det Dataloader Test Suite")
    print("="*50)
    
    # Test file paths first
    test_file_paths()
    
    # Test single sample loading
    test_single_sample()
    
    # Run basic dataloader test
    success = test_dataloader()
    
    if success:
        # Run configuration tests
        test_different_configurations()
    else:
        print("\n❌ Basic test failed, skipping configuration tests")
        
    print("\n" + "="*50)
    print("Test completed!")
