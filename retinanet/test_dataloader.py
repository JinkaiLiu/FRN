#!/usr/bin/env python3
import os
import sys

# Add current directory to path if needed
sys.path.append('.')

# Import the dataloader function (adjust the import based on your filename)
from retinanet.dataloader_dsec_det import create_dsec_det_dataloader  # Changed: function name corrected

def test_dataloader():
    """Test DSEC-Det dataloader functionality"""
    # Root directory for the DSEC dataset
    dataset_root = "/media/data/hucao/zhenwu/hucao/DSEC/DSEC_Det"
    split_config_path = "./retinanet/dsec_split.yaml"
    
    print("Testing DSEC-Det dataloader...")
    print(f"Dataset root: {dataset_root}")
    print(f"Split config: {split_config_path}")
    
    # Check if dataset root exists
    if not os.path.exists(dataset_root):
        print(f"❌ Dataset root does not exist: {dataset_root}")
        return False
    
    # Check if split config exists
    if not os.path.exists(split_config_path):
        print(f"⚠️  Split config not found: {split_config_path}")
        print("Will use auto-discovery instead")
        split_config_path = None
    
    # Test the dataloader creation
    try:
        train_loader, train_dataset = create_dsec_det_dataloader(  # Changed: corrected function name
            root_dir=dataset_root,
            split='train',
            batch_size=2,
            num_workers=1,
            event_representation='time_surface',
            dt=50,
            image_height=480,
            image_width=640,
            split_config_path=split_config_path,
            debug=True,
            normalize_images=True,  # Added: enable normalization
            normalize_events=True,  # Added: enable event normalization
            augment=False  # Added: disable augmentation for testing
        )
        
        print(f"✅ Train dataset: {len(train_dataset)} samples")
        print(f"✅ Classes: {train_dataset.num_classes()}")
        print(f"✅ Class names: {list(train_dataset.classes.keys())}")
        
        # Test the first few batches
        print("\nTesting data loading...")
        for i, batch in enumerate(train_loader):
            print(f"✅ Batch {i}:")
            print(f"   Event images: {batch['img'].shape}")
            print(f"   RGB images: {batch['img_rgb'].shape}")
            print(f"   Annotations: {batch['annot'].shape}")
            print(f"   Event range: [{batch['img'].min():.3f}, {batch['img'].max():.3f}]")
            print(f"   RGB range: [{batch['img_rgb'].min():.3f}, {batch['img_rgb'].max():.3f}]")
            print(f"   Scale factors: {batch['scale']}")
            
            # Check annotation validity
            valid_annots = batch['annot'][batch['annot'][:, :, 0] != -1]  # Filter out padding
            if len(valid_annots) > 0:
                print(f"   Valid annotations: {len(valid_annots)} boxes")
                print(f"   Annotation sample: {valid_annots[0] if len(valid_annots) > 0 else 'None'}")
            else:
                print(f"   No valid annotations in this batch")
            
            if i >= 2:  # Test first 3 batches
                break
        
        # Test validation split if available
        try:
            val_loader, val_dataset = create_dsec_det_dataloader(
                root_dir=dataset_root,
                split='val',
                batch_size=2,
                num_workers=1,
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
        traceback.print_exc()  # Added: print full traceback for debugging
        return False

def test_different_configurations():
    """Test different dataloader configurations"""
    dataset_root = "/media/data/hucao/zhenwu/hucao/DSEC/DSEC_Det"
    split_config_path = "./retinanet/dsec_split.yaml"
    
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
            'name': 'Binary + Downsampled Events',
            'event_representation': 'binary',
            'use_downsampled_events': True,
            'normalize_events': True
        }
    ]
    
    print("\n" + "="*50)
    print("Testing different configurations...")
    
    for config in configurations:
        print(f"\n🧪 Testing: {config['name']}")
        try:
            # Remove 'name' from config before passing to function
            test_config = {k: v for k, v in config.items() if k != 'name'}
            
            loader, dataset = create_dsec_det_dataloader(
                root_dir=dataset_root,
                split='train',
                batch_size=1,
                num_workers=0,  # Single threaded for testing
                split_config_path=split_config_path,
                **test_config
            )
            
            # Test one batch
            batch = next(iter(loader))
            print(f"   ✅ Event shape: {batch['img'].shape}")
            print(f"   ✅ Event range: [{batch['img'].min():.3f}, {batch['img'].max():.3f}]")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")

if __name__ == "__main__":
    # Run basic test
    success = test_dataloader()
    
    if success:
        # Run configuration tests
        test_different_configurations()
    else:
        print("\n❌ Basic test failed, skipping configuration tests")
        
    print("\n" + "="*50)
    print("Test completed!")