#!/usr/bin/env python3
import os
import sys

sys.path.append('.')

from retinanet.dataloader_dsec_det import create_dsec_det_dataloader

def test_downsampled_events():
    """测试使用降采样事件数据"""
    dataset_root = "/media/data/hucao/zhenwu/hucao/DSEC/DSEC_Det"
    
    print("Testing with downsampled events...")
    print(f"Dataset root: {dataset_root}")
    
    try:
        # 使用降采样事件数据，可能没有压缩
        train_loader, train_dataset = create_dsec_det_dataloader(
            root_dir=dataset_root,
            split='train',
            batch_size=1,
            num_workers=0,  # 单线程避免问题
            event_representation='time_surface',
            dt=50,
            image_height=480,
            image_width=640,
            debug=True,
            normalize_images=False,  # 暂时禁用
            normalize_events=True,
            augment=False,  # 暂时禁用
            use_downsampled_events=True  # 关键：使用降采样事件
        )
        
        print(f"✅ Dataset created: {len(train_dataset)} samples")
        
        # 测试加载第一个样本
        print("\nTesting single sample loading...")
        sample = train_dataset[0]
        
        print(f"✅ Sample loaded successfully:")
        print(f"   Event data shape: {sample['img'].shape}")
        print(f"   RGB data shape: {sample['img_rgb'].shape}")
        print(f"   Annotations shape: {sample['annot'].shape}")
        print(f"   Event range: [{sample['img'].min():.3f}, {sample['img'].max():.3f}]")
        print(f"   RGB range: [{sample['img_rgb'].min():.3f}, {sample['img_rgb'].max():.3f}]")
        
        # 测试批量加载
        print("\nTesting batch loading...")
        for i, batch in enumerate(train_loader):
            print(f"✅ Batch {i}:")
            print(f"   Event images: {batch['img'].shape}")
            print(f"   RGB images: {batch['img_rgb'].shape}")
            print(f"   Annotations: {batch['annot'].shape}")
            
            if i >= 2:  # 只测试3个批次
                break
        
        print("✅ Downsampled events test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_downsampled_files():
    """检查是否有降采样事件文件"""
    dataset_root = "/media/data/hucao/zhenwu/hucao/DSEC/DSEC_Det"
    train_dir = os.path.join(dataset_root, "train")
    
    sequences = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    has_downsampled = False
    for seq in sequences[:3]:  # 检查前3个序列
        events_2x_file = os.path.join(train_dir, seq, "events", "left", "events_2x.h5")
        if os.path.exists(events_2x_file):
            print(f"✅ Found downsampled events: {events_2x_file}")
            print(f"   Size: {os.path.getsize(events_2x_file) / (1024*1024):.2f} MB")
            has_downsampled = True
        else:
            print(f"❌ No downsampled events in {seq}")
    
    return has_downsampled

if __name__ == "__main__":
    print("DSEC-Det Downsampled Events Test")
    print("="*50)
    
    # 首先检查是否有降采样文件
    if check_downsampled_files():
        print("\n" + "="*50)
        success = test_downsampled_events()
        
        if success:
            print("\n✅ Downsampled events work! You can use this as a workaround.")
            print("Recommendation: In your main code, set use_downsampled_events=True")
        else:
            print("\n❌ Even downsampled events failed.")
    else:
        print("\n❌ No downsampled events found. Need to fix blosc compression issue.")
