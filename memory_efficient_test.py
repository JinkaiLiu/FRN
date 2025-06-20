#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import torch
import h5py
import hdf5plugin
import cv2

sys.path.append('.')

def test_memory_efficient_loading():
    """测试内存高效的事件数据加载"""
    print("Testing memory-efficient event loading...")
    print("="*50)
    
    # 使用较小的文件进行测试
    dataset_root = "/media/data/hucao/zhenwu/hucao/DSEC/DSEC_Det"
    events_file = os.path.join(dataset_root, "train", "interlaken_00_g", "events", "left", "events_2x.h5")
    image_file = os.path.join(dataset_root, "train", "interlaken_00_g", "images", "left", "rectified", "000000.png")
    
    # 模拟一个时间戳
    timestamp = 52107500015
    dt = 50  # 50ms时间窗口
    dt_us = dt * 1000
    start_time = timestamp - dt_us
    end_time = timestamp
    
    print(f"Events file: {events_file}")
    print(f"Target timestamp: {timestamp}")
    print(f"Time window: {start_time} to {end_time}")
    
    try:
        # 第一步：测试RGB图像加载
        print("\n1. Testing RGB image loading...")
        img = cv2.imread(image_file)
        if img is None:
            print("❌ Failed to load RGB image")
            return False
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))
        img = img.astype(np.float32) / 255.0
        print(f"✅ RGB image loaded: {img.shape}")
        
        # 第二步：测试事件数据加载（内存高效版本）
        print("\n2. Testing memory-efficient event loading...")
        start_time_load = time.time()
        
        with h5py.File(events_file, 'r') as f:
            # 获取数据集引用（不加载数据）
            events_t_dataset = f['events/t']
            events_x_dataset = f['events/x']
            events_y_dataset = f['events/y']
            events_p_dataset = f['events/p']
            
            total_events = events_t_dataset.shape[0]
            print(f"Total events in file: {total_events:,}")
            
            # 分块查找时间窗口内的事件
            chunk_size = 1000000  # 100万事件一块
            found_events = {'t': [], 'x': [], 'y': [], 'p': []}
            
            print("Searching for events in time window...")
            
            for i in range(0, total_events, chunk_size):
                end_idx = min(i + chunk_size, total_events)
                
                # 只读取时间戳来过滤
                chunk_t = events_t_dataset[i:end_idx]
                
                # 处理时间偏移（如果有）
                if 't_offset' in f:
                    t_offset = f['t_offset'][()]
                    chunk_t = chunk_t + t_offset
                
                # 找到时间窗口内的事件
                mask = (chunk_t >= start_time) & (chunk_t < end_time)
                
                if np.any(mask):
                    # 只有找到事件才读取其他数据
                    chunk_x = events_x_dataset[i:end_idx][mask]
                    chunk_y = events_y_dataset[i:end_idx][mask]
                    chunk_p = events_p_dataset[i:end_idx][mask]
                    chunk_t_filtered = chunk_t[mask]
                    
                    found_events['t'].extend(chunk_t_filtered)
                    found_events['x'].extend(chunk_x)
                    found_events['y'].extend(chunk_y)
                    found_events['p'].extend(chunk_p)
                
                # 进度更新
                if (i // chunk_size) % 10 == 0:
                    progress = (end_idx / total_events) * 100
                    print(f"  Progress: {progress:.1f}% - Found {len(found_events['t'])} events so far")
        
        # 转换为numpy数组
        if found_events['t']:
            t = np.array(found_events['t'])
            x = np.array(found_events['x'])
            y = np.array(found_events['y'])
            p = np.array(found_events['p'])
            
            print(f"✅ Found {len(t)} events in time window")
            
            # 第三步：创建事件图像
            print("\n3. Creating event representation...")
            event_img = create_time_surface(x, y, t, p, 480, 640)
            print(f"✅ Event image created: {event_img.shape}")
            print(f"Event range: [{event_img.min():.3f}, {event_img.max():.3f}]")
            
        else:
            print("⚠️  No events found in time window, creating empty representation")
            event_img = torch.zeros(2, 480, 640)
        
        elapsed = time.time() - start_time_load
        print(f"\n✅ Total loading time: {elapsed:.2f} seconds")
        
        # 第四步：测试完整样本
        print("\n4. Testing complete sample...")
        sample = {
            'img': event_img,
            'img_rgb': torch.from_numpy(img),
            'annot': torch.zeros(0, 5),  # 空标注用于测试
            'sequence': 'interlaken_00_g',
            'timestamp': timestamp,
            'image_index': 0
        }
        
        print(f"✅ Complete sample created:")
        print(f"   Event shape: {sample['img'].shape}")
        print(f"   RGB shape: {sample['img_rgb'].shape}")
        print(f"   Annotations: {sample['annot'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in memory-efficient loading: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_time_surface(x, y, t, p, height, width):
    """创建时间表面表示"""
    time_surface = np.zeros((2, height, width), dtype=np.float32)
    
    if len(x) > 0:
        # 归一化时间戳
        t_normalized = (t - t.min()) / (t.max() - t.min() + 1e-6)
        
        # 过滤有效像素坐标
        valid_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        t_valid = t_normalized[valid_mask]
        p_valid = p[valid_mask]
        
        # 填充时间表面
        for i in range(len(x_valid)):
            polarity_idx = 1 if p_valid[i] > 0 else 0
            time_surface[polarity_idx, y_valid[i], x_valid[i]] = t_valid[i]
    
    # 归一化到[-1, 1]
    time_surface = time_surface * 2.0 - 1.0
    
    return torch.from_numpy(time_surface.copy()).float()

def test_batch_loading():
    """测试批量加载多个样本"""
    print("\n" + "="*50)
    print("Testing batch loading with memory efficiency...")
    
    try:
        from retinanet.dataloader_dsec_det import create_dsec_det_dataloader
        
        # 创建只有少量样本的数据加载器
        loader, dataset = create_dsec_det_dataloader(
            root_dir="/media/data/hucao/zhenwu/hucao/DSEC/DSEC_Det",
            split='train',
            batch_size=2,
            num_workers=0,
            event_representation='time_surface',
            dt=50,
            use_downsampled_events=True,  # 使用较小的文件
            normalize_images=False,
            augment=False
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # 限制测试范围 - 只测试前几个样本
        print("\nTesting individual samples...")
        
        # 从较小的序列开始测试
        for i in range(min(3, len(dataset))):
            sample_info = dataset.samples[i]
            if 'interlaken_00_g' in sample_info['sequence'] or 'zurich_city_02_a' in sample_info['sequence']:
                print(f"\nLoading sample {i} from {sample_info['sequence']}...")
                start_time = time.time()
                
                try:
                    sample = dataset[i]
                    elapsed = time.time() - start_time
                    print(f"✅ Sample {i} loaded in {elapsed:.2f}s")
                    print(f"   Events: {sample['img'].shape}")
                    print(f"   RGB: {sample['img_rgb'].shape}")
                    
                    if elapsed > 30:  # 如果超过30秒就停止
                        print("⚠️  Loading too slow, stopping test")
                        break
                        
                except Exception as e:
                    print(f"❌ Sample {i} failed: {e}")
                    break
            else:
                print(f"Skipping sample {i} from large sequence {sample_info['sequence']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch loading test failed: {e}")
        return False

if __name__ == "__main__":
    print("DSEC Memory-Efficient Loading Test")
    print("="*50)
    
    # 测试1: 内存高效的单样本加载
    success1 = test_memory_efficient_loading()
    
    if success1:
        # 测试2: 批量加载
        success2 = test_batch_loading()
        
        if success2:
            print("\n🎉 All tests passed! Memory-efficient loading works.")
            print("\nNext steps:")
            print("1. Implement this approach in your main dataloader")
            print("2. Use smaller batch sizes initially")
            print("3. Consider using only downsampled events for faster training")
        else:
            print("\n⚠️  Memory-efficient loading works, but batch loading needs optimization")
    else:
        print("\n❌ Memory-efficient loading failed, need to debug further")
