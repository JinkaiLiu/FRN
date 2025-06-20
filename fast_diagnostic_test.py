#!/usr/bin/env python3
import os
import sys
import time
import h5py
import hdf5plugin

def check_hdf5_file_sizes():
    """检查HDF5文件大小，找出可能的大文件"""
    dataset_root = "/media/data/hucao/zhenwu/hucao/DSEC/DSEC_Det"
    train_dir = os.path.join(dataset_root, "train")
    
    print("Checking HDF5 file sizes...")
    print("="*60)
    
    file_sizes = []
    
    sequences = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    for seq in sequences[:5]:  # 只检查前5个序列
        events_file = os.path.join(train_dir, seq, "events", "left", "events.h5")
        events_2x_file = os.path.join(train_dir, seq, "events", "left", "events_2x.h5")
        
        if os.path.exists(events_file):
            size_mb = os.path.getsize(events_file) / (1024*1024)
            file_sizes.append((seq, "events.h5", size_mb))
            print(f"{seq:<20} events.h5:     {size_mb:>8.2f} MB")
        
        if os.path.exists(events_2x_file):
            size_mb = os.path.getsize(events_2x_file) / (1024*1024)
            file_sizes.append((seq, "events_2x.h5", size_mb))
            print(f"{seq:<20} events_2x.h5:  {size_mb:>8.2f} MB")
    
    # 找出最大的文件
    if file_sizes:
        largest = max(file_sizes, key=lambda x: x[2])
        print(f"\nLargest file: {largest[0]} - {largest[1]} ({largest[2]:.2f} MB)")
        
        if largest[2] > 1000:  # 大于1GB
            print("⚠️  Warning: Large HDF5 files detected (>1GB)")
            print("   This could cause I/O bottlenecks during loading")
        
    return file_sizes

def test_quick_hdf5_read():
    """测试快速读取HDF5文件的少量数据"""
    dataset_root = "/media/data/hucao/zhenwu/hucao/DSEC/DSEC_Det"
    test_file = os.path.join(dataset_root, "train", "interlaken_00_g", "events", "left", "events_2x.h5")
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    print(f"\nTesting quick read of: {test_file}")
    print(f"File size: {os.path.getsize(test_file) / (1024*1024):.2f} MB")
    
    try:
        start_time = time.time()
        
        with h5py.File(test_file, 'r') as f:
            print("✅ File opened successfully")
            
            # 获取数据集信息但不读取数据
            events_t = f['events/t']
            events_x = f['events/x']
            events_y = f['events/y']
            events_p = f['events/p']
            
            print(f"Dataset shapes:")
            print(f"  t: {events_t.shape} {events_t.dtype}")
            print(f"  x: {events_x.shape} {events_x.dtype}")
            print(f"  y: {events_y.shape} {events_y.dtype}")
            print(f"  p: {events_p.shape} {events_p.dtype}")
            
            total_events = events_t.shape[0]
            print(f"Total events: {total_events:,}")
            
            # 只读取前1000个事件测试速度
            print("Reading first 1000 events...")
            sample_t = events_t[:1000]
            sample_x = events_x[:1000]
            sample_y = events_y[:1000]
            sample_p = events_p[:1000]
            
            elapsed = time.time() - start_time
            print(f"✅ Read 1000 events in {elapsed:.2f} seconds")
            
            # 测试读取更多数据
            print("Reading first 100,000 events...")
            start_time = time.time()
            chunk_t = events_t[:100000]
            elapsed = time.time() - start_time
            print(f"✅ Read 100,000 events in {elapsed:.2f} seconds")
            
            if elapsed > 10:  # 如果读取10万事件超过10秒
                print("⚠️  Warning: Slow I/O detected")
                return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error reading HDF5 file: {e}")
        return False

def test_dataset_creation_only():
    """只测试数据集创建，不加载样本"""
    print("\nTesting dataset creation only...")
    
    try:
        sys.path.append('.')
        from retinanet.dataloader_dsec_det import DSECDetDataset
        
        print("Creating dataset with limited samples...")
        dataset = DSECDetDataset(
            root_dir="/media/data/hucao/zhenwu/hucao/DSEC/DSEC_Det",
            split='train',
            transform=None,
            debug=True
        )
        
        print(f"✅ Dataset created: {len(dataset)} samples")
        print(f"✅ Number of classes: {dataset.num_classes()}")
        
        # 只查看样本信息，不加载数据
        print("\nFirst few samples info:")
        for i in range(min(3, len(dataset))):
            sample_info = dataset.samples[i]
            print(f"Sample {i}:")
            print(f"  Sequence: {sample_info['sequence']}")
            print(f"  Image path: {os.path.basename(sample_info['image_path'])}")
            print(f"  Events path: {os.path.basename(sample_info['events_path'])}")
            print(f"  Timestamp: {sample_info['timestamp']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("DSEC-Det Fast Diagnostic Test")
    print("="*50)
    
    # 1. 检查文件大小
    file_sizes = check_hdf5_file_sizes()
    
    # 2. 测试HDF5读取速度
    hdf5_ok = test_quick_hdf5_read()
    
    # 3. 测试数据集创建
    dataset_ok = test_dataset_creation_only()
    
    print("\n" + "="*50)
    print("Diagnostic Summary:")
    print(f"HDF5 reading: {'✅ OK' if hdf5_ok else '❌ SLOW/FAILED'}")
    print(f"Dataset creation: {'✅ OK' if dataset_ok else '❌ FAILED'}")
    
    if hdf5_ok and dataset_ok:
        print("\n✅ Basic components work. The issue is likely:")
        print("   1. Memory usage when loading full events")
        print("   2. I/O bottleneck with large files")
        print("\nRecommendation: Use optimized loader with caching")
    else:
        print("\n❌ Found fundamental issues that need fixing first")
