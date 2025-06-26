import os
from pathlib import Path
import numpy as np
try:
    import hdf5plugin
    import blosc
    HDF5_PLUGINS_AVAILABLE = True
except ImportError:
    HDF5_PLUGINS_AVAILABLE = False
    print("Warning: hdf5plugin or blosc not available. Some HDF5 files may not load correctly.")
import h5py
from tqdm import tqdm

def convert_h5_to_npz(events_h5_path: Path, output_npz_path: Path):
    """
    修复的H5到NPZ转换函数，正确处理t_offset
    """
    try:
        print(f"🔄 转换: {events_h5_path.name}")
        
        with h5py.File(events_h5_path, 'r') as f:
            # 读取原始事件数据
            try:
                x = np.array(f['events/x'], copy=True)
                y = np.array(f['events/y'], copy=True)
                t = np.array(f['events/t'], copy=True)  # 原始时间戳
                p = np.array(f['events/p'], copy=True)
                
                print(f"  📊 原始数据: {len(x)} 个事件")
                print(f"  ⏰ 原始时间范围: [{t.min()}, {t.max()}]")
                
            except Exception as read_error:
                print(f"  ❌ 读取事件数据失败: {read_error}")
                if "required filter" in str(read_error):
                    print("  🔧 压缩问题，尝试替代方法...")
                    try:
                        x = np.array(f['events/x'])
                        y = np.array(f['events/y'])
                        t = np.array(f['events/t'])
                        p = np.array(f['events/p'])
                    except:
                        raise read_error
                else:
                    raise read_error
            
            # 关键修复：正确处理t_offset
            if 't_offset' in f:
                t_offset = f['t_offset'][()]
                print(f"  ✅ 发现t_offset: {t_offset}")
                print(f"  ⏰ 修正前时间范围: [{t.min()}, {t.max()}]")
                
                # 应用t_offset修正
                t = t + t_offset
                
                print(f"  ✅ 修正后时间范围: [{t.min()}, {t.max()}]")
            else:
                print(f"  ⚠️  没有找到t_offset")
            
            # 数据验证
            if len(x) == 0:
                print(f"  ❌ 事件数据为空")
                return False
            
            # 空间坐标验证
            if x.max() >= 640 or y.max() >= 480 or x.min() < 0 or y.min() < 0:
                print(f"  ⚠️  空间坐标超出范围: X[{x.min()}, {x.max()}], Y[{y.min()}, {y.max()}]")
            
            # 极性验证
            unique_polarities = np.unique(p)
            print(f"  📈 极性值: {unique_polarities}")
            if not np.all(np.isin(unique_polarities, [0, 1])):
                print(f"  ⚠️  异常的极性值")
        
        # 保存NPZ文件
        print(f"  💾 保存到: {output_npz_path.name}")
        np.savez_compressed(output_npz_path, x=x, y=y, t=t, p=p)
        
        # 验证保存的文件
        verify_data = np.load(str(output_npz_path))
        saved_t = verify_data['t']
        print(f"  ✅ 验证保存: {len(saved_t)} 个事件, 时间范围: [{saved_t.min()}, {saved_t.max()}]")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 转换失败: {e}")
        return False

def verify_conversion(events_h5_path: Path, events_npz_path: Path, sample_timestamps=None):
    """
    验证转换结果的正确性
    """
    print(f"\n🔍 验证转换结果: {events_npz_path.name}")
    
    try:
        # 加载NPZ数据
        npz_data = np.load(str(events_npz_path))
        npz_t = npz_data['t']
        
        # 加载H5数据进行对比
        with h5py.File(events_h5_path, 'r') as f:
            h5_t_raw = np.array(f['events/t'][:1000])  # 只验证前1000个事件
            t_offset = f['t_offset'][()] if 't_offset' in f else 0
            h5_t_corrected = h5_t_raw + t_offset
        
        npz_t_sample = npz_t[:1000]
        
        # 比较时间戳
        time_diff = np.abs(npz_t_sample - h5_t_corrected).max()
        print(f"  ⏰ NPZ vs H5时间差异: {time_diff} 微秒")
        
        if time_diff < 1:  # 允许1微秒的误差
            print(f"  ✅ 时间戳验证通过")
        else:
            print(f"  ❌ 时间戳验证失败")
            print(f"    NPZ样本: {npz_t_sample[:5]}")
            print(f"    H5样本: {h5_t_corrected[:5]}")
        
        # 如果提供了图像时间戳，测试时间窗口
        if sample_timestamps:
            img_ts = sample_timestamps[0]
            dt_us = 50 * 1000
            start_time = img_ts - dt_us
            end_time = img_ts
            
            mask = (npz_t >= start_time) & (npz_t < end_time)
            events_in_window = mask.sum()
            
            print(f"  🎯 测试时间窗口:")
            print(f"    图像时间戳: {img_ts}")
            print(f"    窗口: [{start_time}, {end_time}]")
            print(f"    窗口内事件: {events_in_window}")
            
            if events_in_window > 0:
                print(f"  ✅ 时间窗口验证通过")
                return True
            else:
                print(f"  ❌ 时间窗口验证失败")
                return False
    
    except Exception as e:
        print(f"  ❌ 验证过程出错: {e}")
        return False

def process_dataset(root_dir, verify_with_images=True):
    """
    处理整个数据集，转换H5到NPZ
    """
    root = Path(root_dir)
    splits = ['train', 'test']
    total_converted = 0
    total_verified = 0
    
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"⚠️  {split} 目录不存在: {split_dir}")
            continue
        
        print(f"\n📁 处理 {split} 数据集:")
        
        for sequence in split_dir.iterdir():
            if not sequence.is_dir():
                continue
                
            print(f"\n🎬 序列: {sequence.name}")
            
            # 定位文件
            events_h5 = sequence / 'events' / 'left' / 'events_2x.h5'
            events_npz = events_h5.with_suffix('.npz')
            
            if not events_h5.exists():
                print(f"  ⚠️  H5文件不存在: {events_h5}")
                continue
            
            # 检查是否需要转换
            if events_npz.exists():
                print(f"  ✅ NPZ文件已存在: {events_npz.name}")
                
                # 验证现有文件
                if verify_with_images:
                    timestamps_file = sequence / 'images' / 'left' / 'exposure_timestamps.txt'
                    if timestamps_file.exists():
                        # 读取图像时间戳
                        image_timestamps = []
                        with open(timestamps_file, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith('#'):
                                    timestamp = line.split(',')[0].strip()
                                    try:
                                        image_timestamps.append(int(timestamp))
                                    except ValueError:
                                        continue
                        
                        if image_timestamps:
                            is_valid = verify_conversion(events_h5, events_npz, image_timestamps[:3])
                            if is_valid:
                                total_verified += 1
                            else:
                                print(f"  🔄 现有NPZ文件验证失败，重新转换...")
                                success = convert_h5_to_npz(events_h5, events_npz)
                                if success:
                                    verify_conversion(events_h5, events_npz, image_timestamps[:3])
                                    total_converted += 1
                        else:
                            print(f"  ⚠️  无法读取图像时间戳")
                continue
            
            # 执行转换
            print(f"  🔄 开始转换...")
            success = convert_h5_to_npz(events_h5, events_npz)
            
            if success:
                total_converted += 1
                
                # 验证转换结果
                if verify_with_images:
                    timestamps_file = sequence / 'images' / 'left' / 'exposure_timestamps.txt'
                    if timestamps_file.exists():
                        # 读取图像时间戳用于验证
                        image_timestamps = []
                        with open(timestamps_file, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith('#'):
                                    timestamp = line.split(',')[0].strip()
                                    try:
                                        image_timestamps.append(int(timestamp))
                                    except ValueError:
                                        continue
                        
                        if image_timestamps:
                            verify_conversion(events_h5, events_npz, image_timestamps[:3])
                
                print(f"  ✅ 转换完成")
            else:
                print(f"  ❌ 转换失败")
    
    print(f"\n🎉 处理完成:")
    print(f"  新转换: {total_converted} 个文件")
    print(f"  验证通过: {total_verified} 个文件")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="修复的H5到NPZ转换工具")
    parser.add_argument("--root_dir", required=True, help="DSEC-Det数据集根目录")
    parser.add_argument("--verify", action="store_true", help="验证转换结果")
    args = parser.parse_args()
    
    process_dataset(args.root_dir, verify_with_images=args.verify)
