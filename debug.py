#!/usr/bin/env python3

import h5py
import numpy as np
from pathlib import Path

def analyze_event_timing():
    """
    分析事件文件的时间分布，找出为什么没有匹配到事件
    """
    
    # 使用你的数据路径
    root_dir = '/media/data/hucao/zhenwu/hucao/DSEC/DSEC_Det'
    
    # 找第一个序列
    train_dir = Path(root_dir) / 'train'
    sequences = list(train_dir.iterdir())
    seq_path = sequences[0]
    
    print(f"分析序列: {seq_path.name}")
    
    # 读取时间戳文件
    timestamps_file = seq_path / 'images' / 'left' / 'exposure_timestamps.txt'
    
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
    
    print(f"图像时间戳数量: {len(image_timestamps)}")
    print(f"图像时间戳范围: [{min(image_timestamps)}, {max(image_timestamps)}]")
    
    # 读取事件文件
    events_file = seq_path / 'events' / 'left' / 'events_2x.h5'
    if not events_file.exists():
        events_file = seq_path / 'events' / 'left' / 'events.h5'
    
    print(f"事件文件: {events_file}")
    
    with h5py.File(str(events_file), 'r') as f:
        # 检查t_offset
        t_offset = 0
        if 't_offset' in f:
            t_offset = f['t_offset'][()]
            print(f"t_offset: {t_offset}")
        
        # 读取事件时间戳 (只读一部分以节省内存)
        events_t = f['events/t']
        
        # 采样读取
        total_events = len(events_t)
        sample_size = min(100000, total_events)
        step = max(1, total_events // sample_size)
        
        sampled_t = events_t[::step] + t_offset
        
        print(f"事件时间戳采样 ({len(sampled_t)}/{total_events}):")
        print(f"事件时间戳范围: [{sampled_t.min()}, {sampled_t.max()}]")
        
        # 检查时间单位
        print(f"时间单位分析:")
        print(f"  事件时间戳最小值: {sampled_t.min()}")
        print(f"  图像时间戳最小值: {min(image_timestamps)}")
        print(f"  比例: {min(image_timestamps) / sampled_t.min():.2f}")
        
        # 测试前几个图像时间戳
        dt_us = 50 * 1000  # 50ms
        
        print(f"\n测试时间窗口匹配 (dt={dt_us}us):")
        
        for i in range(min(5, len(image_timestamps))):
            img_timestamp = image_timestamps[i]
            start_time = img_timestamp - dt_us
            end_time = img_timestamp
            
            print(f"\n图像 {i}: timestamp={img_timestamp}")
            print(f"  时间窗口: [{start_time}, {end_time}]")
            
            # 检查有多少事件在这个窗口内
            mask = (sampled_t >= start_time) & (sampled_t < end_time)
            num_events = mask.sum()
            
            print(f"  匹配事件数 (采样): {num_events}")
            
            if num_events == 0:
                # 找最近的事件
                time_diffs = np.abs(sampled_t - img_timestamp)
                closest_idx = np.argmin(time_diffs)
                closest_time = sampled_t[closest_idx]
                closest_diff = time_diffs[closest_idx]
                
                print(f"  最近事件时间: {closest_time} (差值: {closest_diff}us)")
                
                # 尝试不同的dt值
                for test_dt in [100, 500, 1000, 5000, 10000]:  # ms
                    test_dt_us = test_dt * 1000
                    test_start = img_timestamp - test_dt_us
                    test_end = img_timestamp + test_dt_us  # 注意这里改为+
                    
                    test_mask = (sampled_t >= test_start) & (sampled_t <= test_end)
                    test_count = test_mask.sum()
                    
                    if test_count > 0:
                        print(f"  dt={test_dt}ms: {test_count} events")
                        break
        
        # 建议的修复参数
        print(f"\n建议的修复方案:")
        
        # 计算时间对齐
        img_time_span = max(image_timestamps) - min(image_timestamps)
        event_time_span = sampled_t.max() - sampled_t.min()
        
        print(f"图像时间跨度: {img_time_span}us ({img_time_span/1e6:.2f}s)")
        print(f"事件时间跨度: {event_time_span}us ({event_time_span/1e6:.2f}s)")
        
        if abs(img_time_span - event_time_span) / max(img_time_span, event_time_span) > 0.1:
            print("⚠️  时间跨度差异较大，可能需要调整时间对齐")
        
        # 建议dt值
        avg_gap = img_time_span / len(image_timestamps) if image_timestamps else 0
        suggested_dt_ms = max(50, avg_gap / 1000 * 2)  # 建议dt为平均间隔的2倍
        
        print(f"建议 dt 值: {suggested_dt_ms:.0f}ms")

if __name__ == "__main__":
    analyze_event_timing()
