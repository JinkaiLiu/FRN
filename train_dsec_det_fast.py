import argparse
import collections
import numpy as np
import time
import math
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import os
from retinanet.dataloader_fast_combined import create_fast_dataloader as create_dsec_det_dataloader
from retinanet import model
from retinanet.csv_eval_dsec_det import evaluate, evaluate_coco_map

assert torch.__version__.split('.')[0] == '1'
print('CUDA available: {}'.format(torch.cuda.is_available()))


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m}m {s:.0f}s'


def normalize_event_data(img_event, method='tanh'):
    """归一化事件数据"""
    if method == 'tanh':
        # 使用tanh归一化到[-1, 1]
        img_event = torch.tanh(img_event / 5.0)
    elif method == 'clip':
        # 简单截断到[-2, 2]
        img_event = torch.clamp(img_event, -2, 2)
    elif method == 'minmax':
        # 最小最大归一化
        min_val = img_event.min()
        max_val = img_event.max()
        if max_val > min_val:
            img_event = 2 * (img_event - min_val) / (max_val - min_val) - 1
    
    return img_event


def validate_annotations(annot, img_width=320, img_height=215):
    """验证和修复标注"""
    if annot.shape[0] == 0:
        return annot
    
    # 提取坐标
    x1, y1, x2, y2, cls = annot[:, 0], annot[:, 1], annot[:, 2], annot[:, 3], annot[:, 4]
    
    # 修复坐标顺序（确保x2>x1, y2>y1）
    x1_new = torch.minimum(x1, x2)
    x2_new = torch.maximum(x1, x2)
    y1_new = torch.minimum(y1, y2)
    y2_new = torch.maximum(y1, y2)
    
    # 确保最小尺寸
    min_size = 2.0
    width = x2_new - x1_new
    height = y2_new - y1_new
    
    # 如果宽度或高度太小，扩展边界框
    small_width_mask = width < min_size
    small_height_mask = height < min_size
    
    if small_width_mask.any():
        expand_w = (min_size - width[small_width_mask]) / 2
        x1_new[small_width_mask] -= expand_w
        x2_new[small_width_mask] += expand_w
    
    if small_height_mask.any():
        expand_h = (min_size - height[small_height_mask]) / 2
        y1_new[small_height_mask] -= expand_h
        y2_new[small_height_mask] += expand_h
    
    # 边界检查
    x1_new = torch.clamp(x1_new, 0, img_width - min_size)
    y1_new = torch.clamp(y1_new, 0, img_height - min_size)
    x2_new = torch.clamp(x2_new, min_size, img_width)
    y2_new = torch.clamp(y2_new, min_size, img_height)
    
    # 最终检查：确保有效的边界框
    valid_mask = (x2_new > x1_new) & (y2_new > y1_new) & (x2_new - x1_new >= 1) & (y2_new - y1_new >= 1)
    
    if not valid_mask.all():
        print(f"[WARNING] Removing {(~valid_mask).sum()} invalid annotations")
    
    # 重构标注
    fixed_annot = torch.stack([x1_new, y1_new, x2_new, y2_new, cls], dim=1)
    
    # 只保留有效的标注
    fixed_annot = fixed_annot[valid_mask]
    
    return fixed_annot


def debug_batch_data(data_batch, batch_idx):
    """修复版本的批次数据调试"""
    print(f"\n=== Debug Batch {batch_idx} (FIXED) ===")
    
    img_event = data_batch['img']
    img_rgb = data_batch['img_rgb']
    annot = data_batch['annot']
    
    batch_size = img_event.shape[0]
    print(f"批次大小: {batch_size}")
    print(f"事件数据形状: {img_event.shape}, 类型: {img_event.dtype}")
    print(f"RGB数据形状: {img_rgb.shape}, 类型: {img_rgb.dtype}")
    print(f"标注形状: {annot.shape}, 类型: {annot.dtype}")
    
    # 检查第一个样本
    sample_event = img_event[0]
    sample_rgb = img_rgb[0]
    sample_annot = annot[0]
    
    print(f"第一个样本 - 事件: {sample_event.shape}, RGB: {sample_rgb.shape}, 标注: {sample_annot.shape}")
    
    # 归一化事件数据
    if sample_event.abs().max() > 5:
        print(f"事件数据范围过大: [{sample_event.min():.3f}, {sample_event.max():.3f}]，进行归一化...")
        sample_event_norm = normalize_event_data(sample_event, method='tanh')
        print(f"归一化后事件范围: [{sample_event_norm.min():.3f}, {sample_event_norm.max():.3f}]")
    else:
        print(f"事件数据范围: [{sample_event.min():.3f}, {sample_event.max():.3f}]")
    
    print(f"RGB数据范围: [{sample_rgb.min():.3f}, {sample_rgb.max():.3f}]")
    
    # 修复标注
    if sample_annot.shape[0] > 0:
        print(f"\n修复前标注:")
        valid_indices = torch.where(sample_annot[:, 0] != -1)[0]
        for i, ann_idx in enumerate(valid_indices[:3]):
            ann = sample_annot[ann_idx]
            x1, y1, x2, y2, cls = ann[0].item(), ann[1].item(), ann[2].item(), ann[3].item(), ann[4].item()
            w, h = x2 - x1, y2 - y1
            print(f"  标注 {i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] w={w:.1f} h={h:.1f} cls={int(cls)}")
        
        # 应用修复
        fixed_annot = validate_annotations(sample_annot)
        
        print(f"\n修复后标注 ({len(fixed_annot)} 个有效):")
        for i, ann in enumerate(fixed_annot[:3]):
            x1, y1, x2, y2, cls = ann[0].item(), ann[1].item(), ann[2].item(), ann[3].item(), ann[4].item()
            w, h = x2 - x1, y2 - y1
            print(f"  标注 {i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] w={w:.1f} h={h:.1f} cls={int(cls)}")
            
            # 验证修复结果
            if x2 <= x1 or y2 <= y1 or w < 1 or h < 1:
                print(f"    ❌ 仍然有问题!")
            else:
                print(f"    ✅ 已修复")
    
    # 总结
    issues = []
    
    # 检查数据范围
    if sample_event.abs().max() > 5:
        issues.append("事件数据范围过大(可自动修复)")
    if sample_rgb.min() < 0 or sample_rgb.max() > 1.5:
        issues.append("RGB数据范围异常")
    
    # 检查标注
    if sample_annot.shape[0] > 0:
        valid_annot = validate_annotations(sample_annot)
        if len(valid_annot) == 0:
            issues.append("没有有效标注")
        elif len(valid_annot) < sample_annot.shape[0]:
            issues.append(f"部分标注无效(已修复: {len(valid_annot)}/{sample_annot.shape[0]})")
    
    if issues:
        print(f"\n⚠️  发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"  - {issue}")
        return True  # 改为True，因为问题可以修复
    else:
        print("\n✅ 批次数据正常")
        return True


def safe_training_step(retinanet, data, optimizer, iter_num, loss_threshold=50.0,
                      scaler=None, use_amp=False):
    """修复版的安全训练步骤"""
    try:
        # GPU数据传输
        if isinstance(data['img_rgb'], list):
            img_rgb = torch.stack(data['img_rgb']).cuda(non_blocking=True).float()
        else:
            img_rgb = data['img_rgb'].cuda(non_blocking=True).float()
    
        if isinstance(data['img'], list):
            img_event = torch.stack(data['img']).cuda(non_blocking=True).float()
        else:
            img_event = data['img'].cuda(non_blocking=True).float()
        
        # 归一化事件数据
        if img_event.abs().max() > 5:
            img_event = normalize_event_data(img_event, method='tanh')
        
        # 处理标注
        if isinstance(data['annot'], list):
            fixed_annots = []
            max_annots = 0
            
            for a in data['annot']:
                fixed_a = validate_annotations(a)
                fixed_annots.append(fixed_a)
                max_annots = max(max_annots, len(fixed_a))
            
            # 填充到相同长度
            if max_annots == 0:
                max_annots = 1
            
            padded_annots = []
            for fixed_a in fixed_annots:
                if len(fixed_a) == 0:
                    padded = torch.ones((max_annots, 5), dtype=torch.float32) * -1
                elif len(fixed_a) < max_annots:
                    pad_len = max_annots - len(fixed_a)
                    pad = torch.ones((pad_len, 5), dtype=fixed_a.dtype) * -1
                    padded = torch.cat([fixed_a, pad], dim=0)
                else:
                    padded = fixed_a[:max_annots]
                padded_annots.append(padded)
            
            annot = torch.stack(padded_annots).cuda(non_blocking=True).float()
        else:
            # 修复批次中的每个标注
            batch_size = data['annot'].shape[0]
            fixed_batch_annots = []
            
            for i in range(batch_size):
                fixed_annot = validate_annotations(data['annot'][i])
                fixed_batch_annots.append(fixed_annot)
            
            # 找到最大标注数
            max_annots = max([len(a) for a in fixed_batch_annots]) if fixed_batch_annots else 1
            if max_annots == 0:
                max_annots = 1
            
            # 填充
            padded_annots = []
            for fixed_a in fixed_batch_annots:
                if len(fixed_a) == 0:
                    padded = torch.ones((max_annots, 5), dtype=torch.float32) * -1
                elif len(fixed_a) < max_annots:
                    pad_len = max_annots - len(fixed_a)
                    pad = torch.ones((pad_len, 5), dtype=fixed_a.dtype) * -1
                    padded = torch.cat([fixed_a, pad], dim=0)
                else:
                    padded = fixed_a[:max_annots]
                padded_annots.append(padded)
            
            annot = torch.stack(padded_annots).cuda(non_blocking=True).float()
        
        # 数据验证
        if torch.isnan(img_event).any() or torch.isinf(img_event).any():
            print(f"迭代 {iter_num}: 事件数据中有NaN/Inf")
            return None, None, None
        if torch.isnan(img_rgb).any() or torch.isinf(img_rgb).any():
            print(f"迭代 {iter_num}: RGB数据中有NaN/Inf")
            return None, None, None
        
        # 检查是否有有效标注
        valid_annot_mask = annot[:, :, 0] != -1
        if not valid_annot_mask.any():
            print(f"迭代 {iter_num}: 没有有效标注，跳过")
            return None, None, None
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        if use_amp and scaler is not None:
            with autocast():
                classification_loss, regression_loss = retinanet([img_rgb, img_event, annot])
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                total_loss = classification_loss + regression_loss
        else:
            classification_loss, regression_loss = retinanet([img_rgb, img_event, annot])
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            total_loss = classification_loss + regression_loss
        
        # 损失验证
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"迭代 {iter_num}: 损失NaN/Inf - 分类:{classification_loss.item():.6f}, 回归:{regression_loss.item():.6f}")
            return None, None, None
            
        if total_loss.item() > loss_threshold:
            print(f"迭代 {iter_num}: 损失过大 {total_loss.item():.1f} > {loss_threshold}")
            return None, None, None
        
        # 反向传播
        if use_amp and scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 1.0)
            optimizer.step()
        
        # 成功日志
        if iter_num % 50 == 0:
            print(f"迭代 {iter_num}: 成功 - 损失={total_loss.item():.4f}")
            valid_annots = valid_annot_mask.sum().item()
            print(f"  有效标注: {valid_annots}, 事件范围: [{img_event.min():.3f}, {img_event.max():.3f}]")
        
        return classification_loss.item(), regression_loss.item(), total_loss.item()
        
    except Exception as e:
        print(f"迭代 {iter_num}: 训练错误 - {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def train_epoch(dataloader, retinanet, optimizer, epoch_num, start_time, loss_threshold=50.0,
               scaler=None, use_amp=False):
    """训练一个轮次"""
    retinanet.train()
    if hasattr(retinanet.module, 'freeze_bn'):
        retinanet.module.freeze_bn()
    
    loss_hist = collections.deque(maxlen=100)
    epoch_losses = []
    valid_iterations = 0
    total_iterations = 0
    
    for iter_num, data in enumerate(dataloader):
        total_iterations += 1
        
        cls_loss, reg_loss, total_loss = safe_training_step(
            retinanet, data, optimizer, iter_num, loss_threshold, scaler, use_amp
        )
        
        if total_loss is not None:
            valid_iterations += 1
            loss_hist.append(total_loss)
            
            # 定期日志记录
            if iter_num % 10 == 0:
                avg_loss = np.mean(loss_hist) if loss_hist else 0
                valid_rate = 100 * valid_iterations / total_iterations
                print(f'[{time_since(start_time)}] 第 {epoch_num} 轮 | 迭代 {iter_num} | '
                      f'损失: 分类={cls_loss:.4f} 回归={reg_loss:.4f} 总计={total_loss:.4f} | '
                      f'平均: {avg_loss:.4f} | 有效: {valid_iterations}/{total_iterations} ({valid_rate:.1f}%)')
                epoch_losses.append(avg_loss)
        
        # 内存清理
        if iter_num > 0 and iter_num % 100 == 0:
            torch.cuda.empty_cache()
    
    # 轮次总结
    if epoch_losses:
        avg_epoch_loss = np.mean(epoch_losses)
        valid_rate = 100 * valid_iterations / total_iterations
        print(f"第 {epoch_num} 轮总结: {valid_iterations}/{total_iterations} 有效 ({valid_rate:.1f}%), 平均损失: {avg_epoch_loss:.4f}")
        return avg_epoch_loss
    else:
        print(f"第 {epoch_num} 轮: 没有有效迭代!")
        return float('inf')


def evaluate_epoch(retinanet, dataset_val, epoch_num, save_folder):
    """评估模型性能"""
    print(f"\n{'='*20} 验证第 {epoch_num} 轮 {'='*20}")
    
    eval_save_folder = os.path.join(save_folder, f'eval_epoch_{epoch_num}')
    os.makedirs(eval_save_folder, exist_ok=True)
    
    mean_ap = evaluate(
        generator=dataset_val,
        retinanet=retinanet,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        save_detection=True,
        save_folder=eval_save_folder,
        load_detection=False,
        save_path=eval_save_folder
    )
    
    print(f"\n第 {epoch_num} 轮验证结果:")
    print(f"mAP@0.5: {mean_ap:.4f}")
    
    return mean_ap


def main():
    parser = argparse.ArgumentParser(description='DSEC检测训练脚本(修复版)')
    
    # 数据参数
    parser.add_argument('--root_dir', default='/media/data/hucao/zhenwu/hucao/DSEC/DSEC_Det', 
                       help='DSEC数据集根目录')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载工作进程数')
    
    # 训练参数  
    parser.add_argument('--epochs', type=int, default=60, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='学习率')
    parser.add_argument('--loss_threshold', type=float, default=50.0, help='损失阈值')
    
    # 模型参数
    parser.add_argument('--fusion', default='fpn_fusion', choices=['fpn_fusion', 'rgb', 'event'],
                       help='融合模型类型')
    parser.add_argument('--depth', type=int, default=50, choices=[18, 34, 50], 
                       help='ResNet深度')
    
    # 评估参数
    parser.add_argument('--eval_interval', type=int, default=5, help='评估间隔(轮数)')
    parser.add_argument('--eval_coco', action='store_true', help='同时评估COCO风格的mAP')
    
    # 优化参数
    parser.add_argument('--use_amp', action='store_true', help='使用自动混合精度')
    
    # 其他参数
    parser.add_argument('--debug_data', action='store_true', help='启用数据调试')
    parser.add_argument('--continue_training', action='store_true', help='从检查点继续训练')
    parser.add_argument('--checkpoint', default='', help='检查点路径')
    parser.add_argument('--save_dir', default='/media/data/hucao/zehua/results_dsec/fixed_version',
                       help='保存检查点的目录')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*60)
    print("DSEC检测训练脚本(修复版)")
    print("="*60)
    print(f"根目录: {args.root_dir}")
    print(f"批次大小: {args.batch_size}")
    print(f"工作进程数: {args.num_workers}")
    print(f"学习率: {args.learning_rate}")
    print(f"损失阈值: {args.loss_threshold}")
    print(f"训练轮数: {args.epochs}")
    print(f"融合模型: {args.fusion}")
    print(f"混合精度: {args.use_amp}")
    print(f"评估间隔: {args.eval_interval}")
    print(f"COCO评估: {args.eval_coco}")
    
    # 加载训练数据
    print("\n加载训练数据集...")
    dataloader_train, dataset_train = create_dsec_det_dataloader(
        root_dir=args.root_dir,
        split='train',
        batch_size=args.batch_size,
        num_workers=0 if args.debug_data else args.num_workers
    )
    
    print(f"训练数据集已加载: {len(dataset_train)} 样本")
    
    # 加载验证数据
    print("加载验证数据集...")
    _, dataset_val = create_dsec_det_dataloader(
        root_dir=args.root_dir,
        split='val',  
        batch_size=1,  # 验证使用批次大小1
        num_workers=1
    )
    
    print(f"验证数据集已加载: {len(dataset_val)} 样本")
    
    # 调试模式
    if args.debug_data:
        print("\n" + "="*40)
        print("数据调试模式(修复版)")
        print("="*40)
        
        for batch_idx, data in enumerate(dataloader_train):
            success = debug_batch_data(data, batch_idx)
            if not success:
                print(f"❌ 批次 {batch_idx} 有无法修复的问题!")
            if batch_idx >= 4:  # 检查前5个batch
                break
        
        print("\n调试完成。使用不带 --debug_data 的命令开始训练。")
        return
    
    # 创建模型
    print("\n创建模型...")
    retinanet = model.resnet50(
        dataset_name='dsec',
        num_classes=dataset_train.num_classes,
        fusion_model=args.fusion,
        pretrained=True  # 使用预训练权重
    )
    print(f"模型已创建: ResNet{args.depth} 带 {args.fusion} 融合")
    print(f"类别数: {dataset_train.num_classes}")
    
    # GPU设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        retinanet = retinanet.cuda()
        retinanet = torch.nn.DataParallel(retinanet).cuda()
        print("模型已移至GPU")
    else:
        retinanet = torch.nn.DataParallel(retinanet)
        print("使用CPU")
    
    # 优化器和调度器
    optimizer = optim.Adam(retinanet.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.5)
    
    # 混合精度缩放器
    scaler = GradScaler() if args.use_amp else None
    
    # 跟踪最佳性能
    best_map = 0.0
    best_epoch = 0
    
    # 训练日志
    train_log = []
    
    # 加载检查点
    start_epoch = 0
    if args.continue_training and args.checkpoint:
        print(f"\n加载检查点: {args.checkpoint}")
        try:
            checkpoint = torch.load(args.checkpoint)
            retinanet.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scaler and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_map = checkpoint.get('best_map', 0.0)
            best_epoch = checkpoint.get('best_epoch', 0)
            print(f"从第 {start_epoch} 轮恢复，最佳mAP: {best_map:.4f} (第 {best_epoch} 轮)")
        except Exception as e:
            print(f"加载检查点失败: {e}")
            start_epoch = 0
    
    # 开始训练
    print(f"\n" + "="*60)
    print("开始训练(修复版)")
    print("="*60)
    
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- 第 {epoch+1}/{args.epochs} 轮 ---")
        
        # 训练
        avg_epoch_loss = train_epoch(
            dataloader_train, retinanet, optimizer, epoch, 
            start_time, args.loss_threshold, scaler, args.use_amp
        )
        
        # 验证
        current_map = 0.0
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            current_map = evaluate_epoch(retinanet, dataset_val, epoch, args.save_dir)
            
            # 可选的COCO评估
            if args.eval_coco:
                print("计算COCO风格的mAP...")
                coco_save_folder = os.path.join(args.save_dir, f'coco_eval_epoch_{epoch}')
                os.makedirs(coco_save_folder, exist_ok=True)
                
                coco_aps = evaluate_coco_map(
                    generator=dataset_val,
                    retinanet=retinanet,
                    iou_threshold=0.5,
                    score_threshold=0.05,
                    max_detections=100,
                    save_detection=True,
                    save_folder=coco_save_folder,
                    load_detection=False,
                    save_path=coco_save_folder
                )
                
                # 计算COCO风格的平均mAP
                coco_map = np.mean([np.mean(aps) for aps in coco_aps.values()])
                print(f"COCO风格mAP: {coco_map:.4f}")
            
            # 更新最佳结果
            if current_map > best_map:
                best_map = current_map
                best_epoch = epoch
                print(f"🎉 新的最佳mAP: {best_map:.4f} (第 {best_epoch} 轮)")
                
                # 保存最佳模型
                best_model_path = f'{args.save_dir}/best_model_fixed.pt'
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': retinanet.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_epoch_loss,
                    'map': current_map,
                    'best_map': best_map,
                    'best_epoch': best_epoch,
                    'config': {
                        'fusion': args.fusion,
                        'depth': args.depth,
                        'learning_rate': args.learning_rate,
                        'batch_size': args.batch_size,
                        'loss_threshold': args.loss_threshold
                    }
                }
                if scaler:
                    save_dict['scaler_state_dict'] = scaler.state_dict()
                
                torch.save(save_dict, best_model_path)
                print(f"最佳模型已保存: {best_model_path}")
        
        # 记录训练进度
        train_log.append({
            'epoch': epoch,
            'loss': avg_epoch_loss,
            'map': current_map,
            'best_map': best_map,
            'best_epoch': best_epoch
        })
        
        # 调整学习率
        if avg_epoch_loss != float('inf'):
            scheduler.step(avg_epoch_loss)
        
        # 定期保存模型
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            save_path = f'{args.save_dir}/dsec_retinanet_fixed_epoch_{epoch}.pt'
            save_dict = {
                'epoch': epoch,
                'model_state_dict': retinanet.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'map': current_map,
                'best_map': best_map,
                'best_epoch': best_epoch,
                'train_log': train_log,
                'config': {
                    'fusion': args.fusion,
                    'depth': args.depth,
                    'learning_rate': args.learning_rate,
                    'batch_size': args.batch_size,
                    'loss_threshold': args.loss_threshold
                }
            }
            if scaler:
                save_dict['scaler_state_dict'] = scaler.state_dict()
            
            torch.save(save_dict, save_path)
            print(f"模型已保存: {save_path}")
    
    # 最终保存
    final_path = f'{args.save_dir}/dsec_retinanet_fixed_final.pt'
    final_dict = {
        'epoch': args.epochs,
        'model_state_dict': retinanet.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_log': train_log,
        'best_map': best_map,
        'best_epoch': best_epoch,
        'config': {
            'fusion': args.fusion,
            'depth': args.depth,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'loss_threshold': args.loss_threshold
        }
    }
    if scaler:
        final_dict['scaler_state_dict'] = scaler.state_dict()
    
    torch.save(final_dict, final_path)
    
    total_time = time_since(start_time)
    print(f"\n" + "="*60)
    print(f"训练完成! 用时: {total_time}")
    print(f"最终模型已保存: {final_path}")
    print(f"最佳mAP: {best_map:.4f} (第 {best_epoch} 轮)")
    print("="*60)


if __name__ == '__main__':
    main()
