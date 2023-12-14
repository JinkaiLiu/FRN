import argparse
import collections

import numpy as np
import time
import math

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CSVDataset_event, collater, Resizer, AspectRatioBasedSampler, \
    Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

# from retinanet import coco_eval
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1' #确定pytorch的版本可用

print('CUDA available: {}'.format(torch.cuda.is_available()))


def time_since(since):
    now = time.time() #当前时间，秒
    s = now - since #耗时，秒
    m = math.floor(s / 60)  #耗时，分
    s -= m * 60
    return '%dm %ds' % (m, s)


def main(args=None):
    #base_dir = '/home/abhishek/connect' #路径要改
    base_dir = '/media/group2/data/hucao/PKU-DDD17_all'
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', default='csv', help='Dataset type, must be one of csv or coco.') #数据格式为coco或者csv
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', default=f'/media/group2/data/hucao/PKU-DDD17_all/annotations_CSV/labels_filtered_train.csv',
                        help='Path to file containing training annotations (see readme)') #训练集的标注文件
    parser.add_argument('--csv_classes', default=f'/media/group2/data/hucao/PKU-DDD17_all/annotations_CSV/labels_filtered_map.csv', #标注的类别说明文件
                        help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)') #验证集的标注文件（缺少路径）
    parser.add_argument('--root_img',default=f'{base_dir}/images/train/aps_images',help='dir to root rgb images') #rgb图片根目录，注意是转化到event frame的images
    parser.add_argument('--root_event', default=f'{base_dir}/images/train/dvs_events',help='dir to toot event files in dsec directory structure')#事件根目录
    parser.add_argument('--fusion', help='Type of fusion:1)early_fusion, fpn_fusion, multi-level', type=str, default='fpn_fusion') #融合的类型：1)early_fusion, fpn_fusion, multi-level
    #传递不同的融合类型，可以是4种任意，前两种要两种数据融合，event/rgb是一种数据
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=34_50) #resnet的深度18, 34, 50, 101, 152（不同种的resnet）
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=200) #训练周期，默认60(遍历数据集的次数)
    parser.add_argument('--continue_training', help='load a pretrained file', default=False) #继续训练，默认关闭
    parser.add_argument('--checkpoint', help='location of pretrained file', default='./csv_dropout_retinanet_63.pt') #预训练文件位置


    parser = parser.parse_args(args)

    if parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')
        
        
        dataset_train = CSVDataset_event(train_file=parser.csv_train, class_list=parser.csv_classes,root_event_dir=parser.root_event,root_img_dir=parser.root_img,
                                         transform=transforms.Compose([Normalizer(), Resizer()]))
        #train_file (string): CSV file with training annotations
        #annotations (string): CSV file with class list
        #test_file (string, optional): CSV file with testing annotations

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset_event(train_file=parser.csv_val, class_list=parser.csv_classes, #原文这里是CSVDataset()
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    # sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, batch_size=8, num_workers=6, shuffle=True,collate_fn=collater) #num_worker是线程数，和cpu有关；collate_fn接收自定义collate函数，该函数在数据加载（即通过Dataloader取一个batch数据）之前，定义对每个batch数据的处理行为。
    dataset_val1 = CSVDataset_event(train_file=f'{base_dir}/DSEC_detection_labels/labels_filtered_test_all.csv', class_list=parser.csv_classes,
                                    root_event_dir=parser.root_event,root_img_dir=parser.root_img, transform=transforms.Compose([Normalizer(), Resizer()]))
    dataloader_val1 = DataLoader(dataset_val1 , batch_size=1, num_workers=6, shuffle=True,collate_fn=collater)

#DataLoader目的：将自定义的Dataset根据batch size大小、是否shuffle等封装成一个Batch Size大小的Tensor，用于后面的训练。

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)#该函数的目的是为给DataLoader()函数传递batch_sample参数
        dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)
        #sampler(Sampler, optional): 自定义从数据集中取样本的策略，如果指定这个参数，那么shuffle必须为False
        #batch_sampler(Sampler, optional): 与sampler类似，但是一次只返回一个batch的indices（索引），需要注意的是，一旦指定了这个参数，那么batch_size,shuffle,sampler,drop_last就不能再制定了（互斥——Mutually exclusive）

    # Create the model
    list_models = ['early_fusion','fpn_fusion', 'event', 'rgb']
    if parser.fusion in  list_models:
        if parser.depth == 50:
            retinanet = model.resnet50(num_classes=dataset_train.num_classes(),fusion_model=parser.fusion,pretrained=False)
        if parser.depth == 101:
            retinanet = model.resnet101(num_classes=dataset_train.num_classes(),fusion_model=parser.fusion,pretrained=False)
        if parser.depth == 34_50:
            retinanet = model.resnet34_50(num_classes=dataset_train.num_classes(),fusion_model=parser.fusion,pretrained=False)
    else:
        raise ValueError('Unsupported model fusion')

    use_gpu = True
    if parser.continue_training:
        checkpoint = torch.load(parser.checkpoint)
        retinanet.load_state_dict(checkpoint['model_state_dict'])
        epoch_loss_all = checkpoint['loss']
        epoch_total = checkpoint['epoch']
        print('training sensor fusion model')
        retinanet.eval()
    else:
        epoch_total = 0
        epoch_loss_all =[]
    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda() #记得添加设备号，只能用两个GPU
        #model = nn.DataParallel(model,device_ids=[1,2])
        #device = torch.device("cuda:1" )
        #model.to(device)
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True #训练模式

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True) #提供了多种基于epoch训练次数进行学习率调整的方法;

    loss_hist = collections.deque(maxlen=100) #Deque 可以从队列 两端添加 (append) 和 弹出 (pop) 元素

    retinanet.train()
    retinanet.module.freeze_bn() #将网络结构中的BN层冻结

    print('Num training images: {}'.format(len(dataset_train)))
    num_batches = 0
    start = time.time()
    # print('sensor fusion, impulse_noise images')
    # mAP = csv_eval.evaluate(dataset_val1, retinanet)
    print(time_since(start))
    epoch_loss = []
    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()
        epoch_total += 1 
        for iter_num, data in enumerate(dataloader_train):
            try:
                classification_loss, regression_loss = retinanet([data['img_rgb'],data['img'].cuda().float(),data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue #会终止执行本次循环中剩下的代码，直接从下一次循环继续执行。

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1) #梯度剪裁

                num_batches += 1
                if num_batches == 2:  # optimize every 5 mini-batches
                    optimizer.step() #更新所有的参数
                    optimizer.zero_grad() #梯度初始化为零，把loss关于weight的导数变成0
                    num_batches = 0


                loss_hist.append(float(loss)) #将loss放到一个列表中

                
                if iter_num % 50 ==0:
                #if iter_num % 50 ==0:
                    print(
                        '[sensor fusion homographic] [{}], Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            time_since(start), epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
                    epoch_loss.append(np.mean(loss_hist))

                del classification_loss
                del regression_loss
            except Exception as e: #检查异常
                print(e)
                continue

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            print('Evaluating dataset')

            mAP = csv_eval.evaluate(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))

        # torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))
        if epoch_num % 10 ==0:
            torch.save({'epoch': epoch_total, 'model_state_dict': retinanet.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': np.append(epoch_loss_all,epoch_loss)}, f'/media/group1/data/yuanhaozhong/DDD17/baseline3450_p2p6/{parser.dataset}_fpn_homographic_retinanet_retinanet101_{epoch_total}.pt')

    # retinanet.eval()

    torch.save({'epoch': epoch_total, 'model_state_dict': retinanet.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': np.append(epoch_loss_all,epoch_loss)}, f'/media/group1/data/yuanhaozhong/DDD17/baseline3450_p2p6/{parser.dataset}_fpn_homographic_retinanet_retinanet101_{epoch_total}.pt')


if __name__ == '__main__':
    main()
