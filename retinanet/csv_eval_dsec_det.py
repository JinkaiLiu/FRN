from __future__ import print_function

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import torch
import pickle

def compute_overlap(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
    ua = np.maximum(ua, np.finfo(float).eps)
    intersection = iw * ih

    return intersection / ua

def _compute_ap(recall, precision):
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def _get_detections(dataset, retinanet, score_threshold=0.05, max_detections=100, save_path=None):
    all_detections = [[None for i in range(dataset.num_classes())] for j in range(len(dataset))]

    retinanet.eval()
    
    with torch.no_grad():
        for index in range(len(dataset)):
            try:
                data = dataset[index]
                scale = data.get('scale', 1.0)

                # Get data from sample
                img_rgb = data['img_rgb']
                img_event = data['img']
                
                # Process RGB image format
                if isinstance(img_rgb, torch.Tensor):
                    if len(img_rgb.shape) == 3:  # HWC format
                        if img_rgb.shape[-1] == 3:  # Confirm HWC format
                            img_rgb = img_rgb.permute(2, 0, 1)  # HWC -> CHW
                        img_rgb = img_rgb.unsqueeze(0)  # CHW -> BCHW
                    elif len(img_rgb.shape) == 4:  # BHWC format
                        if img_rgb.shape[-1] == 3:
                            img_rgb = img_rgb.permute(0, 3, 1, 2)  # BHWC -> BCHW
                else:
                    # Convert numpy array to tensor
                    if len(img_rgb.shape) == 3 and img_rgb.shape[-1] == 3:
                        img_rgb = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)
                    else:
                        img_rgb = torch.from_numpy(img_rgb).unsqueeze(0)
                
                # Process event image format
                if isinstance(img_event, torch.Tensor):
                    if len(img_event.shape) == 3:  # CHW format
                        img_event = img_event.unsqueeze(0)  # CHW -> BCHW
                    elif len(img_event.shape) == 4:  # BCHW format - already correct
                        pass
                else:
                    # Convert numpy array to tensor
                    img_event = torch.from_numpy(img_event)
                    if len(img_event.shape) == 3:
                        img_event = img_event.unsqueeze(0)
                
                # Move to correct device
                if torch.cuda.is_available():
                    img_rgb = img_rgb.cuda().float()
                    img_event = img_event.cuda().float()
                else:
                    img_rgb = img_rgb.float()
                    img_event = img_event.float()
                
                # Model inference
                scores, labels, boxes = retinanet([img_rgb, img_event])
                
                # Convert back to CPU
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()
                boxes = boxes.cpu().numpy()

                # Scale adjustment
                if isinstance(scale, (list, tuple)):
                    boxes[:, [0, 2]] /= scale[0]  # x coordinates
                    boxes[:, [1, 3]] /= scale[1]  # y coordinates
                else:
                    boxes /= scale
                
                # Filter detection results
                indices = np.where(scores > score_threshold)[0]
                if indices.shape[0] > 0:
                    scores = scores[indices]
                    scores_sort = np.argsort(-scores)[:max_detections]

                    image_boxes = boxes[indices[scores_sort], :]
                    image_scores = scores[scores_sort]
                    image_labels = labels[indices[scores_sort]]
                    image_detections = np.concatenate([
                        image_boxes, 
                        np.expand_dims(image_scores, axis=1), 
                        np.expand_dims(image_labels, axis=1)
                    ], axis=1)

                    # Group by class
                    for label in range(dataset.num_classes()):
                        all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
                else:
                    # No detections found
                    for label in range(dataset.num_classes()):
                        all_detections[index][label] = np.zeros((0, 5))

                if (index + 1) % 100 == 0:
                    print(f'Processing: {index + 1}/{len(dataset)}')
                    
            except Exception as e:
                print(f"Error processing sample {index}: {e}")
                # Create empty detection results
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = np.zeros((0, 5))
                continue

    return all_detections

def _get_annotations(generator):
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(len(generator))]

    for i in range(len(generator)):
        try:
            sample = generator[i]
            annotations = sample['annot']
            
            # Ensure annotations is numpy array
            if isinstance(annotations, torch.Tensor):
                annotations = annotations.numpy()
            
            # Filter valid annotations (exclude padded -1 values)
            if len(annotations) > 0:
                valid_mask = annotations[:, 4] >= 0
                valid_annotations = annotations[valid_mask]
            else:
                valid_annotations = np.zeros((0, 5))
            
            # Group annotations by class
            for label in range(generator.num_classes()):
                if len(valid_annotations) > 0:
                    class_annotations = valid_annotations[valid_annotations[:, 4] == label, :4].copy()
                    all_annotations[i][label] = class_annotations
                else:
                    all_annotations[i][label] = np.zeros((0, 4))
                    
        except Exception as e:
            print(f"Error loading annotations for sample {i}: {e}")
            # Create empty annotations
            for label in range(generator.num_classes()):
                all_annotations[i][label] = np.zeros((0, 4))

        if (i + 1) % 100 == 0:
            print(f'Loading annotations: {i + 1}/{len(generator)}')

    return all_annotations

def evaluate_coco_map(
    generator,
    retinanet,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_detection=True,
    save_folder='./',
    load_detection=False,
    save_path=None
):
    os.makedirs(save_folder, exist_ok=True)
    detections_file = os.path.join(save_folder, 'detections.pkl')
    annotations_file = os.path.join(save_folder, 'annotations.pkl')

    if load_detection and os.path.exists(detections_file) and os.path.exists(annotations_file):
        print("Loading cached detections and annotations...")
        with open(detections_file, "rb") as fp:  
            all_detections = pickle.load(fp)
        with open(annotations_file, "rb") as fp: 
            all_annotations = pickle.load(fp)
    else:
        print("Computing detections and annotations...")
        all_detections = _get_detections(generator, retinanet, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
        all_annotations = _get_annotations(generator)
        
        if save_detection:
            with open(detections_file, "wb") as fp:
                pickle.dump(all_detections, fp)
            with open(annotations_file, "wb") as fp:
                pickle.dump(all_annotations, fp)

    # Compute AP at different IoU thresholds
    average_precisions = {}
    average_precisions_coco = {}
    for label in range(generator.num_classes()):
        average_precisions_coco[label] = []

    iou_values = np.arange(0.5, 1.00, 0.05).tolist()

    for label in range(generator.num_classes()):
        false_positives = []
        true_positives = []
        for idx, iou_threshold1 in enumerate(iou_values):
            false_positives.append(np.zeros((0,)))
            true_positives.append(np.zeros((0,)))

        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(generator)):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []
            for idx, iou_threshold1 in enumerate(iou_values):
                detected_annotations.append([])

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    for idx, iou_threshold1 in enumerate(iou_values):
                        false_positives[idx] = np.append(false_positives[idx], 1)
                        true_positives[idx] = np.append(true_positives[idx], 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                for idx, iou_threshold1 in enumerate(iou_values):
                    if max_overlap >= iou_threshold1 and assigned_annotation not in detected_annotations[idx]:
                        false_positives[idx] = np.append(false_positives[idx], 0)
                        true_positives[idx] = np.append(true_positives[idx], 1)
                        detected_annotations[idx].append(assigned_annotation)
                    else:
                        false_positives[idx] = np.append(false_positives[idx], 1)
                        true_positives[idx] = np.append(true_positives[idx], 0)

        if num_annotations == 0:
            average_precisions[label] = 0, 0
            for idx, _ in enumerate(iou_values):
                average_precisions_coco[label].append(0.0)
            continue

        indices = np.argsort(-scores)
        for idx, iou_threshold1 in enumerate(iou_values):
            false_positives[idx] = false_positives[idx][indices]
            true_positives[idx] = true_positives[idx][indices]

            false_positives[idx] = np.cumsum(false_positives[idx])
            true_positives[idx] = np.cumsum(true_positives[idx])

            recall = true_positives[idx] / num_annotations
            precision = true_positives[idx] / np.maximum(true_positives[idx] + false_positives[idx], np.finfo(np.float64).eps)

            average_precision = _compute_ap(recall, precision)
            average_precisions[label] = average_precision, num_annotations
            average_precisions_coco[label].append(average_precision)

        label_name = generator.label_to_name(label)
        
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            plt.figure()
            plt.plot(recall, precision)
            plt.xlabel('Recall') 
            plt.ylabel('Precision') 
            plt.title(f'Precision Recall curve - {label_name}') 
            plt.savefig(os.path.join(save_path, f'{label_name}_precision_recall.jpg'))
            plt.close()

    return average_precisions_coco

def evaluate(
    generator,
    retinanet,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_detection=True,
    save_folder='./',
    load_detection=False,
    save_path=None
):
    os.makedirs(save_folder, exist_ok=True)
    detections_file = os.path.join(save_folder, 'detections.pkl')
    annotations_file = os.path.join(save_folder, 'annotations.pkl')

    if load_detection and os.path.exists(detections_file) and os.path.exists(annotations_file):
        print("Loading cached detections and annotations...")
        with open(detections_file, "rb") as fp:  
            all_detections = pickle.load(fp)
        with open(annotations_file, "rb") as fp: 
            all_annotations = pickle.load(fp)
    else:
        print("Computing detections and annotations...")
        all_detections = _get_detections(generator, retinanet, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
        all_annotations = _get_annotations(generator)
        
        if save_detection:
            with open(detections_file, "wb") as fp:
                pickle.dump(all_detections, fp)
            with open(annotations_file, "wb") as fp:
                pickle.dump(all_annotations, fp)

    average_precisions = {}

    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(generator)):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

        label_name = generator.label_to_name(label)
        print(f'{label_name}: mAP = {average_precision:.4f}, num_annotations = {int(num_annotations)}')
        
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            plt.figure()
            plt.plot(recall, precision)
            plt.xlabel('Recall') 
            plt.ylabel('Precision') 
            plt.title(f'Precision Recall curve - {label_name}') 
            plt.savefig(os.path.join(save_path, f'{label_name}_precision_recall.jpg'))
            plt.close()

    print('\nOverall mAP:')
    mAP_values = [ap[0] for ap in average_precisions.values()]
    total_map = np.mean(mAP_values) if mAP_values else 0.0
    print(f'mAP@0.5: {total_map:.4f}')

    return total_map
            plt.title(f'Precision Recall curve - {label_name}') 
            plt.savefig(os.path.join(save_path, f'{label_name}_precision_recall.jpg'))
            plt.close()

    print('\nOverall mAP:')
    mAP_values = [ap[0] for ap in average_precisions.values()]
    total_map = np.mean(mAP_values) if mAP_values else 0.0
    print(f'mAP@0.5: {total_map:.4f}')

    return total_map