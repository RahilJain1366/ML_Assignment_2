"""
CS 6375 Homework 2 Programming
Implement the compute_loss() function in this python script
"""
import os
import torch
import torch.nn as nn


# compute Intersection over Union (IoU) of two bounding boxes
# the input bounding boxes are in (cx, cy, w, h) format
def compute_iou(pred, gt):
    x1p = pred[0] - pred[2] * 0.5
    x2p = pred[0] + pred[2] * 0.5
    y1p = pred[1] - pred[3] * 0.5
    y2p = pred[1] + pred[3] * 0.5
    areap = (x2p - x1p + 1) * (y2p - y1p + 1)    
    
    x1g = gt[0] - gt[2] * 0.5
    x2g = gt[0] + gt[2] * 0.5
    y1g = gt[1] - gt[3] * 0.5
    y2g = gt[1] + gt[3] * 0.5
    areag = (x2g - x1g + 1) * (y2g - y1g + 1)

    xx1 = max(x1p, x1g)
    yy1 = max(y1p, y1g)
    xx2 = min(x2p, x2g)
    yy2 = min(y2p, y2g)

    w = max(0.0, xx2 - xx1 + 1)
    h = max(0.0, yy2 - yy1 + 1)
    inter = w * h
    iou = inter / (areap + areag - inter)    
    return iou

# TO DO: finish the implementation of this loss function for YOLO training
# output: (batch_size, num_boxes * 5 + num_classes, 7, 7), raw output from the network
# pred_box: (batch_size, num_boxes * 5 + num_classes, 7, 7), predicted bounding boxes from the network (see the forward() function)
# gt_box: (batch_size, 5, 7, 7), ground truth bounding box target from the dataloader
# gt_mask: (batch_size, 7, 7), ground truth bounding box mask from the dataloader
# num_boxes: number of bounding boxes per cell
# num_classes: number of object classes for detection
# grid_size: YOLO grid size, 64 in our case
# image_size: YOLO image size, 448 in our case
def compute_loss(output, pred_box, gt_box, gt_mask, num_boxes, num_classes, grid_size, image_size):
    batch_size = output.shape[0]
    num_grids = output.shape[2]
    
    # Initialize masks for box assignment
    box_mask = torch.zeros(batch_size, num_boxes, num_grids, num_grids, device=output.device)
    box_confidence = torch.zeros(batch_size, num_boxes, num_grids, num_grids, device=output.device)

    # Compute assignment of predicted bounding boxes
    for i in range(batch_size):
        for j in range(num_grids):
            for k in range(num_grids):
                if gt_mask[i, j, k] > 0:
                    # Transform gt box
                    gt = gt_box[i, :, j, k].clone()
                    gt[0] = gt[0] * grid_size + k * grid_size
                    gt[1] = gt[1] * grid_size + j * grid_size
                    gt[2] = gt[2] * image_size
                    gt[3] = gt[3] * image_size

                    select = 0
                    max_iou = -1
                    # Select box with maximum IoU
                    for b in range(num_boxes):
                        pred = pred_box[i, 5*b:5*b+4, j, k].clone()
                        iou = compute_iou(gt, pred)
                        if iou > max_iou:
                            max_iou = iou
                            select = b
                    
                    box_mask[i, select, j, k] = 1
                    box_confidence[i, select, j, k] = max_iou

    # Adjust loss weights - you might want to experiment with these
    weight_coord = 5.0  # Try lower values like 2.0 or 1.0
    weight_noobj = 0.5  # Try lower values like 0.1

    losses = {
        "x": 0, "y": 0, "w": 0, "h": 0, "obj": 0, "noobj": 0, "cls": 0
    }

    for b in range(num_boxes):
        base = 5 * b
        mask = box_mask[:, b, :, :]

        # Extract predictions and targets
        pred = output[:, base:base+5, :, :]
        true = gt_box[:, :4, :, :].clone()  # Use gt_box directly instead of pred_box
        
        # Only compute loss where objects exist
        masked_cells = mask.sum().item()
        if masked_cells > 0:
            # Coordinate losses - only apply to cells with objects
            losses["x"] += torch.sum(mask * (pred[:, 0] - true[:, 0]) ** 2)
            losses["y"] += torch.sum(mask * (pred[:, 1] - true[:, 1]) ** 2)
            
            # Square root scaling for width and height (as in the paper)
            pred_w = torch.sqrt(torch.abs(pred[:, 2]) + 1e-6)
            pred_h = torch.sqrt(torch.abs(pred[:, 3]) + 1e-6)
            true_w = torch.sqrt(torch.abs(true[:, 2]) + 1e-6)
            true_h = torch.sqrt(torch.abs(true[:, 3]) + 1e-6)
            
            losses["w"] += torch.sum(mask * (pred_w - true_w) ** 2)
            losses["h"] += torch.sum(mask * (pred_h - true_h) ** 2)
            
            # Confidence losses
            conf_target = box_confidence[:, b, :, :]
            losses["obj"] += torch.sum(mask * (pred[:, 4] - conf_target) ** 2)
        
        # No-object confidence loss
        losses["noobj"] += torch.sum((1 - mask) * pred[:, 4] ** 2)

    # Classification loss
    if num_classes > 0:
        pred_cls = output[:, 5 * num_boxes:, :, :]
        target_cls = torch.zeros_like(pred_cls)
        
        for i in range(batch_size):
            for j in range(num_grids):
                for k in range(num_grids):
                    if gt_mask[i, j, k] > 0:
                        class_idx = int(gt_box[i, 4, j, k])
                        if 0 <= class_idx < num_classes:
                            target_cls[i, class_idx, j, k] = 1.0
        
        # Apply classification loss only where objects exist
        cls_mask = gt_mask.unsqueeze(1).expand(-1, num_classes, -1, -1)
        losses["cls"] += torch.sum(cls_mask * (pred_cls - target_cls) ** 2)

    # Calculate total loss with weights
    total_loss = (
        weight_coord * (losses["x"] + losses["y"] + losses["w"] + losses["h"])
        + losses["obj"]
        + weight_noobj * losses["noobj"]
        + losses["cls"]
    )

    '''# For debugging, returning individual loss components
    loss_components = {
        "total": total_loss.item(),
        "x": losses["x"].item(), 
        "y": losses["y"].item(),
        "w": losses["w"].item(), 
        "h": losses["h"].item(),
        "obj": losses["obj"].item(), 
        "noobj": losses["noobj"].item(),
        "cls": losses["cls"].item()
    }
    
    print(f"Loss components: {loss_components}")'''
    
    return total_loss