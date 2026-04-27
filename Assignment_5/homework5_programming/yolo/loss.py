"""
CS 6384 Homework 5 Programming
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

# TODO: finish the implementation of this loss function for YOLO training
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
    # compute mask with shape (batch_size, num_boxes, 7, 7) for box assignment
    box_mask = torch.zeros(batch_size, num_boxes, num_grids, num_grids)
    box_confidence = torch.zeros(batch_size, num_boxes, num_grids, num_grids)

    # compute assignment of predicted bounding boxes for ground truth bounding boxes
    for i in range(batch_size):
        for j in range(num_grids):
            for k in range(num_grids):
 
                # if the gt mask is 1
                if gt_mask[i, j, k] > 0:
                    # transform gt box
                    gt = gt_box[i, :, j, k].clone()
                    gt[0] = gt[0] * grid_size + k * grid_size
                    gt[1] = gt[1] * grid_size + j * grid_size
                    gt[2] = gt[2] * image_size
                    gt[3] = gt[3] * image_size
                    # print('gt in loss %.2f, %.2f, %.2f, %.2f' % (gt[0], gt[1], gt[2], gt[3]))

                    select = 0
                    max_iou = -1
                    # select the one with maximum IoU
                    for b in range(num_boxes):
                        # center x, y and width, height
                        pred = pred_box[i, 5*b:5*b+4, j, k].clone()
                        iou = compute_iou(gt, pred)
                        if iou > max_iou:
                            max_iou = iou
                            select = b
                    box_mask[i, select, j, k] = 1
                    box_confidence[i, select, j, k] = max_iou
                    print('select box %d with iou %.2f' % (select, max_iou))

    # compute yolo loss
    weight_coord = 5.0
    weight_noobj = 0.5

    # according to the YOLO paper, we compute the following losses
    # loss_x: loss function on x coordinate (cx)
    # loss_y: loss function on y coordinate (cy)
    # loss_w: loss function on width 
    # loss_h: loss function on height
    # loss_obj: loss function on confidence for objects
    # loss_nonobj: loss function on confidence for non-objects
    # loss_cls: loss function for object class

    # This is implementation for the loss_obj
    # Follow this example to compute other losses
    loss_obj = torch.sum(box_mask * torch.pow(box_confidence - output[:, 4:5*num_boxes:5], 2.0))

    ### ADD YOUR CODE HERE ###
    # Use weight_coord and weight_noobj defined above

    # ------------- Slice the network output into per-box prediction channels -------------
    # Each box uses 5 channels: [cx, cy, w, h, conf]; classes are at the tail
    # Predicted (cx, cy, w, h) for box b are at output[:, 5b:5b+4, :, :]
    # Slicing with stride 5 gives shape (batch_size, num_boxes, 7, 7)
    pred_cx = output[:, 0:5*num_boxes: 5] # (B, num_boxes,7,7)
    pred_cy = output[:, 1:5*num_boxes: 5]
    pred_w = output[:, 2:5*num_boxes: 5]
    pred_h = output[:, 3:5*num_boxes: 5]
    pred_conf = output[:, 4:5*num_boxes: 5]

    # Ground truth (cx, cy, w, h) live in gt_box channels 0-3 with shape (B,7,7)
    # Add a singleton box dimension so they broadcast tagainst the (B, num_boxes, 7,7) predicitons
    # box_mask handles which predictor is responsible for which gt box, and also which cells contain objects
    gt_cx = gt_box[:, 0:1] # (B,1,7,7)
    gt_cy = gt_box[:, 1:2]
    gt_w = gt_box[:, 2:3]
    gt_h = gt_box[:, 3:4]

    # ------------- Coordinate losses -------------
    # lambda_coord * sum_{i,j} 1^{obj}_{ij} (x -xHhat)^2
    loss_x = weight_coord * torch.sum(box_mask * torch.pow(pred_cx - gt_cx, 2.0))
    loss_y = weight_coord * torch.sum(box_mask * torch.pow(pred_cy - gt_cy, 2.0))

    # -------------- Width and height losses -------------
    # lambda_coord * sum 1^{obj}_{ij} (sqrt(w) - sqrt(w_hat))^2
    # Sigmoid outputs are in [0,1] so sqrt is safe, but clampt at a tiny eps to keep gradients finite if a prediction is 0
    eps = 1e-8
    loss_w = weight_coord * torch.sum(
                               box_mask * torch.pow(
                                                   torch.sqrt(torch.clamp(pred_w, min=eps))
                                                     - torch.sqrt(torch.clamp(gt_w, min=eps)),
                                                2.0)
                                    )
    loss_h = weight_coord * torch.sum(
                                 box_mask * torch.pow(
                                                    torch.sqrt(torch.clamp(pred_h, min=eps))
                                                      - torch.sqrt(torch.clamp(gt_h, min=eps)),
                                                    2.0)
                                        )
    
    # -------------- No-object Confidence losses -------------
    # lambda_noobj * sum 1^{noobj}_{ij} (C - C_hat)^2
    # 1^{noobk}_{ij} = 1 - box_mask: every (box, cell) pair that is NOT the responsible predictor for an object. Target confidence for those is 0
    noobj_mask = 1 - box_mask
    loss_noobj = weight_noobj * torch.sum(noobj_mask * torch.pow(pred_conf - 0.0, 2.0))

    # -------------- Class losses -------------
    # sum 1^{obj}_i sum{c} (p(c) - p_hat(c))^2
    # The class indicator is per-cell (gt_mask), not per-box. Class scores are the trailing channels of output.
    # With num_classes==1 the GT class probability is 1 wherever an object is present
    if num_classes > 0:
        pred_cls = output[:, 5*num_boxes:5*num_boxes + num_classes] # (B, C, 7, 7)
        # add a class-dim to the per-cell mask so it broadcasts against the class predictions
        cell_mask = gt_mask.unsqueeze(1) # (B, 1, 7, 7)
        # Single-class case: target probability is 1 at object cells, 0 elsewhere
        # For multi-class extension build a one-hot target from the GT class id; we just have 1 class here
        gt_cls = cell_mask # (B, 1, 7, 7) broadcasts over classes if C>1
        loss_cls = torch.sum(cell_mask * torch.pow(pred_cls - gt_cls, 2.0))
    else:
        loss_cls = torch.tensor(0.0)
    # print('lx: %.4f, ly: %.4f, lw: %.4f, lh: %.4f, lobj: %.4f, lnoobj: %.4f, lcls: %.4f' % (loss_x, loss_y, loss_w, loss_h, loss_obj, loss_noobj, loss_cls))

    # the totol loss
    loss = loss_x + loss_y + loss_w + loss_h + loss_obj + loss_noobj + loss_cls
    return loss
