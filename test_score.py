
import torch

def dice_coefficient_3d(pred, target):
    """
    计算3D数据的Dice系数。
    :param pred: 预测体积，形状为 [depth, height, width]，二值化（0和1）
    :param target: 真实标签体积，形状为 [depth, height, width]，二值化（0和1）
    :return: Dice系数
    """
    # pred = pred[0]
    # target = target[0]
    #print(pred.shape)

    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    dice = (2. * intersection) / (pred.sum() + target.sum())

    return dice


import numpy as np
from scipy.spatial.distance import cdist


def hausdorff_distance_95_3d(u_volume, v_volume):
    """
    计算3D体积数据的HD95，使用cdist避免形状不匹配的问题。
    :param u_volume: 第一个体积，形状为 [depth, height, width]，二值化（0和1）
    :param v_volume: 第二个体积，形状为 [depth, height, width]，二值化（0和1）
    :return: HD95距离
    """
    # 将体积转换为点集
    u_volume = u_volume[0].cpu()
    v_volume = v_volume[0].cpu()
    #print(u_volume.shape)

    u_points = np.argwhere(u_volume)
    v_points = np.argwhere(v_volume)
    u_points = u_points.transpose(1,0)
    v_points = v_points.transpose(1, 0)

    # 计算所有成对点之间的距离
    dist_matrix = cdist(u_points, v_points)

    hd95 = np.percentile(dist_matrix.flatten(), 95)

    return hd95

# import torch
#
# # 假设output是神经网络的输出，形状为[batch_size, 1, height, width]
# output = torch.sigmoid(model_output)  # 使用sigmoid获取概率值
# predictions = (output > 0.5).int()  # 大于0.5的为1，否则为0

# import torch
#
# model_output = torch.rand(1,2,4,6,6)
# target = torch.rand(1,2,4,6,6)
# # 假设output是神经网络的输出，形状为[batch_size, num_classes, height, width]
# output = torch.softmax(model_output, dim=1)  # 使用softmax
# predictions = torch.argmax(output, dim=1)  # 沿num_classes维度选择最高概率的类别
#
# target1 = torch.softmax(target, dim=1)  # 使用softmax
# target2 = torch.argmax(target1, dim=1)  # 沿num_classes维度选择最高概率的类别
# #print(predictions.size())
#
#
# # 示例使用
# # 假设 pred_volume 和 target_volume 都是形状为 [depth, height, width] 的 PyTorch Tensors
# pred_volume = predictions
# target_volume = target2
#
# dice = dice_coefficient_3d(pred_volume, target_volume)
# print(f"Dice Coefficient: {dice}")
#
# # 示例使用
# # 假设 u_volume 和 v_volume 都是形状为 [depth, height, width] 的 NumPy arrays，表示3D二值化体积数据
# u_volume = predictions
# v_volume = target2
# hd95_distance = hausdorff_distance_95_3d(u_volume, v_volume)
# print(f"HD95 Distance: {hd95_distance}")


import torch

def calculate_metrics(pred, target):
    TP = ((pred == 1) & (target == 1)).sum().float()
    TN = ((pred == 0) & (target == 0)).sum().float()
    FP = ((pred == 1) & (target == 0)).sum().float()
    FN = ((pred == 0) & (target == 1)).sum().float()

    #print(TP,TN,FP,FN)

    return TP, TN, FP, FN

def dice_score(TP, FP, FN):
    return (2 * TP) / (2 * TP + FP + FN)

def iou_score(TP, FP, FN):
    return TP / (TP + FP + FN)

def accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)

def specificity(TN, FP):
    return TN / (TN + FP)

def sensitivity(TP, FN):
    return TP / (TP + FN)


# 假设predictions和targets是整个数据集的预测和真实标签，形状为[batch_size, depth, height, width]
# 以下代码遍历数据集中的每个样本
# predictions = [u_volume,v_volume,u_volume]
# targets = [v_volume,u_volume,v_volume]
# metrics = {'DSC': [], 'IoU': [], 'Accuracy': [], 'Specificity': [], 'Sensitivity': [],'HD95':[],'dice':[]}
#
# for i in range(len(predictions)):
#     pred = predictions[i]
#     target = targets[i]
#
#     # 二值化操作，如果你的模型输出不是二值化的，需要根据实际情况调整
#     # pred = (pred >= threshold).int()
#     # target = (target >= threshold).int()
#
#     TP, TN, FP, FN = calculate_metrics(pred, target)
#
#     # 计算各项指标
#     metrics['DSC'].append(dice_score(TP, FP, FN).item())
#     metrics['IoU'].append(iou_score(TP, FP, FN).item())
#     metrics['Accuracy'].append(accuracy(TP, TN, FP, FN).item())
#     metrics['Specificity'].append(specificity(TN, FP).item())
#     metrics['Sensitivity'].append(sensitivity(TP, FN).item())
#     metrics['dice'].append(dice_coefficient_3d(pred,target).item())
#     metrics['HD95'].append(hausdorff_distance_95_3d(pred,target).item())
#
# # 计算平均指标
# for metric in metrics.keys():
#     metrics[metric] = sum(metrics[metric]) / len(metrics[metric])
#
# print("Average Metrics Across the Dataset:")
# print(metrics)



