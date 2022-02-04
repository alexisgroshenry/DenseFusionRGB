import _init_paths
import argparse
import os
import random
import numpy as np
import yaml
import copy
import cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseNetRGB, PoseRefineNet, PoseRefineNetRGB
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--metric', type=str, default='3d', help='evaluation metric among [2d, 3d] (default: 3d)')
parser.add_argument('--depth_pred', help='use depth predictions by TransDepth', action='store_true')
parser.add_argument('--keep_symmetric', help='keep symmetric objects in the linemod dataset', action='store_true')
parser.add_argument('--rmv_cloud_embed', help='remove the point cloud embedding', action='store_true')
parser.add_argument('--rmv_iter_refine', help='remove the iterative refinement', action='store_true')
parser.add_argument('--vis_pred', help='visualize the predicted pose', action='store_true')
opt = parser.parse_args()

if opt.keep_symmetric:
    objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
else:
    objlist = [1, 2, 4, 5, 6, 8, 9, 12, 13, 14, 15]

if opt.vis_pred:
    for obj in objlist:
        pred_folder = '{0}/data/{1}/pose_prediction'.format(opt.dataset_root, '%02d' % obj)
        if not os.path.exists(pred_folder):
            os.mkdir(pred_folder)
    
num_objects = len(objlist)
num_points = 500
if opt.rmv_iter_refine:
    iteration = 0
else:
    iteration = 4
bs = 1
dataset_config_dir = 'datasets/linemod/dataset_config'
output_result_dir = 'experiments/eval_result/linemod'


if not opt.rmv_cloud_embed:
    estimator = PoseNet(num_points = num_points, num_obj = num_objects)
else:
    estimator = PoseNetRGB(num_points = num_points, num_obj = num_objects)
estimator.cuda()
estimator.load_state_dict(torch.load(opt.model))
estimator.eval()

if not opt.rmv_iter_refine:
    if not opt.rmv_cloud_embed:
        refiner = PoseRefineNet(num_points = num_points, num_obj = num_objects)
    else:
        refiner = PoseRefineNetRGB(num_points = num_points, num_obj = num_objects)
    refiner.cuda()
    refiner.load_state_dict(torch.load(opt.refine_model))
    refiner.eval()


testdataset = PoseDataset_linemod('eval', num_points, False, opt.dataset_root, 0.0, True, opt.depth_pred, opt.keep_symmetric, opt.rmv_cloud_embed)
testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=0)

sym_list = testdataset.get_sym_list()
num_points_mesh = testdataset.get_num_points_mesh()
criterion = Loss(num_points_mesh, sym_list, not opt.rmv_cloud_embed)
if not opt.rmv_iter_refine:
    criterion_refine = Loss_refine(num_points_mesh, sym_list, not opt.rmv_cloud_embed)

diameter = []
meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
meta = yaml.load(meta_file)
for obj in objlist:
    diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)
print(len(diameter),diameter)


success_count = [0 for i in range(num_objects)]
num_count = [0 for i in range(num_objects)]
obj_dis = [0 for i in range(num_objects)]
fw = open('{0}/eval_result_logs.txt'.format(output_result_dir), 'w')

for i, data in enumerate(testdataloader, 0):
    points, choose, img, target, model_points, idx = data
    if objlist[idx[0].item()] in [10,11]:
        num_count[idx[0].item()] += 1
        continue
    if not opt.rmv_cloud_embed and len(points.size()) == 2:
        print('No.{0} NOT Pass! Lost detection!'.format(i))
        fw.write('No.{0} NOT Pass! Lost detection!\n'.format(i))
        continue
    points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                     Variable(choose).cuda(), \
                                                     Variable(img).cuda(), \
                                                     Variable(target).cuda(), \
                                                     Variable(model_points).cuda(), \
                                                     Variable(idx).cuda()
    try:
        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
    except:
        print(points.shape, model_points.shape, idx)
        continue
    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
    pred_c = pred_c.view(bs, num_points)
    how_max, which_max = torch.max(pred_c, 1)
    pred_t = pred_t.view(bs * num_points, 1, 3)

    if opt.rmv_cloud_embed and opt.rmv_iter_refine:
        my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
        my_t = pred_t[which_max[0]].view(-1).cpu().data.numpy()

    else:
        my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
        my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
        my_pred = np.append(my_r, my_t)

        for ite in range(0, iteration):
            T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
            my_mat = quaternion_matrix(my_r)
            R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
            my_mat[0:3, 3] = my_t
            
            
            new_points = torch.bmm((points - T), R).contiguous()
            pred_r, pred_t = refiner(new_points, emb, idx)
            pred_r = pred_r.view(1, 1, -1)
            pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
            my_r_2 = pred_r.view(-1).cpu().data.numpy()
            my_t_2 = pred_t.view(-1).cpu().data.numpy()
            my_mat_2 = quaternion_matrix(my_r_2)
            my_mat_2[0:3, 3] = my_t_2

            my_mat_final = np.dot(my_mat, my_mat_2)
            my_r_final = copy.deepcopy(my_mat_final)
            my_r_final[0:3, 3] = 0
            my_r_final = quaternion_from_matrix(my_r_final, True)
            my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

            my_pred = np.append(my_r_final, my_t_final)
            my_r = my_r_final
            my_t = my_t_final

    # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)

    model_points = model_points[0].cpu().detach().numpy()
    my_r = quaternion_matrix(my_r)[:3, :3]
    pred = np.dot(model_points, my_r.T) + my_t
    target = target[0].cpu().detach().numpy()

    if idx[0].item() in sym_list:
        pred = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        target = torch.from_numpy(target.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        inds = KNearestNeighbor.apply(1, target.unsqueeze(0), pred.unsqueeze(0))
        target = torch.index_select(target, 1, inds.view(-1) - 1)
        dis = torch.mean(torch.norm((pred.transpose(1, 0) - target.transpose(1, 0)), dim=1), dim=0).item()
    else:
        if opt.metric == '3d':
            dis = np.mean(np.linalg.norm(pred - target, axis=1))
        elif opt.metric == '2d':
            dis = np.mean(np.linalg.norm((pred - target)[:2], axis=1))
        
    if dis < diameter[idx[0].item()]:
        success_count[idx[0].item()] += 1
        #print('No.{0} Pass! Distance: {1}'.format(i, dis))
        fw.write('No.{0} Pass! Distance: {1}\n'.format(i, dis))
    else:
        #print('No.{0} NOT Pass! Distance: {1}'.format(i, dis))
        fw.write('No.{0} NOT Pass! Distance: {1}\n'.format(i, dis))
    num_count[idx[0].item()] += 1
    obj_dis[idx[0].item()] += dis

    if opt.vis_pred:
        cam_cx = 325.26110
        cam_cy = 242.04899
        cam_fx = 572.41140
        cam_fy = 573.57043
        cam_mat = np.array([[cam_fx, 0., cam_cx], [0., cam_fy, cam_cy], [0., 0., 1.]])
        distortion = np.array([[0., 0.0,  0.0, 0.0, 0.0]])
        imgpts, _ = cv2.projectPoints(model_points, my_r, my_t, cam_mat, distortion)
        img = cv2.imread(testdataset.list_rgb[i])
        for pt in imgpts.reshape(-1,2):
            img = cv2.circle(img, tuple(pt), radius=1, color=(0, 255, 255), thickness=-1)
        cv2.imwrite(testdataset.list_rgb[i].replace('rgb','pose_prediction'), img)

for i in range(num_objects):
    try:
        print('Object {0} success rate: {1} avg distance: {2}'.format(objlist[i], float(success_count[i]) / num_count[i], obj_dis[i] / num_count[i]),)
        fw.write('Object {0} success rate: {1} avg distance: {2}\n'.format(objlist[i], float(success_count[i]) / num_count[i], obj_dis[i] / num_count[i]))
    except:
        print('Object {0} symmetric'.format(objlist[i]))
print('ALL success rate: {0}'.format(float(sum(success_count)) / sum(num_count)))
fw.write('ALL success rate: {0}\n'.format(float(sum(success_count)) / sum(num_count)))
fw.close()
