import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch
import json

from visual_utils import visualize_utils as V
from pyntcloud import PyntCloud
from scipy.spatial.transform import Rotation


def main():
	#pc = np.loadtxt('kitti_sample_scan.txt')
	#pc = np.load('/home/success/Documents/rciServer/mnt/beegfs/gpu/argoverse-tracking-all-training/KITTI_test/training/lidar/2011_09_26_0005_0001.npy')
	#bbox = np.load('/home/success/Documents/rciServer/mnt/beegfs/gpu/argoverse-tracking-all-training/KITTI_test/training/label/2011_09_26_0005_0001.npy')
	pc = np.load('/home/success/Documents/rciServer/mnt/beegfs/gpu/argoverse-tracking-all-training/KITTI/validation/lidar/2011_09_26_0001_0005.npy')
	bbox = np.load('/home/success/Documents/rciServer/mnt/data/vras/data/gebreawe/Modified_Experiments/ST_PointRCNN_KITTI2/Teacher/tools/script/test/DET/val/kitti_train_mf1500_w3_p16384_agum_f1_1/per_sweep_annotations_amodal/2011_09_26_0001_0005.npy')
	pc = '/home/success/Documents/rciServer/mnt/beegfs/gpu/argoverse-tracking-all-training/WOD/WOD_ARGO/validation/f-5_5/f4500/validation/mframe/val/10203656353524179475_7625_000_7645_000/lidar/PC_1507330244649398000.ply'
	bbox = '/home/success/Documents/rciServer/mnt/data/vras/data/gebreawe/Modified_Experiments/ST_PointRCNN_WOD/Teacher/tools/script/test/DET/wod_train_mf1500_w3_p16384_agum_f1_0/10203656353524179475_7625_000_7645_000/per_sweep_annotations_amodal/tracked_object_labels_1507330244649398000.json'
	#bbox = '/home/success/Documents/rciServer/mnt/beegfs/gpu/argoverse-tracking-all-training/WOD/WOD_ARGO/validation/f-5_5/f4500/validation/mframe/val/10203656353524179475_7625_000_7645_000/per_sweep_annotations_amodal/tracked_object_labels_1507330244649398000.json'
	#'''
	pc = '/home/success/Documents/rciServer/mnt/data/vras/data/raw_Argoverse/Datasets/argoverse-tracking-all-training/argoverse-tracking/processed/source/val_v1_all/argoverse-tracking/val/f-5_5/validation/mframe/val1/00c561b9-2057-358d-82c6-5b06d76cebcf/lidar/PC_315969635820272000.ply'
	bbox = '/home/success/Documents/rciServer/mnt/data/vras/data/raw_Argoverse/Datasets/argoverse-tracking-all-training/argoverse-tracking/processed/source/val_v1_all/argoverse-tracking/val/f-5_5/validation/validation_with_speed/00c561b9-2057-358d-82c6-5b06d76cebcf/per_sweep_annotations_amodal/tracked_object_labels_315969635820272000.json'
	#'''

	argo_to_kitti = np.array([[0, -1, 0],[0, 0, -1],[1, 0, 0]])
	kitti_to_argo = np.array([[0, 0, 1],[-1, 0, 0],[0, -1, 0]])

	data = PyntCloud.from_file(pc)
	print(data)
	x = np.array(data.points.x5)[:, np.newaxis]
	y = np.array(data.points.y5)[:, np.newaxis]
	z = np.array(data.points.z5)[:, np.newaxis]

	pc = np.concatenate((x,y,z), axis=1)


	#print(pc.shape, bbox.shape)

	f = open(bbox)
	label_data = json.load(f)
	det = len(label_data)
	bbox3d = np.zeros((det, 8))
	for k, label in enumerate (label_data): 
		cls_type = label['label_class']
		
		alpha = np.arctan2(label['center']['z'],label['center']['x'])
		
		h = float(label['height'])
		w = float(label['width'])
		l = float(label['length'])
		pos = np.array([float(label['center']['x']), float(label['center']['y']), float(label['center']['z'])], dtype=np.float32)
		
		#pos = np.dot(kitti_to_argo,pos_argo)
		w,x,y,z = label['rotation']['w'],label['rotation']['x'],label['rotation']['y'],label['rotation']['z']
		q = np.array([x, y, z, w])       
		rot_mat_argo = Rotation.from_quat(q).as_dcm()
		
		ry = -Rotation.from_quat(q).as_euler('xyz')[-1] + np.pi/2.
		
		bbox3d[k,0] = label['center']['x']
		bbox3d[k,1] = label['center']['y']
		bbox3d[k,2] = label['center']['z']
		bbox3d[k,3] = label['length']
		bbox3d[k,4] = label['width']
		bbox3d[k,5] = label['height']
		bbox3d[k,6] = (np.pi/2. -ry)
		bbox3d[k,7] = 1 #label['label_class']
		
	V.draw_scenes(
		points=pc[:,:3], ref_boxes=bbox3d,ref_scores=None, ref_labels=None)
	mlab.show(stop=True)


if __name__ == '__main__':
    main()
