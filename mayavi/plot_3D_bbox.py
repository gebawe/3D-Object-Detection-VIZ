import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch

from visual_utils import visualize_utils as V




def main():
	pc = np.load('data/lidar/2011_09_26_0036_0030.npy')
	bbox = np.load('data/label/2011_09_26_0036_0030.npy')

	print(pc.shape, bbox.shape)
	bbox3d = bbox

	V.draw_scenes(
		points=pc[:,:3], ref_boxes=bbox,ref_scores=None, ref_labels=None)
	mlab.show(stop=True)


if __name__ == '__main__':
    main()
