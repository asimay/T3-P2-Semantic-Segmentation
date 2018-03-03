import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm


def gen_augumentation_data(data_folder, image_shape):
	"""
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
	image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
	label_paths = {
		re.sub(r'_(lane|road_)', '_', os.path.basename(path)): path
		for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))
	}

	print("image_paths:", image_paths)
	print("label_paths:", label_paths)

	for image_path, label_path in zip(image_paths, label_paths):
		gt_image_file = label_paths[os.path.basename(image_path)]
		#gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
		gt_image = scipy.misc.imread(gt_image_file)

		#origal_image = scipy.misc.imresize(scipy.misc.imread(image_path), image_shape)
		original_image = scipy.misc.imread(image_path)
		# horizon flip images
		flip_image = np.fliplr(original_image)
		new_image_file = 'new_' + os.path.basename(image_path)
		print('new_image_file:', new_image_file)
		scipy.misc.imsave(os.path.join(data_folder, new_image_file), flip_image)

		# horizon flip gt images
		flip_gt_image = np.fliplr(gt_image)
		new_gt_image_file = 'new_' + os.path.basename(gt_image_file)
		print('new_gt_image_file:', new_gt_image_file)
		scipy.misc.imsave(os.path.join(data_folder, new_gt_image_file), flip_gt_image)

		#yield scipy.misc.imsave(os.path.join(data_folder, new_image_file), flip_image), scipy.misc.imsave(os.path.join(data_folder, new_gt_image_file), flip_gt_image)

	return


if __name__ == '__main__':
	image_shape = (160, 576)
	data_dir = './test'
	print("come in")
	gen_augumentation_data(data_dir, image_shape)
	print("exit")
