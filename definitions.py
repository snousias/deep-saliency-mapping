import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
from numpy import genfromtxt
import random
import gzip
import tarfile
import pickle
import os
from six.moves import urllib
from commonReadOBJPointCloud import *
import scipy.io
import sklearn as sk
from scipy.spatial.distance import pdist, squareform
import pickle
import hilbertcurve.hilbertcurve.hilbertcurve as hb
import glob
import saliencyDatasetClass as dset
from sklearn.metrics import confusion_matrix
from nets import *
from definitionskeras import *
from matplotlib import pyplot as plt
print("Import complete")
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# import tensorflow as tf
import time
import numpy as np
import os

import h5py
def show_progress(epoch, acc, val_acc, val_loss, total_epochs):
    msg = "Training Epoch {0}/{4} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss, total_epochs))

def get_session(gpu_fraction=0.333):
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=gpu_fraction,
        allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))