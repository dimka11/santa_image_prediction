import numpy as np
import pandas as pd
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import datetime
import shutil

import os

if __name__ == "__main__":
	batch_size = 128
	img_height = 224
	img_width = 224

	shutil.unpack_archive('./data/weights/tf_model/tf_model.zip', './data/weights/tf_model/', 'zip')

	test_dir = './data/test'
	predict_ds = tf.keras.utils.image_dataset_from_directory(test_dir,seed=123,label_mode=None,image_size=(img_height, img_width),batch_size=batch_size, shuffle=False)

	model = keras.models.load_model('./data/weights/tf_model')

	preds_all = None
	for item in predict_ds:
	    preds = model.predict(item)
	    preds = np.argmax(preds,axis=1)
	    if preds_all is not None:
	        preds_all = np.concatenate([preds_all, preds])
	    else:
	        preds_all = preds.copy()

	labels = {'0': 0, '1': 1, '2': 2}
	labels = dict((v,k) for k,v in labels.items())
	predictions = [labels[k] for k in preds_all]
	submit = pd.DataFrame({'image_name': [path.split('/')[-1] for path in predict_ds.file_paths], 'class_id': predictions})
	submit.to_csv('./data/out/submission.csv', sep='\t', index=False)