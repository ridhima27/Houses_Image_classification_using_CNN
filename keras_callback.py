import keras
from keras import optimizers
from keras.callbacks import *
from sklearn.metrics import roc_auc_score
import numpy as np
from com.opensource.cnn.utils.cnn_logger import get_logger

logger = get_logger(__name__)

# callback to append loss
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))

# learning rate decay
class DecayLR(keras.callbacks.Callback):
	def __init__(self, base_lr=0.001, decay_epoch=1):
		super(DecayLR, self).__init__()
		self.base_lr = base_lr
		self.decay_epoch = decay_epoch
		self.lr_history = []

	# set lr on_train_begin
	def on_train_begin(self, logs={}):
		K.set_value(self.model.optimizer.lr, self.base_lr)

	# change learning rate at the end of epoch
	def on_epoch_end(self, epoch, logs={}):
		new_lr = self.base_lr * (0.5 ** (epoch // self.decay_epoch))
		self.lr_history.append(K.get_value(self.model.optimizer.lr))
		K.set_value(self.model.optimizer.lr, new_lr)


class roc_callback(Callback):

	def __init__(self,validation_generator):
		self.validation_generator = validation_generator

	def on_train_begin(self, logs={}):
		logs={}
		logs['val_auc'] = 0

	def on_epoch_end(self, epoch, logs={}):
		y_p = []
		y_v = []
		for i in range(len(self.validation_generator)):
			x_val, y_val = self.validation_generator[i]
			y_pred = self.model.predict(x_val)
			y_p.append(y_pred)
			y_v.append(y_val)
		y_p = np.concatenate(y_p)
		y_v = np.concatenate(y_v)
		roc_auc = roc_auc_score(y_v, y_p)
		logger.info('Val AUC for epoch{0}: {1}'.format(epoch, roc_auc))
		logs['val_auc'] = roc_auc