import glob
from skimage import io
from keras.preprocessing.image import ImageDataGenerator
import os
from skimage.transform import rescale, resize
from com.opensource.cnn.utils.cnn_logger import get_logger
import numpy as np
import keras


logger = get_logger(__name__)
class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'

	def __init__(self, mode='train', ablation=None, f_cls=['kitchen', 'bathroom','bedroom'],
				 batch_size=32, dim=(100, 100), n_channels=3, shuffle=True,data_set=""):
		"""
		Initialise the data generator
		"""
		self.dim = dim
		self.batch_size = batch_size
		self.labels = {}
		self.list_IDs = []
		DATASET_PATH = data_set
		# glob through directory of each class
		for i, cls in enumerate(f_cls):
			logger.debug("taking data from class:{}".format(cls))
			paths = glob.glob(os.path.join(DATASET_PATH, cls, '*'))
			brk_point = int(len(paths) * 0.8)
			if mode == 'train':
				paths = paths[:brk_point]
			else:
				paths = paths[brk_point:]
			if ablation is not None:
				paths = paths[:ablation]
			self.list_IDs += paths
			self.labels.update({p: i for p in paths})

		self.n_channels = n_channels
		self.n_classes = len(f_cls)
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(list_IDs_temp)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp):
		'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size, *self.dim, self.n_channels))
		y = np.empty((self.batch_size), dtype=int)

		delete_rows = []

		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			# Store sample
			try:
				img = io.imread(ID)
			except Exception as e:
				logger.error("Error opening the image:{}".format(ID))
				raise Exception("Error opening the image:{}".format(ID))
			img = img / 255
			#img = resize(img, (100, 100), anti_aliasing=True)

			X[i,] = img

			# Store class
			y[i] = self.labels[ID]

		#X = np.delete(X, delete_rows, axis=0)
		#y = np.delete(y, delete_rows, axis=0)

		#X = np.delete(X, delete_rows, axis=0)
		#y = np.delete(y, delete_rows, axis=0)
		return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


class AugmentedDataGenerator(keras.utils.Sequence):
		"""
		Generates data for Keras
		Use this class for generating the data. not the above class
		"""

		def __init__(self, mode='train', ablation=None,  f_cls=['kitchen', 'bathroom'],
					 batch_size=32, dim=(1000, 1000), n_channels=3, shuffle=True,data_set=""):
			'Initialization'
			self.dim = dim
			self.batch_size = batch_size
			self.labels = {}
			self.list_IDs = []
			self.mode = mode
			DATASET_PATH = data_set

			self.datagen = ImageDataGenerator(
				featurewise_center=True,
				featurewise_std_normalization=True,
				rotation_range=20,
				width_shift_range=0.2,
				height_shift_range=0.2,
				horizontal_flip=True,
				zoom_range=0.5,
				vertical_flip=True,
				rescale = 1./255)

			for i, cls in enumerate(f_cls):
				paths = glob.glob(os.path.join(DATASET_PATH, cls, '*'))
				brk_point = int(len(paths) * 0.8)
				if self.mode == 'train':
					paths = paths[:brk_point]
				else:
					paths = paths[brk_point:]
				if ablation is not None:
					paths = paths[:ablation]
				self.list_IDs += paths
				self.labels.update({p: i for p in paths})

			self.n_channels = n_channels
			self.n_classes = len(f_cls)
			self.shuffle = shuffle
			self.on_epoch_end()

		def __len__(self):
			'Denotes the number of batches per epoch'
			return int(np.floor(len(self.list_IDs) / self.batch_size))

		def __getitem__(self, index):
			'Generate one batch of data'
			# Generate indexes of the batch
			indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

			# Find list of IDs
			list_IDs_temp = [self.list_IDs[k] for k in indexes]

			# Generate data
			X, y = self.__data_generation(list_IDs_temp)

			return X, y

		def on_epoch_end(self):
			'Updates indexes after each epoch'
			self.indexes = np.arange(len(self.list_IDs))
			if self.shuffle == True:
				np.random.shuffle(self.indexes)

		def __data_generation(self, list_IDs_temp):

			'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
			# Initialization
			X = np.empty((self.batch_size, *self.dim, self.n_channels))
			y = np.empty((self.batch_size), dtype=int)

			delete_rows = []

			# Generate data
			for i, ID in enumerate(list_IDs_temp):
				# Store sample
				img = io.imread(ID)
				img = img / 255
				#no need to resize as we already sending the resized images
				X[i,] = img

				# Store class
				#print("class label:{}".format(self.labels[ID]))
				y[i] = self.labels[ID]

			# data augmentation
			if self.mode == 'train':
				aug_x = np.stack([self.datagen.random_transform(img) for img in X])
				X = np.concatenate([X, aug_x])
				y = np.concatenate([y, y])
			return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


