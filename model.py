import math

from keras import Sequential
from keras.callbacks import ModelCheckpoint

from com.opensource.cnn.generator.data_gen import DataGenerator
from com.opensource.cnn.utils.cnn_logger import get_logger
from com.opensource.cnn.model.resnet import ResnetBuilder
from com.opensource.cnn.model.resnet import *
from com.opensource.cnn.generator.data_gen import AugmentedDataGenerator
from com.opensource.cnn.model.keras_callback import *
import glob
import os
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
from com.opensource.cnn.processors.processors import AbstractProcessor
from keras.preprocessing.image import ImageDataGenerator

import matplotlib
import matplotlib.pyplot as plt

logger = get_logger(__name__)

MODEL_PATH="model_path"
DATASET="data_set"
LABEL="label"
LEARNING_RATE="learning_rate"
IMAGE_DIM="image_dim"
EPOCHS="epochs"
MODEL_TYPE="model_type"
BATCH_SIZE="batch_size"

TEST_DIR="test_dir"
HOME_CLS_MODEL_PATH="home_cls_path"
INTERIOR_CLS_MODEL_PATH="interior_cls_path"
EXTERIOR_CLS_MODEL_PATH="exterior_cls_path"

BUILD = "build"
PREDICT = "predict"


class Model(AbstractProcessor):
	"""
		Class for building resnet model
	"""
	def __init__(self,dict):


		if  dict.__contains__(BUILD) and  dict[BUILD] is True:
			self.__verify(MODEL_PATH,dict)
			logger.debug("Model path {}".format(dict[MODEL_PATH]))
			self.model_path = dict[MODEL_PATH]

			self.__verify(DATASET, dict)
			logger.debug("Dataset path {}".format(dict[DATASET]))
			self.dataset = dict[DATASET]

			self.__verify(EPOCHS, dict)
			logger.debug("Epoch settings {}".format(dict[EPOCHS]))
			self.epochs = int(dict[EPOCHS])

			self.__verify(LABEL, dict)
			logger.debug("Labels provided {}".format(dict[LABEL]))
			self.labels= dict[LABEL].split(",")

			self.__verify(LEARNING_RATE, dict)
			logger.debug("Learning rate provided {}".format(dict[LEARNING_RATE]))
			self.lr = float(dict[LEARNING_RATE])

			self.__verify(IMAGE_DIM, dict)
			logger.debug("Target image dimension {}".format(dict[IMAGE_DIM]))
			self.image_dim=int(dict[IMAGE_DIM])

			self.__verify(MODEL_TYPE, dict)
			logger.debug("Model to be used {}".format(dict[MODEL_TYPE]))
			self.model_type = str(dict[MODEL_TYPE])

			self.__verify(BATCH_SIZE, dict)
			logger.debug("Batch size to be used {}".format(dict[BATCH_SIZE	]))
			self.batch_size = int(dict[BATCH_SIZE])


		elif dict.has_key(PREDICT) and dict[PREDICT] is True:
			#variable for prediction
			self.__verify(HOME_CLS_MODEL_PATH, dict)
			logger.info("Loading the home classification model")
			self.home_classification_model = keras.models.load_model(dict[HOME_CLS_MODEL_PATH])

			self.__verify(INTERIOR_CLS_MODEL_PATH, dict)
			logger.info("Loading the interior classification model")
			self.interior_classification_model =  keras.models.load_model(dict[INTERIOR_CLS_MODEL_PATH])

			self.__verify(EXTERIOR_CLS_MODEL_PATH, dict)
			logger.info("Loading the exterior classification model")
			self.exterior_classification_model =  keras.models.load_model(dict[EXTERIOR_CLS_MODEL_PATH])

			self.__verify(TEST_DIR, dict)
			self.test_image_directory = dict[TEST_DIR]
		else:
			raise Exception("Model class expects either build or predict only")


	def __verify(self,key, dict):
		"""
		Helper method to verify if a key is present in the dictionory or not. And throws exception if the key is not present in the
		dictionary
		"""
		if not dict.__contains__(key):
			raise Exception("Expecting {} to be part of dictionary".format(key))


	@classmethod
	def build_model(cls,dict):
		"""
		Use this constructor when you want to build the model
		"""
		return cls(dict)


	@classmethod
	def load_model_for_predict(cls, dict):
		return cls(dict)
		
	def process(self):
		if not self.labels is None:
			logger.info("Building the model with model_path:{0} dataset:{1} epochs:{2} labels:{3} learning rate:{4}".
						format(self.model_path,self.dataset,self.epochs,self.labels,self.lr))
			self.__build()
		else:
			logger.info("Classifying the images from the path:{}".format(self.test_image_directory))
			self.predict()


	def __build(self):
		#self.__build_model_test_data_gen(1)
		#self.__build_model_test_data_gen(20)
		self.__build_final_model_1()


	def __build_model_test_data_gen(self,epochs):
		logger.debug("Testing the model with {} epoch".format(epochs))
		model = ResNet18((1000, 1000, 3), 3)
		model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
		# create data generator objects in train and val mode
		# specify ablation=number of data points to train on
		training_generator = DataGenerator('train', ablation=100,f_cls=self.labels)
		validation_generator = DataGenerator('val', ablation=100,f_cls=self.labels)
		# fit: this will fit the net on 'ablation' samples, only 1 epoch
		model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=epochs)

	def __build_final_model(self):
		model_path = self.model_path
		logger.debug("Building final model and saving the best models at {} location".format(model_path))
		print("length of labels:{} and labels are :{}".format(len(self.labels), self.labels))
		model = ResNet18((self.image_dim, self.image_dim, 3), len(self.labels))
		sgd = optimizers.SGD()
		model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
		# create data generator objects in train and val mode
		# specify ablation=number of data points to train on
		dim = (self.image_dim, self.image_dim)
		training_generator = AugmentedDataGenerator('train', ablation=32, batch_size=32, dim=dim, data_set=self.dataset,
													f_cls=self.labels)
		validation_generator = AugmentedDataGenerator('val', ablation=32, dim=dim, batch_size=32, data_set=self.dataset,
													  f_cls=self.labels)

		checkpoint = ModelCheckpoint(model_path, monitor='val_auc', verbose=1, save_best_only=True, mode='max')
		auc_logger = roc_callback(validation_generator)
		history = LossHistory()
		# Using a decay rate of 0.1
		decay = DecayLR(base_lr=self.lr)

		model.fit_generator(generator=training_generator,
							validation_data=validation_generator,
							epochs=self.epochs, callbacks=[auc_logger, history, decay, checkpoint])

	# self.__plot_model_accuracy(model)
	def __build_final_model_1(self):
		model_path = self.model_path
		logger.debug("Building  model of type {0} and saving the best models at {1} location".format(self.model_type,model_path))
		print("length of labels:{} and labels are :{}".format(len(self.labels), self.labels))
		if self.model_type.__eq__("resnet18"):
			model = ResNet18((self.image_dim, self.image_dim, 3), len(self.labels))
		elif self.model_type.__eq__("resnet34"):
			model = ResNet34((self.image_dim, self.image_dim, 3), len(self.labels))
		elif self.model_type.__eq__("resnet50"):
			model = ResNet50((self.image_dim, self.image_dim, 3), len(self.labels))
		elif self.model_type.__eq__("resnet101"):
			model = ResNet101((self.image_dim, self.image_dim, 3), len(self.labels))
		elif self.model_type.__eq__("resnet152"):
			model = ResNet152((self.image_dim, self.image_dim, 3), len(self.labels))
		elif self.model_type.__eq__("sequential_m"):
			#calling a different method for model_building altogether
			self.__build_sequential_m_model()
			return

		#The below code is same for all the resnet models
		sgd = optimizers.SGD()
		model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
		# create data generator objects in train and val mode
		# specify ablation=number of data points to train on
		dim = (self.image_dim, self.image_dim)

		train_path = os.path.join(self.dataset, "train")

		if not os.path.exists(train_path):
			raise Exception("Expecting a folder by name train in the input data directory {}".format(self.dataset))

		training_generator = ImageDataGenerator(
			rescale=1. / 255,
			rotation_range=45,
			width_shift_range=.15,
			height_shift_range=.15,
			horizontal_flip=True,
			zoom_range=0.5
		).flow_from_directory(batch_size=self.batch_size,directory=train_path,shuffle=True,
															 target_size=(self.image_dim, self.image_dim),
															 class_mode='categorical',classes=self.labels)

		test_path = os.path.join(self.dataset, "test")
		if not os.path.exists(test_path):
			raise Exception("Expecting a folder by name test in the input data directory {}".format(self.dataset))

		validation_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(batch_size=self.batch_size,
														 directory=test_path,
														 target_size=(self.image_dim, self.image_dim),
														 class_mode='categorical',classes=self.labels)

		checkpoint = ModelCheckpoint(model_path, monitor='val_auc', verbose=1, save_best_only=True, mode='max')
		auc_logger = roc_callback(validation_generator)
		history = LossHistory()
		# Using a decay rate of 0.1
		decay = DecayLR(base_lr=self.lr)
		BATCH_SIZE = self.batch_size

		TRAINING_SIZE = 2988
		VALIDATION_SIZE = 300
		compute_steps_per_epoch = lambda x: int(math.ceil(1. * x / BATCH_SIZE))
		steps_per_epoch = compute_steps_per_epoch(TRAINING_SIZE)
		val_steps = compute_steps_per_epoch(VALIDATION_SIZE)
		model.fit_generator(generator=training_generator,
							validation_data=validation_generator,steps_per_epoch=steps_per_epoch,validation_steps=val_steps,
							epochs=self.epochs, callbacks=[auc_logger, history, decay, checkpoint])


	# self.__plot_model_accuracy(model)

	def __build_final_model_custom(self):
		model_path = self.model_path
		logger.debug("Building final model and saving the best models at {} location".format(model_path))
		print("length of labels:{} and labels are :{}".format(len(self.labels),self.labels))
		#model = ResNet18((self.image_dim,self.image_dim, 3), len(self.labels))
		model = Sequential([
			Conv2D(16, 3, padding='same', activation='relu', input_shape=(self.image_dim, self.image_dim, 3)),
			MaxPooling2D(),
			Conv2D(32, 3, padding='same', activation='relu'),
			MaxPooling2D(),
			Conv2D(64, 3, padding='same', activation='relu'),
			MaxPooling2D(),
			Flatten(),
			Dense(512, activation='relu'),
			Dense(5)
		])
		sgd = optimizers.SGD()
		#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
		model.compile(optimizer='adam',loss=keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])
		# create data generator objects in train and val mode
		# specify ablation=number of data points to train on
		dim = (self.image_dim,self.image_dim)

		# Generate the training image generator
		train_image_gen = ImageDataGenerator(
			rescale=1. / 255,
			rotation_range=45,
			width_shift_range=.15,
			height_shift_range=.15,
			horizontal_flip=True,
			zoom_range=0.5
		)

		train_path = os.path.join(self.dataset,"train")
		train_data_gen = train_image_gen.flow_from_directory(batch_size=32,
															 directory=train_path,
															 shuffle=True,
															 target_size=(self.image_dim, self.image_dim),
															 class_mode='binary')


		test_path = os.path.join(self.dataset,"validation")
		# Generating the validation image generator
		image_gen_val = ImageDataGenerator(rescale=1. / 255)
		val_data_gen = image_gen_val.flow_from_directory(batch_size=32,
														 directory=test_path,
														 target_size=(self.image_dim, self.image_dim),
														 class_mode='binary')

		checkpoint = ModelCheckpoint(model_path, monitor='val_auc', verbose=1, save_best_only=True, mode='max')
		auc_logger = roc_callback(val_data_gen)
		history = LossHistory()
		#Using a decay rate of 0.1
		decay = DecayLR(base_lr=self.lr)

		#model.fit_generator(generator=train_data_gen,validation_data=val_data_gen,steps_per_epoch=100,validation_steps=10,epochs=self.epochs, callbacks=[auc_logger, history, decay, checkpoint])
		#self.__plot_model_accuracy(model)
		history = model.fit_generator(train_data_gen,steps_per_epoch=100,epochs=self.epochs,validation_data=val_data_gen,validation_steps=10)

	def predict(self):
		"""
		Method to scan the images from the directory and classify them
		"""
		paths = glob.glob(os.path.join(self.test_image_dir, '*'))
		for im_path in paths:
			logger.info("Processing image:{}".format(im_path))
			image = io.imread(im_path)
			image_resized = resize(image, (100, 100),anti_aliasing=True)
			outer_prediction = self.get_home_classification_prediction(image_resized)
			inner_prediction=None
			if outer_prediction.__eq__("exterior"):
				inner_prediction = self.get_exterior_classification(image_resized)
			else:
				inner_prediction = self.get_interior_classification(image_resized)
			logger.info(" {0} image is predicted as {1} and belongs to {2}".format(im_path,outer_prediction,inner_prediction))



	def get_home_classification_prediction(self,image_resized):
		y_prob = self.home_classification_model.predict(image_resized[np.newaxis, :])
		logger.info("Home classification probability:{}".format(y_prob))
		max_arg_value = y_prob.argmax()
		if max_arg_value == 0:
			return "exterior"
		else:
			return "interior"

	def get_exterior_classification(self,image_resized):
		y_prob = self.exterior_classification_model.predict(image_resized[np.newaxis, :])
		logger.info("Exterior classification probability:{}".format(y_prob))
		max_arg_value = y_prob.argmax()
		if max_arg_value == 0:
			return "apartment"
		elif max_arg_value == 1:
			return "bungalow"
		else:
			return "mansion"

	def get_interior_classification(self, image_resized):
		y_prob =  self.interior_classification_model.predict(image_resized[np.newaxis, :])
		logger.info("Interior classification probability:{}".format(y_prob))
		max_arg_value = y_prob.argmax()
		if max_arg_value == 0:
			return "kitchen"
		elif max_arg_value == 1:
			return "bathroom"
		elif max_arg_value == 2:
			return "bedroom"
		else:
			return "livingroom"

	def __plot_model_accuracy(self,model):
		"""
		Method to save the the model accuracy between training and prediction
		"""
		model_path_dir = os.path.dirname(self.model_path)
		model_plot_filename = os.path.join(model_path_dir,"model_accuracy.jpeg")

		acc = model.history['accuracy']
		val_acc = model.history['val_accuracy']

		loss = model.history['loss']
		val_loss = model.history['val_loss']

		epochs_range = range(self.epochs)

		plt.figure(figsize=(8, 8))
		plt.subplot(1, 2, 1)
		plt.plot(epochs_range, acc, label='Training Accuracy')
		plt.plot(epochs_range, val_acc, label='Validation Accuracy')
		plt.legend(loc='lower right')
		plt.title('Training and Validation Accuracy')

		plt.subplot(1, 2, 2)
		plt.plot(epochs_range, loss, label='Training Loss')
		plt.plot(epochs_range, val_loss, label='Validation Loss')
		plt.legend(loc='upper right')
		plt.title('Training and Validation Loss')
		plt.savefig(model_plot_filename)
		logger.info("Model accuracy plot is saved at {}".format(model_plot_filename))


	def __build_custom_model(self):

		model = ResNet18((self.image_dim, self.image_dim, 3), len(self.labels))
		sgd = optimizers.SGD()
		model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

		checkpoint = ModelCheckpoint(self.model_path, monitor='val_auc', verbose=1, save_best_only=True, mode='max')
		auc_logger = roc_callback(val_data_gen)
		history = LossHistory()
		# Using a decay rate of 0.1
		decay = DecayLR(base_lr=self.lr)

		model.fit_generator(generator=train_data_gen,
							validation_data=val_data_gen,
							epochs=self.epochs, callbacks=[auc_logger, history, decay, checkpoint])

	#Custom sequential model for mayank
	def __build_sequential_m_model(self):
		"""
		custom sequential model
		"""
		# creating structure of CNN
		logger.info("Running sequential_m on input directory {0}".format(self.dataset))
		classifier = Sequential()
		classifier.add(Conv2D(32, (3, 3), input_shape=(self.image_dim, self.image_dim, 3), activation='relu'))
		classifier.add(MaxPooling2D(pool_size=(2, 2)))
		classifier.add(Flatten())
		classifier.add(Dense(units=128, activation='relu'))
		classifier.add(Dense(units=1, activation='sigmoid'))
		classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

		# preprocessing training/test images
		train_datagen = ImageDataGenerator(rescale=1. / 255,
										   shear_range=0.2,
										   zoom_range=0.2,
										   horizontal_flip=True)
		test_datagen = ImageDataGenerator(rescale=1. / 255)

		train_path = os.path.join(self.dataset, "train")
		if(not os.path.exists(train_path)):
			raise Exception("Program is expecting a folder by name train under the input directory {}".format(self.dataset))

		test_path = os.path.join(self.dataset, "test")

		if (not os.path.exists(test_path)):
			raise Exception(
				"Program is expecting a folder by name test under the input directory {}".format(self.dataset))

		training_set = train_datagen.flow_from_directory(train_path,
														 target_size=(self.image_dim, self.image_dim),
														 batch_size=self.batch_size,
														 class_mode='binary')
		test_set = test_datagen.flow_from_directory(test_path,
													target_size=(self.image_dim,self.image_dim),
													batch_size=self.batch_size,
													class_mode='binary')

		# training the network
		classifier.fit_generator(training_set,
								 steps_per_epoch=9000,
								 epochs=self.epochs,
								 validation_data=test_set,
								 validation_steps=1800)

		# saving model
		classifier.save(self.model_path)
		logger.info("Sequential model building complete and the model file is saved {0}".format(self.model_path))
