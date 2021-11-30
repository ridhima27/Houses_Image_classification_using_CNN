import argparse
from com.opensource.cnn.utils.cnn_logger import get_logger
import os
import requests
import zipfile
import shutil

from com.opensource.cnn.model.model import *
from com.opensource.cnn.scraping.web_scrap import Webscrapper
from com.opensource.cnn.utils.compress_image import compress_image
from com.opensource.cnn.processors.processors import AbstractProcessor

logger = get_logger(__name__)

class Processor:
		def __init__(self):
			self.parser =  argparse.ArgumentParser(description='Processing operation to be performed.')
			self.parser.add_argument("-d","--download", help="download data set from the url")
			self.parser.add_argument("-o","--out",help="Save the download data to a directory.")
			self.parser.add_argument("-p", "--preprocess", help="Save the download data to a directory.")
			self.parser.add_argument("-b", "--build", help="Provide the path to save the model")
			#self.parser.add_argument("-cb", "--custom_build", help="Provide the path to build custom model and to save the path")
			self.parser.add_argument("-ds", "--dataset", help="Provide the dataset path")
			#self.parser.add_argument("-trds", "--training_dataset", help="Provide the training dataset path for custom model building")
			#self.parser.add_argument("-teds", "--test_dataset", help="Provide the test dataset path for custom model building")
			self.parser.add_argument("-e","--epochs", help="Number of epochs to train the model")
			self.parser.add_argument("-lr","--learningrate", help="specify the learning rate for the model")
			self.parser.add_argument("-l", "--label", help="Number of class to identify.It should be in comma seperated values. example:kitchen,bathroom,bedroom")
			self.parser.add_argument("-id", "--image_dim", help="give the image dimension")
			self.parser.add_argument("-c", "--compress",help="compress all the images in a given directory")
			self.parser.add_argument("-pr", "--predict", help="provide the test directory")
			self.parser.add_argument("-hcm", "--home_classification_model", help="provide the home classification model path")
			self.parser.add_argument("-exm", "--exterior_classification_model",help="provide the exterior classification model path")
			self.parser.add_argument("-icm", "--interior_classification_model",help="provide the interior classification model path")
			self.parser.add_argument("-emp","--extra_model_params",help="provide extra model parameters")




		def start_processing(self, command_line_options):
			processor = self.__process_command_line_args(command_line_options)
			processor.process()

		def __process_command_line_args(self, command_line_options):
			logger.info("Processing the command line args: {}".format(command_line_options))
			args = self.parser.parse_args(command_line_options)
			if args.download is not None:
				if os.path.exists(args.download) and os.path.isdir(str(args.download)):
					return Webscrapper(args.download)
				else:
					raise Exception("Expecting the option to be a directory")

			elif args.build is not None:
				if self.__is_directory(args.build):
					raise Exception("Model path must be a file")
				if args.label is None:
					raise Exception("Provide the labels")
				if args.learningrate is None:
					raise Exception("Provide the learning rate for the model.")
				if args.image_dim is None:
					raise Exception("Provide the image dimension as single value. Example: if image dimension is 64*64, provide the value as 64")
				# we are expecting comma separate values to be given here with the following conventions
				# model_type:restnet18,
				# k1:v1,k2:v2
				dict = {}

				if not args.extra_model_params is None:
					extra_params = args.extra_model_params
					model_params = extra_params.split(",")
					for entry in model_params:
						kv = entry.split(":")
						key = kv[0]
						value = kv[1]
						dict[key] = value

				if args.dataset is not None and args.epochs is not None:
					dict[MODEL_PATH]=args.build
					dict[DATASET] = args.dataset
					dict[EPOCHS] = args.epochs
					dict[LABEL] = args.label
					dict[LEARNING_RATE] = args.learningrate
					dict[IMAGE_DIM] = args.image_dim
					dict[BUILD] = True
					return Model.build_model(dict)
				else :
					raise Exception("Either dataset path or epochs is missing")
			elif args.compress is not None:
				if self.__is_directory(args.compress):
					logger.info("Compressing all the images from the directory:{}".format(args.compress))
					compress_image(args.compress)
				else:
					raise Exception("Expecting the option to be a directory")
			elif args.predict is not None:
				if not self.__is_directory(args.predict):
					raise Exception("Prediction folder must be a directory")
				if args.home_classification_model is None:
					raise Exception("Expecting the home classification model path")
				if self.__is_directory(args.home_classification_model):
					raise Exception("Home classification model path must be an hdf5 file")

				if args.exterior_classification_model is None:
					raise Exception("Expecting the exterior classification model path")
				if self.__is_directory(args.exterior_classification_model):
					raise Exception("Exterior classification model path must be an hdf5 file")

				if args.interior_classification_model is None:
					raise Exception("Expecting the interior classification model path")
				if self.__is_directory(args.interior_classification_model):
					raise Exception("Interior classification model path must be an hdf5 file")

				dict = {}
				dict[HOME_CLS_MODEL_PATH] = args.home_classification_model
				dict[INTERIOR_CLS_MODEL_PATH] = args.interior_classification_model
				dict[EXTERIOR_CLS_MODEL_PATH] = args.exterior_classification_model
				dict[TEST_DIR] = args.predict
				dict[PREDICT] =  True

				return Model.load_model_for_predict(dict)


			else:
				raise Exception("No valid options are presents. use main.py -h to get the list of valid options")

		def is_directory(self, path):
			return os.path.isdir(path)



