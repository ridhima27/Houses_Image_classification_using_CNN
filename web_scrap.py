
import os
import shutil

from com.opensource.cnn.processors.processors import AbstractProcessor
from com.opensource.cnn.utils.cnn_logger import get_logger
import requests
import zipfile
import urllib
from urllib import request
from urllib.request import Request
import random
from bs4 import BeautifulSoup
import re


logger = get_logger(__name__)


def get_url_file(file_name):
	"""
	Method to find the url file from the resource folder
	"""
	curr_directory = os.path.dirname(__file__)
	mb_url_path = os.path.join(curr_directory, "..", "resources", file_name)
	return mb_url_path

def create_output_directory(output_dir,dir_name):
	output_path = os.path.join(output_dir,dir_name)
	os.mkdir(output_path)
	return output_path

def download_data_from_github(download_path):
	"""
	Method to download the housing data from the git hub url
	"""
	f = open(get_url_file('housing_data_github_url'))
	for url in f:
		logger.info("Processing url:{}".format(url))
		__download_and_segregate(url,download_path)


def __download_and_segregate(self, url, output_dir):
	"""
	Download the data from the github and segregate them into following
	1. building_classification
		1.1 exterior
			all frontal images
		1.2 interior
			all images belonging kitchen, bathroom, and bedroom
	2. interior_classification
		2.1 kitchen
			all kitchen images
		2.2 bathroom
			all bathroom images
		2.3 bedroom
			all bedroom images
	"""

	temp_dir = os.path.join(output_dir, "temp_dir")
	if os.path.exists(temp_dir):
		shutil.rmtree(temp_dir)

	os.mkdir(temp_dir)
	file_name = os.path.join(temp_dir, "data.zip")

	logger.info("Downloading data from url:{0} and saving the results at {1}".format(url, output_dir))
	try:
		response = requests.get(url, verify=False)
		with open(file_name, mode='wb') as localfile:
			localfile.write(response.content)
	except Exception as e:
		print("Received exception {0} while downloading data from url {1}".format(e, url))
		raise Exception("Received exception while downloading the files from url")
	zip_ref = zipfile.ZipFile(file_name, 'r')
	zip_ref.extractall(temp_dir)

	building_classification_dir = os.path.join(output_dir, 'building_classification')
	exterior_dir = os.path.join(building_classification_dir,"exterior")
	interior_dir = os.path.join(building_classification_dir,"interior")

	interior_classification_dir = os.path.join(output_dir, 'interior_classification')
	kitchen_dir = os.path.join(interior_classification_dir, "kitchen")
	bedroom_dir = os.path.join(interior_classification_dir, "bedroom")
	bathroom_dir = os.path.join(interior_classification_dir, "bathroom")

	#delete the building classification directory if exist
	if os.path.exists(building_classification_dir):
		shutil.rmtree(building_classification_dir)

	#delete the interior_classification_dir if exist
	if os.path.exists(interior_classification_dir):
		shutil.rmtree(interior_classification_dir)

	#now create all the directories
	os.mkdir(building_classification_dir)
	os.mkdir(exterior_dir)
	os.mkdir(interior_dir)

	os.mkdir(interior_classification_dir)
	os.mkdir(kitchen_dir)
	os.mkdir(bedroom_dir)
	os.mkdir(bathroom_dir)

	file_list = []
	# list comprehension
	[[file_list.append(os.path.join(root, f)) for root, dirs, files in os.walk(temp_dir) for f in files if
	  f.__contains__("jpg") ]]

	for file in file_list:
		if file.__contains__("kitchen"):
			#copy to kitchen directory
			shutil.copy(file, kitchen_dir)
			#copy to interior bucket as well
			shutil.copy(file,interior_dir)
		elif file.__contains__("bedroom"):
			shutil.copy(file, bedroom_dir)
			# copy to interior bucket as well
			shutil.copy(file, interior_dir)
		elif file.__contains__("bathroom"):
			shutil.copy(file, bathroom_dir)
			shutil.copy(file, interior_dir)
		elif file.__contains__("frontal"):
			shutil.copy(file, exterior_dir)

	# remove the temp directory here
	if os.path.exists(temp_dir):
		shutil.rmtree(temp_dir)

	logger.info("Data set download and segregated into appropriate classes")

def download_images_from_magic_bricks(output_dir):
	"""
	Method to download the data from the magicbricks.com
	"""
	f = open(get_url_file('magic_bricks_url'))
	op_path = create_output_directory(output_dir, 'magic_bricks')
	for url in f:
		logger.info("Processing url:{} from magicbricks.com".format(url))
		__download_images(url, op_path)


def __download_images(url,input_path):
		"""
		Method to download the images from magic bricks
		"""
		try:
			req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
			page = request.urlopen(req).read()
			soup = BeautifulSoup(page)
			# soup.prettify()
			print(soup)
			tags = soup.find_all('img')
			for tag in tags:
				if not tag.get('data-src') is None:
					file_url = tag['data-src']
					print(file_url)
					fName = file_url.split("/")[-1]
					file_name = os.path.join(input_path, fName)
					if os.path.exists(file_name):
						rand_int = random.randint(2000, 5000)
						file_name = os.path.join(input_path, fName + "_" + rand_int)
					try:
						urllib.request.urlretrieve(file_url, file_name)
					except Exception as e:
						print("Received exception {0} while downloading data from url {1}".format(e, url))

		except Exception as e:
			logger.error("Exception received while downloading images from url:{}".format(url))


def __download_images_from_hc(url, input_path):
	"""
	Method to download the images from hc (housing.com)
	"""
	try:
		logger.info("Downloading images")
		regex_src = r"(?<=[\"]src[\"]:{1})(.*?)(?:jpg|png|jpeg)"
		regex_all_cities = r"(?<=[\"]buy[\"]:{1})(?:[\"](.*?)[\"])"
		regex_page_no = r"(?<=[\"]numPages[\"]:{1})(.*?)(?=[}])"
		req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
		page = request.urlopen(req).read()
		soup = BeautifulSoup(page)
		all_city_links = re.findall(regex_all_cities, str(soup))
		for links in all_city_links:
			logger.info("Processing city link : {0}".format(links))
			req1 = Request(links, headers={'User-Agent': 'Mozilla/5.0'})
			page1 = request.urlopen(req1).read()
			soup1 = BeautifulSoup(page1)
			page_num = re.findall(regex_page_no, str(soup1))
			image_num = 0
			for i in range(1, int(page_num[0])):
				logger.info("Total images downloaded till now: {}".format(image_num))
				link1 = links + "?page=" + str(i)
				req2 = Request(link1, headers={'User-Agent': 'Mozilla/5.0'})
				page2 = request.urlopen(req2).read()
				soup2 = BeautifulSoup(page2)
				matches = re.finditer(regex_src, str(soup2), re.MULTILINE)
				for matchNum, match in enumerate(matches, start=1):
					file_url = match.group()
					logger.info("Processing image url {0} from city {1}".format(file_url, links))
					file_url = file_url.replace("/version/", "/medium/")
					file_url = file_url.replace("'", "").replace('"', '')
					logger.info("Modified image url {0}".format(file_url))
					fName = file_url.split("/")[-1]
					#	fName="temp.jpeg"
					file_name = os.path.join(input_path,fName)
					if os.path.exists(file_name):
						rand_int = random.randint(2000, 5000)
						file_name = os.path.join(input_path, str(rand_int) + "_" + fName)
					try:
						logger.info("Trying to download image:{0}".format(file_url))
						# urllib.parse.urlencode(file_url)
						urllib.request.urlretrieve(file_url, file_name)
						logger.info("Successfully downloaded the image {0}".format(file_url))
						image_num = image_num + 1
					except Exception as e:
						logger.info(
								"Received exception {0} while downloading data from url {1}".format(e, file_url))

		logger.info("Total number of images dowmaloaded from: {0}".format(image_num))


	except Exception as e:
		logger.error("Exception received while downloading images from url:{}".format(url))

def download_images_from_housing_com(output_dir):
	"""
	Downloading the images from housing.com
	"""
	f = open(get_url_file('housing_com_url'))
	op_path = create_output_directory(output_dir,"housing_com")

	for url in f:
		logger.info("Processing url:{} from housing.com".format(url))
		__download_images_from_hc(url, op_path)


class Webscrapper(AbstractProcessor):
	"""
	class to scrap the web pages and download the images
	"""
	def __init__(self,path):
		#super.__init__()
		self.download_path = path


	def process(self):
		logger.info("Downloading images from git hub")
		#download_data_from_github(self.download_path)

		logger.info("Downloading images from magicbricks.com")
		#download_images_from_magic_bricks(self.download_path)

		logger.info("Downloading images from housing.com")
		download_images_from_housing_com(self.download_path)