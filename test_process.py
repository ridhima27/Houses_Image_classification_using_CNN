from unittest import TestCase

from os import path

from com.opensource.cnn.utils.process import Processor


class Test(TestCase):



	def test_directory_present(self):
		"""
		Testing the positive scenario where the path is directory
		"""
		p = Processor()
		curr_directory = path.dirname(__file__)

		test_path  = path.join(curr_directory,"resources")
		result = p.is_directory(test_path)
		self.assertEqual(True, result)

	def test_directory_not_present(self):
		"""
		Testing the positive scenario where the path is directory
		"""
		p = Processor()
		curr_directory = path.dirname(__file__)
		#this path doesn't exist. and hence assert should be false
		test_path  = path.join(curr_directory,"res")
		result = p.is_directory(test_path)
		self.assertEqual(False, result)



