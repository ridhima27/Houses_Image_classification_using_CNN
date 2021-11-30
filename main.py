

import sys

from com.opensource.cnn.utils.process import Processor


def main():
	processor = Processor()
	processor.start_processing(sys.argv[1:])



if __name__ == "__main__":
	main()
