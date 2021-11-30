
import abc
from com.opensource.cnn.utils.cnn_logger import get_logger
logger = get_logger(__name__)


#base class  for individual processor to extend from
class AbstractProcessor(abc.ABC):
	def __init__(self):
		pass
	"""
	Base class for all the processors
	"""
	@abc.abstractmethod
	def process(self):
		pass




