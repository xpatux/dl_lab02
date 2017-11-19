import logging
import logging.handlers
import sys
import os

logger = logging.getLogger(__name__)



def set_logger(default_level=logging.DEBUG,
               default_path='trace.log'):

    _path = os.path.join(os.path.dirname(__file__),default_path)

    handlers = [logging.StreamHandler(),
                logging.handlers.RotatingFileHandler(_path,
                                                 backupCount=10)]
        
    logging.basicConfig(level=default_level,
                        handlers = handlers)