# code reference
# https://github.com/victoresque/pytorch-template/blob/41dc06f6f8f3f38c6ed49f01ff7d89cd5688adc4/logger/logger.py

import logging
import logging.config
from pathlib import Path
from utils import read_json


def setup_logging(save_dir, log_config='src/jsons/logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)