import os
from configparser import ConfigParser


class Config:
    '''
    TODO
    '''
    def __init__(self):
        self.config = ConfigParser()
        current_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(current_dir, 'config.cfg')
        with open(config_path, 'r') as config_file:
            self.config.read(config_file)