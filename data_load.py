import os
import re
import json
import numpy as np
from pickle import load

class DataLoad():

    @staticmethod
    def load_ids(path_to_ids):
        # load all the unique image ids from the given id path
        with open(path_to_ids, 'r') as id_file:
            id_str = id_file.read()
        id_list = [line.split('.')[0] for line in id_str.split('\n') if len(line)>=1]
        return list(set(id_list))
    
    @staticmethod
    def load_descriptions(path_to_ids, descriptions_path):
        # loading all the unique ids in given id path 
        id_list = DataLoad.load_ids(path_to_ids)
        # load all the descriptions from the given description path
        with open(descriptions_path, 'r') as description_file:
            descriptions_str = description_file.read()
        # filter descriptions that are in the required id list
        descriptions_dict = {}
        for line in descriptions_str.split('\n'):
            tokens_in_line = line.split()
            idx, desc = tokens_in_line[0], tokens_in_line[1:]
            if idx in id_list:
                if idx not in descriptions_dict:
                    descriptions_dict[idx] = []
                descriptions_dict[idx].append('startseq '+' '.join(desc)+' endseq')
        return descriptions_dict

    @staticmethod
    def load_photo_features(path_to_ids, features_path):
        # loading all the unique ids in given id path 
        id_list = DataLoad.load_ids(path_to_ids)
        #load all the photo features from the given
        with open(features_path, 'rb') as features_file:
            all_features = load(features_file)
        features_dict = {key: all_features.get(key) for key in id_list}
        return features_dict
