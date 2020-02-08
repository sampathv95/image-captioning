import os
import re
import json
import pickle
import numpy as np
from pickle import load
from keras.models import Model
from keras.applications import vgg16, inception_v3
from keras.preprocessing.image import load_img, img_to_array


class ImageDataPreprocess():

    @staticmethod
    def get_pretrained_model(model_name='VGG16'):
        if model_name == 'VGG16':
            vgg_model = VGG16()
            #removing the top layer to get the vector
            vgg_model.layers.pop()
            vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-1].output)
            return vgg_model
        if model_name == 'InceptionV3':
            inception_model = inception_v3.InceptionV3(weights='imagenet')
            #removing the top layer to get the vector representation if the image
            inception_model.layers.pop()
            inception_model = Model(inputs=inception_model.inputs, outputs=inception_model.layers[-1].output)
            return inception_model

    @staticmethod
    def compute_save_features(pretrained_model, model_name='VGG16',img_dir='Flicker8k_Dataset',target_size=(224, 224), save=True, return_dict=False, pkl_name='pickle_files/image_features.pkl'):
        features_dict = {}
        for image_name in os.listdir(img_dir):
            image_id, image_ext = tuple(image_name.split('.'))
            # taking the images with jpg extension only
            if image_ext != 'jpg':
                continue
            image_path = img_dir+'/'+image_name
            # making the image size into (224, 224)
            image_resized = load_img(image_path, target_size = target_size)
            # converting the image into numpy array
            image_array = np.asarray(image_resized)
            # reshaping the array into tensor with batch sizes
            image_array_reshaped = image_array.reshape(1, image_array.shape[0], image_array.shape[1], image_array.shape[2])
            # pre processing image based on VGG16 dataset
            image_preprocessed = vgg16.preprocess_input(image_array_reshaped)
            if model_name == 'InceptionV3':
                image_preprocessed = inception_v3.preprocess_input(image_array_reshaped)
            # computing feature vector for the image
            image_feature_vector = pretrained_model.predict(image_preprocessed, verbose = 0)
            features_dict[image_id] = image_feature_vector
        if save:
            with open(pkl_name, 'wb') as pickle_file:
                pickle.dump(features_dict, pickle_file)
        if return_dict:
            return features_dict
    
class TextDataPreprocess():

    @staticmethod
    def get_descriptions_dict(desc_path='Flicker8k_text/Flickr8k.token.txt', clean=True, **kwargs):
        # intitializing an empty dictionary in which keys are image id and value is list of descriptions of that image
        descriptions_dict = {}
        # opening the file containing tokens and descriptions
        with open(desc_path, 'r') as captions_file:
            descriptions = captions_file.read()
        for line in descriptions.split('\n'):
            line_split = line.split()
            image_id, image_desc = line_split[0], line_split[1:]
            # removing the file extension from image_id
            image_id = image_id.split('.')[0]
            # if image_id is not already in dictionary, a new list is initialized for that image_id
            if image_id not in descriptions_dict:
                descriptions_dict[image_id] = []
            descriptions_dict[image_id].append(' '.join(image_desc))
        if clean:
            for key, desc_list in descriptions_dict.items():
                for i in range(len(desc_list)):
                    desc_sent = desc_list[i]
                    # removing all the punctuations
                    desc_sent = re.sub(r'[^\w\s]','',desc_sent)
                    desc_sent = desc_sent.split()
                    # removing all the words like a etc and lower casing all the alphabets
                    desc_sent = [word.lower() for word in desc_sent if len(word)>1 and word.isalpha()]
                    desc_list[i] = ' '.join(desc_sent)
        return descriptions_dict

    @staticmethod
    def get_vocab_set(descriptions_dict):
        vocab_set = set()
        for key, desc_list in descriptions_dict.items():
            [vocab_set.update(desc.split()) for desc in desc_list]
        return vocab_set

    @staticmethod
    def save_descriptions(descriptions_dict, path_to_save='descriptions.txt'):
        # initializing empty list to store all the descriptions
        all_desc = []
        for key, desc_list in descriptions_dict.items():
            for desc in desc_list:
                all_desc.append(key+' '+desc)
        # creating a string that holds each key and description pair in a single line
        data_to_file = '\n'.join(all_desc)
        with open(path_to_save, 'w') as desc_file:
            desc_file.write(data_to_file)