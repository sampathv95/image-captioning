import numpy as np
from keras.applications import vgg16, inception_v3
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img, img_to_array
from sb_preprocess import ImageDataPreprocess, TextDataPreprocess

class Inference():
    @staticmethod
    def get_image_features(inf_img_path):
        feature_extractor_model = ImageDataPreprocess.get_pretrained_model('InceptionV3')
        # the target size of test/new image should be same as the train image
        target_size = (299, 299)
        image_resized = load_img(inf_img_path, target_size = target_size)
        # converting the image into numpy array
        image_array = np.asarray(image_resized)
        # reshaping the array into tensor with batch sizes
        image_array_reshaped = image_array.reshape(1, image_array.shape[0], image_array.shape[1], image_array.shape[2])
        # pre processing image based on VGG16 dataset
        image_preprocessed = inception_v3.preprocess_input(image_array_reshaped)
        # computing feature vector for the image
        image_feature_vector = feature_extractor_model.predict(image_preprocessed, verbose = 0)
        return image_feature_vector
    
    @staticmethod    
    def word_for_id(num, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == num:
                return word
        return None
 
    # generate a description for an image
    @staticmethod
    def generate_desc(model, mappings_dict,reverse_mappings_dict, img, max_length):
        # seed the generation process
        in_text = 'startseq'
        # iterate over the whole length of the sequence
        for i in range(max_length):
            # integer encode input sequence
            sequence = [mappings_dict[word] for word in in_text.split() if word in mappings_dict]
            # pad input
            sequence = pad_sequences([sequence], maxlen=max_length)
            # predict next word
            y_pred = model.predict([img,sequence], verbose=0)
            # convert probability to integer
            y_pred = np.argmax(y_pred)
            # map integer to word
            word = reverse_mappings_dict[y_pred]
            # stop if we cannot map the word
            if word is None:
                break
            # append as input for generating the next word
            in_text += ' ' + word
            # stop if we predict the end of the sequence
            if word == 'endseq':
                break
        return in_text