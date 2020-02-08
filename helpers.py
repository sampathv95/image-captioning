import os
import re
import json
import pickle
import numpy as np
from pickle import load
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class Helpers():

    @staticmethod
    def dict_to_list(descriptions_dict):
        all_desc = []
        for key, desc_list in descriptions_dict.items():
            all_desc.extend(desc_list)
        return all_desc
    
    @staticmethod
    def max_sentence_length(descriptions_dict):
        all_desc = Helpers.dict_to_list(descriptions_dict)
        max_len = max([len(desc.split()) for desc in all_desc])
        return max_len
    
    @staticmethod
    def get_mappings(descriptions_dict, freq_threshold=10):
        all_desc = Helpers.dict_to_list(descriptions_dict)
        word_counts, word_to_id, id_to_word = {}, {}, {}
        for desc in all_desc:
            for word in desc.split():
                word_counts[word] = word_counts.get(word, 0)+1
        vocab_set = [word for word in word_counts if word_counts[word]>=freq_threshold]
        # initializing id
        idx = 1
        for word in vocab_set:
            word_to_id[word] = idx
            id_to_word[idx] = word
            idx += 1
        return word_counts, word_to_id, id_to_word
    
    @staticmethod
    def get_pairs(descriptions_dict, photo_features, mappings_dict, max_length):
        vocab_size = len(mappings_dict) + 1
        X1, X2, y = [], [], []
        # walk through each image identifier
        for key, desc_list in descriptions_dict.items():
            # walk through each description for the image
            for desc in desc_list:
                # encode the sequence
                seq = [mappings_dict[word] for word in desc.split() if word in mappings_dict]
                # split one sequence into multiple X,y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo_features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
        return np.array(X1), np.array(X2), np.array(y)
    
    @staticmethod
    def sequences_to_embeddings(seq, embeddings_dict):
        train_3d = np.zeros(shape=(seq.shape[0], seq.shape[1], embedding_len))
        for i in range(len(train_3d)):
            for j in range(len(train_3d[i])):
                word = index_word_dict.get(seq[i][j])
                if word is not None:
                    if word in ['start', 'end']:
                        if word=='startseq':
                            train_3d[i][j] = np.ones(shape=(100))
                        else:
                            train_3d[i][j] = np.array([2]*100)
                    else:
                        embedding = embeddings_dict.get(word)
                        if embedding is not None:
                            train_3d[i][j] = embedding
                        else:
                            train_3d[i][j] = np.zeros(shape=(100))
                else:
                    train_3d[i][j] = np.zeros(shape=(100))
        return train_3d