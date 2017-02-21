import sys


codegit_root = '/home/intuinno/codegit'

sys.path.insert(0, codegit_root)

import pandas as pd
import numpy as np
import os
import nltk
import scipy
import json
import cPickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TreebankWordTokenizer
import pdb

from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np

base_model = VGG19(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('block5_conv4').output)

TRAIN_SIZE = 16
TEST_SIZE = 2

annotation_path = '/local/wangxin/Data/Flickr8k_Dataset/Flickr8k_text/Flickr8k.token.txt'
flickr_image_path = '/local/wangxin/Data/Flickr8k_Dataset/Flicker8k_Dataset'

feat_path='feat/flickr8k'

def my_tokenizer(s):
    return s.split()

annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
annotations = annotations[:100]
annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])
annotations['image'] = annotations['image'].map(lambda x: os.path.join(flickr_image_path,x.split('#')[0]))

captions = annotations['caption'].values

words = nltk.FreqDist(' '.join(captions).split()).most_common()

wordsDict = {i+2: words[i][0] for i in range(len(words))}

# vectorizer = CountVectorizer(token_pattern='\\b\\w+\\b').fit(captions)
# dictionary = vectorizer.vocabulary_
# dictionary_series = pd.Series(dictionary.values(), index=dictionary.keys()) + 2
# dictionary = dictionary_series.to_dict()

# # Sort dictionary in descending order
# from collections import OrderedDict
# dictionary = OrderedDict(sorted(dictionary.items(), key=lambda x:x[1], reverse=True))

with open('dictionary.pkl', 'wb') as f:
    cPickle.dump(wordsDict, f)


images = pd.Series(annotations['image'].unique())
image_id_dict = pd.Series(np.array(images.index), index=images)

DEV_SIZE = 20 - TRAIN_SIZE - TEST_SIZE

caption_image_id = annotations['image'].map(lambda x: image_id_dict[x]).values
cap = zip(captions, caption_image_id)

# split up into train, test, and dev
all_idx = range(len(images))
np.random.shuffle(all_idx)
train_idx = all_idx[0:TRAIN_SIZE]
train_ext_idx = [i for idx in train_idx for i in xrange(idx*5, (idx*5)+5)]
test_idx = all_idx[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]
test_ext_idx = [i for idx in test_idx for i in xrange(idx*5, (idx*5)+5)]
dev_idx = all_idx[TRAIN_SIZE+TEST_SIZE:]
dev_ext_idx = [i for idx in dev_idx for i in xrange(idx*5, (idx*5)+5)]

## TRAINING SET

# Select training images and captions
if 0:
    images_train = images[train_idx]
    captions_train = captions[train_ext_idx]
    
    # Reindex the training images
    images_train.index = xrange(TRAIN_SIZE)
    image_id_dict_train = pd.Series(np.array(images_train.index), index=images_train)
    # Create list of image ids corresponding to each caption
    caption_image_id_train = [image_id_dict_train[img] for img in images_train for i in xrange(5)]
    # Create tuples of caption and image id
    cap_train = zip(captions_train, caption_image_id_train)
    
    for i in range(len(images_train)):
        
        image_files = images_train[i]
        if image_files!="/local/wangxin/Data/Flickr8k_Dataset/Flicker8k_Dataset/2258277193_586949ec62.jpg.1":
            img = image.load_img(image_files, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            block5_conv4_pool_features = model.predict(x)
            print np.shape(block5_conv4_pool_features)
            if i == 0:
                feat_flatten_list_train = scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), block5_conv4_pool_features)))
            else:
                feat_flatten_list_train = scipy.sparse.vstack([feat_flatten_list_train, scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), block5_conv4_pool_features)))])
        
            print "processing images %d "% (i)
    
    with open('data/toyset/flicker_8k_align.train.pkl', 'wb') as f:
        cPickle.dump(cap_train, f,-1)
        cPickle.dump(feat_flatten_list_train, f)
        pdb.set_trace()

if 0:
    ## TEST SET
    
    # Select test images and captions
    images_test = images[test_idx]
    captions_test = captions[test_ext_idx]
    
    # Reindex the test images
    images_test.index = xrange(TEST_SIZE)
    image_id_dict_test = pd.Series(np.array(images_test.index), index=images_test)
    # Create list of image ids corresponding to each caption
    caption_image_id_test = [image_id_dict_test[img] for img in images_test for i in xrange(5)]
    # Create tuples of caption and image id
    cap_test = zip(captions_test, caption_image_id_test)
    
    for i in range(len(images_test)):
        image_files = images_test[i]
        img = image.load_img(image_files, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        block5_conv4_pool_features = model.predict(x)
        print np.shape(block5_conv4_pool_features)
        if i == 0:
            feat_flatten_list_test = scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), block5_conv4_pool_features)))
        else:
            feat_flatten_list_test = scipy.sparse.vstack([feat_flatten_list_test, scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), block5_conv4_pool_features)))])
    
        print "processing images %d "% (i)
    
    
    with open('data/toyset/flicker_8k_align.test.pkl', 'wb') as f:
        cPickle.dump(cap_test, f)
        cPickle.dump(feat_flatten_list_test, f)

## DEV SET
if 1:
    # Select dev images and captions
    images_dev = images[dev_idx]
    captions_dev = captions[dev_ext_idx]
    
    # Reindex the dev images
    images_dev.index = xrange(DEV_SIZE)
    image_id_dict_dev = pd.Series(np.array(images_dev.index), index=images_dev)
    # Create list of image ids corresponding to each caption
    caption_image_id_dev = [image_id_dict_dev[img] for img in images_dev for i in xrange(5)]
    # Create tuples of caption and image id
    cap_dev = zip(captions_dev, caption_image_id_dev)
    
    for i in range(len(images_dev)):
        image_files = images_dev[i]
        img = image.load_img(image_files, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        block5_conv4_pool_features = model.predict(x)
        print np.shape(block5_conv4_pool_features)
        if i == 0:
            feat_flatten_list_dev = scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), block5_conv4_pool_features)))
        else:
            feat_flatten_list_dev = scipy.sparse.vstack([feat_flatten_list_dev, scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), block5_conv4_pool_features)))])
    
        print "processing images %d "% (i)
    
    
    with open('data/toyset/flicker_8k_align.dev.pkl', 'wb') as f:
        cPickle.dump(cap_dev, f)
        cPickle.dump(feat_flatten_list_dev, f)
