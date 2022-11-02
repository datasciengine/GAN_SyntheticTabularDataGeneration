import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score,\
                            accuracy_score, balanced_accuracy_score,classification_report,\
                            plot_confusion_matrix, confusion_matrix
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
import tensorflow.keras.backend as K
from sklearn.utils import shuffle
import keras

np.random.seed(1635848)

class Generator(keras.Model):
    def __init__(self, latent_dim=32, out_shape=14, num_classes=2):
        super(Generator, self).__init__(name="generator")
        
        self.latent_dim = latent_dim
        self.out_shape = out_shape 
        self.num_classes = num_classes
        
        self.dense_in = Dense(128, use_bias=False, input_dim=self.latent_dim, name="Dense1")
        self.dense_out = Dense(self.out_shape, activation='tanh')
        self.dense1 = Dense(256)
        self.dense2 = Dense(512)
        self.dropout02 = Dropout(0.2)
        self.bn1 = BatchNormalization(momentum=0.4)
        self.bn2 = BatchNormalization(momentum=0.8)
        self.leaky_relu01 = LeakyReLU(alpha=0.1)
        
        
    def call(self, model_input):
        x = self.dense_in(model_input)
        x = self.dropout02(x)
        x = self.leaky_relu01(x)
        x = self.bn1(x)
        x = self.dense1(x)
        x = self.dropout02(x)
        x = self.leaky_relu01(x)
        x = self.bn2(x)
        x = self.dense2(x)
        x = self.dropout02(x)
        x = self.leaky_relu01(x)
        gen_sample = self.dense_out(x)
        return gen_sample