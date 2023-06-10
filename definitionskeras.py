#
# # from nnom.scripts.nnom import *
# from nnom.scripts.nnom import *
from tensorflow.keras.utils import to_categorical

# import tensorflow.keras as keras
# from tensorflow.keras.models import Sequential,Model,load_model,save_model
# from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization,Flatten,Conv2D,MaxPooling2D,Input,LeakyReLU,ReLU,Softmax
# from tensorflow.keras.callbacks import TensorBoard
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.losses import categorical_crossentropy
# from tensorflow.keras import optimizers
# from tensorflow.keras.layers import Input
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Reshape
# from tensorflow.keras.layers import Convolution1D, MaxPooling1D, BatchNormalization,Conv2DTranspose,Concatenate
# from tensorflow.keras.layers import Lambda, concatenate


import keras
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential,Model,load_model,save_model
from keras.layers import Dense, Dropout, LSTM, BatchNormalization,Flatten,Conv2D,MaxPooling2D,Input,LeakyReLU,ReLU,Softmax
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy
from keras import optimizers
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Reshape
from keras.layers import Convolution1D, MaxPooling1D, BatchNormalization,Conv2DTranspose,Concatenate
from keras.layers import Lambda, concatenate