#Import 'Sequential' from 'keras.models'
from keras.models import Sequential
#Import 'Dense' from 'keras.layers'
from keras.layers import Dense
#Inport 'SGD' from 'keras.optimizers' (Stochastic Gradient Descenc)
from keras.optimizers import SGD
#Import 'numpy" for using to numpy arrays
import numpy as np
#Import 'Activation' from 'keras.core'
from keras.layers.core import Activation
#Import 'Droupout' from 'keras.core'
from keras.layers.core import Dropout
#Import 'to_categorical' from 'keras.utils.np_utils'
from keras.utils.np_utils import to_categorical
#Import 'cv2' for making/creating to model
import cv2
#Import 'ploy' from 'keras.utils.visualize_util'
#from keras.utils.visualize_util import plot
#Import 'Image' from 'IPython.display'
#from IPython.display import Image
#____________________________________________________________________________________________________
#(index numbers) - Train Data 
#1: Bean (0-99)
#2: Cake (100-199)
#3: Candy (200-299)
#4: Chips (300-399)
#5: Chocolate (400-499)
#6: Coffee (500-599)
#7: Honey (600-699)
#8: Jam (700-799)
#9: Juice (800-899)
#10: Milk (900-999)
#11: Nuts (1000-1099)
#12: Oil (1100-1199)
#13: Pasta (1200-1299)
#14: Rice (1300-1399)
#15: Soda (1400-1499)
#16: Spices (1500-1599)
#17: Tea (1600-1699)
#18: Tomato Sauce (1700-1799)
#19: Vinegar (1800-1899)
#20: Water (1900-1999)
#____________________________________________________________________________________________________
#(index numbers) - Test Data 
#1: Bean (0-19)
#2: Cake (20-39)
#3: Candy (40-59)
#4: Chips (60-79)
#5: Chocolate (80-99)
#6: Coffee (100-119)
#7: Honey (120-139)
#8: Jam (140-159)
#9: Juice (160-179)
#10: Milk (180-199)
#11: Nuts (200-219)
#12: Oil (220-239)
#13: Pasta (240-259)
#14: Rice (260-279)
#15: Soda (280-299)
#16: Spices (300-319)
#17: Tea (320-339)
#18: Tomato Sauce (340-359)
#19: Vinegar (360-379)
#20: Water (380-399)
#____________________________________________________________________________________________________
#For name of X to pictures
type_of_x = []
for i1 in range(20):
    for l1 in range(100):
        type_of_x.append(i1)
#____________________________________________________________________________________________________
#For "one-hot encoding" -X TRAIN
x_train_ohe = to_categorical(type_of_x)
#____________________________________________________________________________________________________
#For name of Y to pictures
type_of_y = []
for i2 in range(20):
    for l2 in range(20):
        type_of_y.append(i2)
#____________________________________________________________________________________________________
#For "one-hot encoding" -Y TRAIN
y_testing_ohe = to_categorical(type_of_y)
#____________________________________________________________________________________________________
#For get training data ( There are 65536 x values for each training image. )
x = 1
training = []
while(x<=2000):
    img = cv2.imread("C:\\Users\\Mert\\Desktop\\Training20x100\\pictures ("+ str(x) +').png',0)
    training.append(img)
    x = x + 1
#____________________________________________________________________________________________________
#One-dimensional array construction for "training x"
for m in range(2000):
    training[m] = training[m].reshape(65536,1)
#____________________________________________________________________________________________________
#For get test data ( There are 65536 x values for each testing image. )
x = 1
testing = []
while(x<=400):
    img = cv2.imread("C:\\Users\\Mert\\Desktop\\Test20x20\\pictures ("+ str(x) +').png',0)     
    testing.append(img)
    x = x + 1    
#____________________________________________________________________________________________________
#One-dimensional array construction for "testing x"
for n in range(400):
    testing[n] = testing[n].reshape(65536,1)
#____________________________________________________________________________________________________
#For converting to numpy array
train = np.array(training)
test = np.array(testing)
#____________________________________________________________________________________________________
#Converting 3D to 2D
train1 = train.reshape(65536,2000)
test1 = test.reshape(65536,400)
#____________________________________________________________________________________________________
#Change the Data Type to float32
train1 = train1.astype("float32")
test1  = test1.astype("float32")
#____________________________________________________________________________________________________
#Normalize the data such that all pixels in range [0,1]
train1 /=255 # Uint8 has max 255
test1  /=255 # Uint8 has max 255
#____________________________________________________________________________________________________
#MODEL DESIGN
#Initialize the model
model = Sequential()
#Add input layer
model.add(Dense(10,input_dim=1,init = 'uniform',activation='relu'))
#Add hidden layer -using tanh function-
model.add(Dense(10, init='uniform'))
model.add(Activation('tanh'))
#Add dropout to avoid "overclocking"
model.add(Dropout(0.5))
#Add hidden layer -using Rectified Linear Unit(relu) function-
model.add(Dense(10, init='uniform'))
model.add(Activation('relu'))
#Add hidden layer -using softmax function- For OHE (one-hot encoding)
model.add(Dense(10, init='uniform'))
model.add(Activation('softmax'))
#____________________________________________________________________________________________________
#For "optimization algorithm"
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#____________________________________________________________________________________________________
#For "compile"
model.compile(loss = 'categorical_crossentropy',optimizer = sgd)
#____________________________________________________________________________________________________
#For "fit"
model.fit(train1,x_train_ohe,batch_size = 256,epochs = 50,verbose = 1, validation_split = (test,y_testing_ohe))