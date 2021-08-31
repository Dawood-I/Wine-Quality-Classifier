"""
Last modified Oct 2019
@author: mvm for INFR3700/MITS6800
"""
#imports
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import pandas as pd
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", delimiter=";") # reading data file and separating data by delimiter specified (;)
data["quality"].replace({3:0,4:1,5:2,6:3,7:4,8:5}, inplace=True) # change values of qualities from 3-8 to 0-5 for dense (6)
# Split train and test sets
#data["quality"].replace({5:0,6:1}, inplace=True)  #testing/ training with only qualities 5,6
#data = data[data.quality != 3]
#data = data[data.quality != 4]
#data = data[data.quality != 7]
#data = data[data.quality != 8]
from sklearn.model_selection import train_test_split # splitting data to train and test | data = 80% | testing set = 20%
train, test = train_test_split(data, test_size=0.2, random_state=42)
train_label = train.iloc[:, -1] # test/train label = quality column
x_train_prescaler = train.drop(['citric acid', "quality"], axis=1) #test/train data before scaler = all columns - quality,citric acid
test_label = test.iloc[:, -1]
x_test_prescaler = test.drop(['citric acid', "quality"], axis=1)

print(x_train_prescaler) # printing
#import
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # applying standard scaler
scaler.fit(x_train_prescaler)
x_train = scaler.transform(x_train_prescaler) # fitting data,test
x_test = scaler.transform(x_test_prescaler)



model = Sequential() # model sequential, and dense/dropout with different activations
model.add(Dense(100, activation='relu', input_dim=10)) # number of attributes used to train DNN))
model.add(Dropout(0.5))
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))

# Configure learning process
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])

# Train,fit model with training set
history = model.fit(x_train, train_label, epochs=20, batch_size=4, validation_data=(x_test, test_label), verbose=1)# verbose = 1 to display iterations, validation data to print val acc

# Evaluating on testing set
loss, acc = model.evaluate(x_test, test_label, batch_size=8)
print('Testing result: {0:2.5f}, acc: {1:2.3f}'.format(loss, acc)) # printing output, loss and accuracy



# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# plotting training and validation accuracy values
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
