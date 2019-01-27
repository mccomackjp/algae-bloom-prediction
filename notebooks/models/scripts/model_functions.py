from keras.layers import Dense, Dropout, Conv2D, Flatten, Activation, MaxPooling2D
from keras.models import Sequential
import keras as ks
import numpy as np
import pandas as pd

'''
Creates a sequential model 

@param - filters:       The dimensionality of the output space. i.e. the number of otuput filters in the convolution
@param - kernal_size:   An integer or tuple/list of 2 integers, specifying the height and 
                          width of the 2d convolution window. Can be a single integer to 
                          specify the same for all spacial dimensions
@param - i_shape        The shape of the input.
@param - class_number:  The number of classes create it down into
@param - learning_r:    the learning rate to compile the model for.
@param - layers:        A list of keras.layer objects to add to the the model.
'''
def create_model(filters, kernal_size, i_shape, class_number, learning_r, activation='relu', layers=[]):
    model = Sequential()
    
    # use default layers if the list of layers is empty.
    layers = layers if layers else [Conv2D(filters, kernal_size, input_shape=i_shape, activation=activation, padding='same'),
                  MaxPooling2D(pool_size=(4,4)), Conv2D(filters * 2, (3,3), activation='relu',padding='same'),
                  MaxPooling2D(pool_size=(1,1)), Flatten(), Dropout(0.2), Dense(filters), 
                  Dense(class_number, activation='softmax')]
    
    for layer in layers:
        model.add(layer)
    
    model.compile(loss=ks.losses.categorical_crossentropy,
                optimizer=ks.optimizers.Adam(lr=learning_r),
                metrics=['accuracy'])
    return model


def create_model_mult(mult, kernal_size, i_shape, class_number, learning_r, activation='relu'):
    model = Sequential()
    model.add(Conv2D(44, kernal_size, input_shape=i_shape, activation=activation, padding='same'))
    model.add(MaxPooling2D(pool_size=(4,4)))
    print(int(44 * mult))
    # creates adds another Convelutional layer with twice the number of filter layers, expanding the neurons
    model.add(Conv2D( int(44 * mult) , (3,3), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(1,1)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    # compress it back down to the original passed in size
    model.add(Dense(44))
    model.add(Dense(class_number, activation='softmax'))
    
    model.compile(loss=ks.losses.categorical_crossentropy,
                optimizer=ks.optimizers.Adam(lr=learning_r),
                metrics=['accuracy'])
    return model

    
'''
determines the window size for the data set
@param dataset - The dataset to get windows for
@param window_size - the size of the window  
@param shift - the amout to shift the window
@param point_ahead - the amount of points ahead to return
'''
def __create_windows(dataset, window_size, shift, points_ahead=0):
    start = 0
    while start+window_size < dataset.shape[0] and start + window_size + points_ahead < dataset.shape[0]: 
        yield (int(start), int(start+window_size), int(start+window_size+points_ahead))
        # shift the window shift blocks of time
        start += shift
        if start % 100 == 0:
            print('Window Segmentation {0:.2f}% done'.format(((start+window_size) / dataset.shape[0]) * 100 ))
'''
Segments the dataset based on the parameters that are passed in.
@param dataset - the dataset to segment into window
@param columns - the array of columns from the dataset to be looked at
@param window_size - the size of the window you would like to be looked at. Defualt is 10
@param pts_ahead - the number of points ahead of the window to get back. Default is 0
'''
def segment_dataset(self, dataset, columns, target, window_size=10, pts_ahead=0):
    print(dir(self))
    print('WINDOW SIZE', window_size)
    print('NUMBER OF COULUMNS',len(columns))
    segments = np.empty((0, window_size, len(columns)))
    labels = np.empty((0))
    count = 0
    for (start, end, pts_ahd) in Helper_Functions.__create_windows(dataset, window_size, 1, points_ahead=pts_ahead):
        count+=1
        values = dataset[columns][start:end]
        if(values.shape[0] == window_size):
            segments = np.vstack([segments, np.stack([values])])
            # Takes the larger of the two variables if there are more than one. 
            # This makes it more likly to predict a bloom. Can be changed to iloc[0] to
            # be less likly to predict a bloom (more 0s in the label array)
            if pts_ahead == 0:
                targets = np.append(targets, dataset[target][start:end].mode().iloc[-1])
            else:
                targets = np.append(targets, dataset[target][end:pts_ahd].mode().iloc[-1])
        else:
            print("No more Windows available... Exiting")
            break
    return (segments, targets)