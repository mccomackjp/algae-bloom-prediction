from keras.layers import Dense, Dropout, Conv2D, Flatten, Activation, MaxPooling2D
from keras.models import Sequential
import keras as ks
'''
Creates a sequential model 

@param - filters:       The dimensionality of the output space. i.e. the number of otuput filters in the convolution
@param - kernal_size:   An integer or tuple/list of 2 integers, specifying the height and 
                          width of the 2d convolution window. Can be a single integer to 
                          specify the same for all spacial dimensions
@param - i_shape        The shape of the input.
@param - class_number:  The number of classes create it down into
@param - learning_r:    the learning rate to compile the model for.
'''
def create_model(filters, kernal_size, i_shape, class_number, learning_r, activation='relu', ):
    model = Sequential()
    model.add(Conv2D(filters, kernal_size, input_shape=i_shape, activation=activation, padding='same'))
    model.add(MaxPooling2D(pool_size=(4,4)))
    
    # creates adds another Convelutional layer with twice the number of filter layers, expanding the neurons
    model.add(Conv2D(filters * 2, (3,3), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(1,1)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    # compress it back down to the original passed in size
    model.add(Dense(filters))
    model.add(Dense(class_number, activation='softmax'))
    
    model.compile(loss=ks.losses.categorical_crossentropy,
                optimizer=ks.optimizers.Adam(lr=learning_r),
                metrics=['accuracy'])
    return model