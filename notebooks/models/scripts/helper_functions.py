import numpy as np
import pandas as pd
from keras import backend as K

'''
determines the window size for the data set
@param dataset - The dataset to get windows for
@param window_size - the size of the window  
@param shift - the amout to shift the window
@param point_ahead - the amount of points ahead to return
'''
def create_windows(dataset, window_size, shift, points_ahead=0):
    start = 0
    while start+window_size < dataset.shape[0] and ((start + window_size + points_ahead) < dataset.shape[0]): 
        yield (int(start), int(start+window_size), int(start+window_size+points_ahead))
        # shift the window shift blocks of time
        start += shift
        if start % 100 == 0:
            print('Window Segmentation {0:.2f}% done'.format(((start+window_size+points_ahead) / dataset.shape[0]) * 100 ))
'''
Segments the dataset based on the parameters that are passed in.
@param dataset - the dataset to segment into window
@param columns - the array of columns from the dataset to be looked at
@param window_size - the size of the window you would like to be looked at. Defualt is 10
@param pts_ahead - the number of points ahead of the window to get back. Default is 0
'''
def segment_dataset(dataset, columns, target, window_size=10, pts_ahead=0):
    print('WINDOW SIZE', window_size)
    print('NUMBER OF COULUMNS',len(columns))
    print('LOOKING AHEAD {} points'.format(pts_ahead))
    segments = np.empty((0, window_size, len(columns)))
    if pts_ahead == 0:
        targets = np.empty((0))
    else:
        targets = np.empty((0,pts_ahead))
    for (start, end, pts_ahd) in create_windows(dataset, window_size, 1, points_ahead=pts_ahead):
        values = dataset[columns][start:end]
        if(values.shape[0] == window_size):
            segments = np.vstack([segments, np.stack([values])])
            # Takes the larger of the two variables if there are more than one. 
            # This makes it more likly to predict a bloom. Can be changed to iloc[0] to
            # be less likly to predict a bloom (more 0s in the label array)
            if pts_ahead == 0:
                targets = np.append(targets, dataset[target][start:end].values[-1])
            else:
                targets = np.vstack([targets, dataset[target][end:pts_ahd].values])
        else:
            print("No more Windows available... Exiting")
            break
    return (segments, targets)

'''
Obtains the guesses for the model
'''
def create_class_predictions(pred):
    retval = np.array([])
    for row in pred:
        max_value = (-1,-1)
        for index, value in enumerate(row):
            if value > max_value[1]:
                max_value = (index, value)
        retval = np.append(retval, max_value[0])
    return retval

'''
Saves the model withe the specified parameters to the path located at 
./../../saved_models/<MODEL_NAME>/<MODEL_VERSION>

@param model - the model to be saved
@param model_name - the name you would want to model to be saved as
@param model_versiuon - the version of the model.
'''
def save_model(model, model_name, model_version):
    # ignoring dropout for deployment
    K.set_learning_phase(0)
     
    # Set a file path to save the model in.
    tf_path = "./../../saved_models/{}/{}".format(model_name, model_version)
     
    # Get the session from the Keras back-end to save the model in TF format.
    with K.get_session() as sess:
        tf.saved_model.simple_save(sess, tf_path, inputs={'input': model.input}, outputs={t.name: t for t in model.outputs})