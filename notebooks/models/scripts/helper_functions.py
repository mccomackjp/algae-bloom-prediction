import numpy as np
import pandas as pd
from keras import backend as K

def create_windows(dataset, window_size, shift, days_ahead, time_col):
    '''
    determines the window size for the data set
    @param dataset - The dataset to get windows for
    @param window_size - the size of the window
    @param shift - the amout to shift the window
    @param point_ahead - the amount of points ahead to return
    '''
    
    start = 0
    end = start + window_size
     
    while True:
        start_time = dataset[time_col][start]
        end_time = dataset[time_col][end]
        elapsed = end_time - start_time
        if elapsed.days == window_size:
            print("days ahead {} window size {}".format(elapsed.days, window_size))
            input()
            break
        end += 1
    ahead = end + 1
    while True:
        start_time = dataset[time_col][end]
        end_time = dataset[time_col][ahead]
        elapsed = end_time - start_time
        if elapsed.days == days_ahead:
            print("elapsed {} window size {}".format(elapsed.days, days_ahead))
            input()
            break
        ahead += 1
    print("Starting the window")
    while (ahead < dataset.shape[0]): 
        yield ( int( start ), int(end), int(ahead))
        # shift the window 'shift' blocks of time
        start += shift
        end += shift
        ahead += shift
        if start % 500 == 0:
            print('Window Segmentation {0:.2f}% done'.format(((ahead) / dataset.shape[0]) * 100 ))
def segment_dataset(dataset, time_col, window_size=2, days_ahead=1):
    '''
    Segments the dataset based on the parameters that are passed in.
    @param dataset - the dataset to segment into windows
    @param columns - the array of columns from the dataset to be looked at
    @param window_size - the size of the window (in days) you would like to be looked at. Defualt is 2
    @param days_ahead - the number of days ahead of the window to get back. Default is 1
    '''
    print('WINDOW SIZE', window_size)
    print('LOOKING AHEAD {} day(s)'.format(days_ahead))
    segments = []
    targets = []
    for (start, end, ahead) in create_windows(dataset, window_size, 1, days_ahead, time_col):
        segments.append(dataset.iloc[[start, end],:])
        targets.append(dataset.iloc[[end,ahead],:])
    return np.array(segments), np.array(targets)

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
        
