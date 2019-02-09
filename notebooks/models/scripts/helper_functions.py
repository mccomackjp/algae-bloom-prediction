import numpy as np
import pandas as pd
from math import floor
from keras import backend as K

def create_windows(dataset, window_size, shift, time_col, days_ahead=None, hours_ahead=None):
    """
    determines the window size for the data set
    :param dataset: The dataset to get windows for
    :param window_size: the size of the window
    :param shift: the amout to shift the window
    :param time_col: the time column to determine the window size on
    :param days_ahead: the numer of days ahead to look for the target window
    :param hours_ahead: the number of hours ahead to look for the target window
   
    :yield: the indexes for the next window 
    """
    start = dataset[time_col][0]
    
    if hours_ahead != None:
        end_delta = pd.Timedelta(window_size,unit='h')
    else: 
        end_delta = pd.Timedelta(window_size, unit='D')    
    if hours_ahead != None:
        ahead_delta = end_delta + pd.Timedelta(hours_ahead,unit='h')
    else:
        ahead_delta = end_delta + pd.Timedelta(days_ahead, unit='D')
    print(start)
    print(end_delta)
    print(ahead_delta)
    input()
    while ( start+ahead_delta < dataset[time_col][dataset.shape[0]-1]): 
        yield (  start , end_delta, ahead_delta )
        # shift the window 'shift' hour blocks of time
        start = update_indicies( shift, start )
        
def segment_dataset(dataset, time_col, shift=1, window_multiplier=2, days_ahead=1, hours_ahead=None):
    """
    Segments the dataset based on the parameters that are passed in.
    
    :param dataset: the dataset to segment into windows
    :param time_col: the name of the time column in the dataset
    :param window_multiplier: the size times larger for the window of features to be compared to the target. Default is twice the size
    :param hours_ahead: the number of hours ahead for the window to get back
    :param days_ahead: the number of days ahead of the window to get back. If no other  Default is 1

    :return: An array of Dataframes windowed for features and targets
    """
    
    if hours_ahead != None:
        window_size = floor(hours_ahead * window_multiplier)
    else:
        window_size = floor(days_ahead * window_multiplier)
    segments = []
    targets  = []
    if hours_ahead != None:
        for (start, end_delta, ahead_delta) in create_windows(dataset, window_size, shift, time_col, hours_ahead=hours_ahead):
            segments.append(dataset[start:start + end_delta])
            targets.append(dataset[start + end_delta: start+ahead_delta])
    else:
        # if no option is selected will default to days ahead
        for (start, end_delta, ahead_delta) in create_windows(dataset, window_size, shift, time_col, days_ahead=days_ahead):
            segments.append(dataset[start:start + end_delta])
            targets.append(dataset[start + end_delta: start+ahead_delta])
    return segments, targets

def update_indicies(shift, value):
    """
    Updates the indicies with the newest indicies based on the shift value passed in
    
    :param shift: the amount to shift the window by
    :param value: the value to shift
    
    :return: the next index for the specified value
    """
    return value + pd.Timedelta(shift, unit='h')
         
    
def create_class_predictions(pred):
    """
    Obtains the guesses for the model
    """
    retval = np.array([])
    for row in pred:
        max_value = (-1,-1)
        for index, value in enumerate(row):
            if value > max_value[1]:
                max_value = (index, value)
        retval = np.append(retval, max_value[0])
    return retval

def save_model(model, model_name, model_version):
    """
    Saves the model withe the specified parameters to the path located at 
    ./../../saved_models/<MODEL_NAME>/<MODEL_VERSION>
    
    @param model - the model to be saved
    @param model_name - the name you would want to model to be saved as
    @param model_versiuon - the version of the model.
    """

    # ignoring dropout for deployment
    K.set_learning_phase(0)
     
    # Set a file path to save the model in.
    tf_path = "./../../saved_models/{}/{}".format(model_name, model_version)
     
    # Get the session from the Keras back-end to save the model in TF format.
    with K.get_session() as sess:
        tf.saved_model.simple_save(sess, tf_path, inputs={'input': model.input}, outputs={t.name: t for t in model.outputs})
        
