import numpy as np
import pandas as pd
from math import floor
from keras import backend as K

def create_windows(dataset, window_size, shift, time_col, days_ahead=None, hours_ahead=None, weeks_ahead=None):
    """
    determines the window size for the data set
    :param dataset: The dataset to get windows for
    :param window_size: the size of the window
    :param shift: the amout to shift the window
    :param time_col: the time column to determine the window size on
    :param days_ahead: the numer of days ahead to look for the target window
    :param hours_ahead: the number of hours ahead to look for the target window
    :param weeks_ahead: the number of weeks ahead to look for the target window
   
    :yield: the indexes for the next window 
    """
    
    start = 0
    end = 1
    while end < dataset.shape[0]:
        start_time = dataset[time_col][start]
        end_time = dataset[time_col][end]
        elapsed = end_time - start_time
        
        if hours_ahead != None:
            if floor(elapsed.seconds /3600) == window_size:
                break
        elif weeks_ahead != None:
            if floor(elapsed.days/7) == window_size:
                break
        else: 
            if elapsed.days == window_size:
                break
        end += 1
    ahead = end + 1
    while ahead < dataset.shape[0]:
        start_time = dataset[time_col][end]
        end_time = dataset[time_col][ahead]
        elapsed = end_time - start_time
        
        if hours_ahead != None:
            if floor(elapsed.seconds / 3600) == hours_ahead:
                break
        elif weeks_ahead != None:
            if floor(elapsed.days / 7) == weeks_ahead:
                break
        else:
            if elapsed.days == days_ahead:
                break
        ahead += 1
    while (ahead < dataset.shape[0]): 
        yield ( int( start ), int(end), int(ahead))
        # shift the window 'shift' hour blocks of time
        start = update_indicies(dataset, time_col, shift, start)
        end   = update_indicies(dataset, time_col, shift, end)
        ahead = update_indicies(dataset, time_col, shift, ahead)
        if start % 200 == 0:
            print('Data Segmentation {0:.2f}% complete'.format(((ahead) / dataset.shape[0]) * 100 ))
            
def segment_dataset(dataset, time_col, shift=1, window_multiplier=2, days_ahead=1, hours_ahead=None, weeks_ahead=None):
    """
    Segments the dataset based on the parameters that are passed in.
    
    :param dataset: the dataset to segment into windows
    :param time_col: the name of the time column in the dataset
    :param window_multiplier: the size times larger for the window of features to be compared to the target. Default is twice the size
    :param hours_ahead: the number of hours ahead for the window to get back
    :param weeks_ahead: the number of weeks ahead to look.
    :param days_ahead: the number of days ahead of the window to get back. If no other  Default is 1

    :return: An array of Dataframes windowed for features and targets
    """
    
    if hours_ahead != None:
        window_size = floor(hours_ahead * window_multiplier)
    elif weeks_ahead != None:
        window_size = floor(weeks_ahead * window_multiplier)
    else:
        window_size =  floor(days_ahead * window_multiplier)
    segments = []
    targets  = []
    if hours_ahead != None:
        for (start, end, ahead) in create_windows(dataset, window_size, shift, time_col, hours_ahead=hours_ahead):
            segments.append(dataset.iloc[start:end,:])
            targets.append(dataset.iloc[end:ahead,:])
    elif weeks_ahead != None:
        for (start, end, ahead) in create_windows(dataset, window_size, shift, time_col, weeks_ahead=weeks_ahead):
            segments.append(dataset.iloc[start:end,:])
            targets.append(dataset.iloc[end:ahead,:])
    else:
        # if no option is selected will default to days ahead
        for (start, end, ahead) in create_windows(dataset, window_size, shift, time_col, days_ahead=days_ahead):
            segments.append(dataset.iloc[start:end,:])
            targets.append(dataset.iloc[end:ahead,:])
    return segments, targets

def update_indicies(dataframe, time_col, shift, value):
    """
    updates the indicies with the newest indicies based on the shift value passed in
    
    :param dataframe: the dataframe to get the next indicies of
    :param time_col: the column that contains the timestamp
    :param shift: the amout to shift in hours
    :param value: the value to get the next shifted value of
    
    :return: the next index for the specivied value
    """
    next_index = value + 1
    while next_index < dataframe.shape[0]:
        start_time = dataframe[time_col][value]
        end_time = dataframe[time_col][next_index]
        elapsed = end_time - start_time
        if elapsed.seconds / 3600 == shift:
            break
        next_index += 1
    return next_index
         
    
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
        
