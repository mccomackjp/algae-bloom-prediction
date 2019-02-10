import numpy as np
import pandas as pd
from keras import backend as K


def extract_percentile(windows, time_column, percentile=0.95):
    """
    Extracts the percentiles from the list of windowed DataFrames into a single DataFrame.

    :param windows: List of windowed DataFrames to be extracted.
    :param time_column: name of the datetime object column in the DataFrame
    :param percentile: float percentage of the value to extract.
        example: max = 1.0, min = 0.0, average = 0.5

    :return: datetimeIndexed DataFrame of percentiled windows.
    """
    extracted = pd.DataFrame()
    for df in windows:
        extracted = extracted.append(df.quantile(percentile, numeric_only=False))
    extracted[time_column + 'Index'] = extracted[time_column]
    return extracted.set_index(time_column + 'Index')


def segment_dataset(df, time_col,
                    x_win_size=pd.Timedelta(2, unit='d'),
                    y_win_size=pd.Timedelta(1, unit='d'),
                    shift=pd.Timedelta(1, unit='h')):
    """
    Segments the data set based on the parameters that are passed in.

    :param df: the data frame to segment into windows, must be indexed by datetime
    :param time_col: the name of the time column in the dataset
    :param x_win_size: Timedelta for the size of feature windows.
    :param y_win_size: Timedelta for the size of target windows.
    :param shift: Timedelta for the amount to shift windows by.

    :return: An array of Dataframes windowed for features and targets
    """
    segments = []
    targets = []
    start = df[time_col][0]
    end = df[time_col][len(df[time_col]) - 1]
    offset = pd.Timedelta(1, unit='s')  # to remove overlap between x and y
    while start + x_win_size + y_win_size <= end:
        segments.append(df[start:start + x_win_size])
        targets.append(df[start + x_win_size + offset: start + x_win_size + y_win_size])
        start += shift
    return segments, targets


def create_class_predictions(pred):
    """
    Obtains the guesses for the model
    """
    retval = np.array([])
    for row in pred:
        max_value = (-1, -1)
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
        tf.saved_model.simple_save(sess, tf_path, inputs={'input': model.input},
                                   outputs={t.name: t for t in model.outputs})
