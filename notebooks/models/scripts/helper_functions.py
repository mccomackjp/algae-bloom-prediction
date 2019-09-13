import numpy as np
import pandas as pd
import datetime
from scripts.SurfaceStationReading import SurfaceStationReading
from keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import scripts.logistic_regression_functions as lrf
import warnings


def create_correlation_plots(dataframe, target, figsize=(10, 50)):
    """
    Creates a series of numpy correlation plots between all numerical columns and the target column.

    :param dataframe: DataFrame to plot.
    :param target: Target column to compare with.
    :param figsize: Size of each plot.
    :return: matplot lib subplots.
    """
    numerical_columns = []
    for col in dataframe.columns:
        if col != target and lrf.is_numerical(dataframe[col]):
            numerical_columns.append(col)
    f, axes = plt.subplots(nrows=len(numerical_columns), ncols=1, figsize=figsize)
    for i, col in enumerate(numerical_columns):
        a = dataframe[target]
        b = dataframe[col]
        a = (a - a.mean()) / (a.std() * len(a))
        b = (b - b.mean()) / (b.std())
        data = np.correlate(a, b, mode='full')
        data = data[-len(dataframe[target]):]  # Grab just the last half of the curve
        temp_column = '{} : {} Correlation'.format(target, col)
        day_ratio = 15 / 60 / 24
        days = [i * day_ratio for i in range(0, len(dataframe[target]))]
        temp = pd.DataFrame({temp_column: data,
                             "Elapsed Days": days})
        temp.plot(ax=axes[i], title=temp_column, x="Elapsed Days", y=temp_column)
    plt.tight_layout()


def show_graphs(dataframe, target):
    """
    Plots graphs for each variable vs the target in the dataframe.
    :param dataframe: The dataframe that will need to need to get the coorilation graphs for
    :param target: The entire dataframe column of the target variable that will need to find the coorliation for
    """
    f, axes = plt.subplots(nrows=len(dataframe.columns), ncols=1, figsize=(10, 50))
    plt.subplots_adjust(hspace=1)
    for i, col in enumerate(dataframe.columns):
        if lrf.is_numerical(dataframe[col]):
            dataframe[col].plot(ax=axes[i], title=col, color='b')
            axe = axes[i].twinx()
            dataframe[target].plot(ax=axe, color='r')
            lines, labels = axes[i].get_legend_handles_labels()
            lines2, labels2 = axe.get_legend_handles_labels()
            axe.legend(lines + lines2, labels + labels2, loc="best")
        else:
            sns.boxplot(data=dataframe, x=col, y=target, ax=axes[i])


def bin_series(series, bins, quantile_binning=False):
    """
    Bins a series into categories.

    :param series: Series to be binned.
    :param bins: Number of bins to create.
    :param quantile_binning: Whether to use quantile based binning.
    :return: Binned Series object
    """
    if quantile_binning:
        labels = [str(x) for x in set(pd.qcut(series, q=bins))]
        return pd.qcut(series, q=bins, labels=labels)
    else:
        labels = [str(x) for x in set(pd.cut(series, bins=bins))]
        return pd.cut(series, bins=bins, labels=labels)


def bin_df(df, bins, quantile_binning=False):
    """
    Bins each column in the given DataFrame
    :param df: DataFrame to bin
    :param bins: Number of bins to create.
    :param quantile_binning: Whether to use quantile based binning.
    :return: DataFrame with additional binned category columns.
    """
    new_df = pd.DataFrame(df[[]])
    new_columns = []
    for col in df.columns:
        new_col = '{}_{}_q_bins' if quantile_binning else '{}_{}_bins'
        new_col = new_col.format(col, bins)
        new_columns.append(new_col)
        new_df[new_col.format(col)] = bin_series(df[col], bins, quantile_binning)
    return new_df, new_columns


def data_window_reduction(df, time_column, target_column,
                          x_win_size=pd.Timedelta('28 days'),
                          y_win_size=pd.Timedelta(1, unit='d'),
                          shift=pd.Timedelta(14, unit='h'),
                          percentile=0.95):
    """
    Reduces data based on a sliding window method.

    :param df: DataFrame to reduce
    :param time_column: name of the datetime object column in the DataFrame
    :param target_column: Column which is the target for predictions.
    :param x_win_size: Timedelta for the size of feature windows.
    :param y_win_size: Timedelta for the size of target windows.
    :param shift: Timedelta for the amount to shift windows by.
    :param percentile: float percentage of the value to extract.
        example: max = 1.0, min = 0.0, average = 0.5
    :return: Reduced DataFrame.
    """
    warnings.warn("data_window_reduction is depricated, use windowize instead.",
                  DeprecationWarning)
    print("Segmenting...")
    x_windows, y_windows = segment_dataset(df, time_column, x_win_size=x_win_size, y_win_size=y_win_size, shift=shift)
    print("Extracting feature windows...")
    x_windows = extract_percentile(x_windows, time_column, percentile=percentile)
    print("Extracting target windows...")
    y_windows = extract_percentile(y_windows, time_column, percentile=percentile)
    print("Combining extractions...")
    x_windows[target_column] = y_windows[target_column].values
    return x_windows


def extract_percentile(windows, time_column,
                       target_column='BGA-Phycocyanin RFU',
                       feature_percentiles=[0.95],
                       target_percentile=0.95):
    """
    Extracts the percentiles from the list of windowed DataFrames into a single DataFrame.

    :param windows: List of windowed DataFrames to be extracted.
    :param time_column: name of the datetime object column in the DataFrame
    :param percentile: float or array-like percentage(s) of the value(s) to extract.
        example: max = 1.0, min = 0.0, average = 0.5

    :return: datetimeIndexed DataFrame of percentiled windows.
    """
    extracted = pd.DataFrame()
    if not isinstance(feature_percentiles, list):
        feature_percentiles = [feature_percentiles]
    feature_percentiles = sorted(feature_percentiles)

    for df in windows:
        dat_ser = pd.Series()
        dat_frm = pd.DataFrame()
        for percent in feature_percentiles:
            # Get the series for the percent
            # add the column name with the percent appended to it if we have multiple percentiles.
            suffix = '_' + str(percent) if len(feature_percentiles) > 1 else ''
            dat_ser = dat_ser.append(df.drop(columns=target_column).quantile(percent, numeric_only=False).add_suffix(suffix))
        dat_ser = dat_ser.append(df[[target_column]].quantile(target_percentile))
        dat_frm = dat_frm.append(dat_ser, ignore_index=True)
        # Take the last time in the df to ensure we get the last one.
        dat_frm[time_column] = df[time_column][-1]
        extracted = extracted.append(dat_frm)
    # else:
    #     extracted = extracted.append(df.quantile(percentile, numeric_only=False))
    extracted[time_column + 'Index'] = extracted[time_column]

    return extracted.set_index(time_column + 'Index')


def windowize(df, time_column,
              target_column='BGA-Phycocyanin RFU',
              x_win_size=pd.Timedelta('28 days'),
              y_win_size=pd.Timedelta(1, unit='d'),
              shift=pd.Timedelta(14, unit='h'),
              feature_percentiles=[0.95],
              target_percentile=0.95,
              separation=pd.Timedelta(0),
              custom_parameters=None):
    """
    Reduces data based on a sliding window method.

    :param custom_parameters:
    :param df: DataFrame to reduce
    :param time_column: name of the datetime object column in the DataFrame
    :param target_column: Column which is the target for predictions.
    :param x_win_size: Timedelta for the size of feature windows.
    :param y_win_size: Timedelta for the size of target windows.
    :param shift: Timedelta for the amount to shift windows by.
    :param percentile: float percentage of the value to extract.
        example: max = 1.0, min = 0.0, average = 0.5
    :param separation: Timedelta for the amount to separate the x window and y window by
    :param custom_parameters: Dictionary of dictionaries containing custom x_window_size
        and separation amounts for specific columns (all parameters are optional).
        'x_win_size' and 'separation' are the only 2 custom parameter keys in use.
        where the key is the column name, and the value is a dictionary of Timedeltas,
        with key 'x_win_size' being the x window size,
        and key 'separation' being the separation from the y window amount.
        generic example:
            custom_parameters = {'column name': {'x_win_size':pd.Timedelta(), 'separation':pd.Timedelta()}}

        sepecific use example:
            custom_paramaters =
            {'Temp C': {'x_win_size':pd.Timedelta('7 days'), 'separation':pd.Timedelta('21 days')},
              'pH': {'x_win_size':pd.Timedelta('4 days'), 'separation':pd.Timedelta(0)}}

    :return: Reduced DataFrame.
    """
    print("Segmenting...")
    x_windows, y_windows = extract_windows(df, time_column, x_win_size=x_win_size,
                                           y_win_size=y_win_size, shift=shift,
                                           separation=separation,
                                           custom_parameters=custom_parameters)
    print("Extracting feature and target windows...")
    x_windows = extract_percentile(x_windows, time_column, target_column,
                                   feature_percentiles=feature_percentiles,
                                   target_percentile=target_percentile)
    print("Extracting target windows...")
    y_windows = extract_percentile(y_windows, time_column, target_column,
                                   feature_percentiles=feature_percentiles,
                                   target_percentile=target_percentile)
    print("Combining extractions...")
    x_windows[target_column] = y_windows[target_column].values
    return x_windows


def extract_windows(df, time_col,
                    x_win_size=pd.Timedelta('28 days'),
                    y_win_size=pd.Timedelta(1, unit='d'),
                    shift=pd.Timedelta(14, unit='h'),
                    separation=pd.Timedelta(0),
                    custom_parameters=None):
    """
    Extracts the data set into feature and target windows.

    :param df: the data frame to segment into windows, must be indexed by datetime
    :param time_col: the name of the time column in the dataset
    :param x_win_size: Timedelta for the size of feature windows.
    :param y_win_size: Timedelta for the size of target windows.
    :param shift: Timedelta for the amount to shift windows by.
    :param separation: Timedelta for the amount to separate the x window and y window by
    :param custom_parameters: Dictionary of dictionaries containing custom shift
        and separation amounts for specific columns (all parameters are optional).
        where the key is the column name, and the value is a dictionary of Timedeltas,
        with key 'x_win_size' being the x window size,
        and key 'separation' being the separation from the y window amount.
        generic example:
            custom_parameters = {'column name': {'x_win_size':pd.Timedelta(), 'separation':pd.Timedelta()}}

        sepecific use example:
            custom_paramaters =
            {'Temp C': {'x_win_size':pd.Timedelta('7 days'), 'separation':pd.Timedelta('21 days')},
              'pH': {'x_win_size':pd.Timedelta('4 days'), 'separation':pd.Timedelta(0)}}

    :return: An array of Dataframes windowed for features and targets
    """
    if custom_parameters is None:
        custom_parameters = dict()
    features = []
    targets = []
    start = df[time_col][0]
    end = df[time_col][len(df[time_col]) - 1]
    offset = pd.Timedelta(1, unit='s')  # removes overlap between x and y since indexing is inclusive
    max_x_win = x_win_size + separation
    # Check if the custom window and separation values are larger than the default.
    for key, value in custom_parameters.items():
        temp_x_win = value['x_win_size'] if 'x_win_size' in value else x_win_size
        temp_sep = value['separation'] if 'separation' in value else separation
        max_x_win = max(max_x_win, temp_x_win + temp_sep)
    while start + max_x_win + y_win_size <= end:
        # We want to anchor off of the start of the y window
        y_start = start + max_x_win
        targets.append(df[y_start + offset:y_start + y_win_size])
        feature_window = extract_feature_window(df, x_win_size, separation, y_start, custom_parameters)
        features.append(feature_window)
        start += shift
    return features, targets


def extract_feature_window(df, x_win_size, separation, y_start, custom_parameters):
    """
    Extracts a feature window from the given DataFrame.

    :param df: DataFrame to extract from
    :param x_win_size: Timedelta for the size of feature windows.
    :param separation: Timedelta for the amount to separate the x window and y window by
    :param y_start: DateTime object of target window start point.
    :param custom_parameters: Dictionary of dictionaries containing custom shift
        and separation amounts for specific columns (all parameters are optional).
        where the key is the column name, and the value is a dictionary of Timedeltas,
        with key 'x_win_size' being the x window size,
        and key 'separation' being the separation from the y window amount.
        generic example:
            custom_parameters = {'column name': {'x_win_size':pd.Timedelta(), 'separation':pd.Timedelta()}}

        sepecific use example:
            custom_paramaters =
            {'Temp C': {'x_win_size':pd.Timedelta('7 days'), 'separation':pd.Timedelta('21 days')},
              'pH': {'x_win_size':pd.Timedelta('4 days'), 'separation':pd.Timedelta(0)}}

    :return: Extracted feature window as a DataFrame.
    """
    feature_window = pd.DataFrame()
    for col in df.columns:
        temp_x_window = x_win_size
        temp_sep = separation
        if col in custom_parameters:
            if 'x_win_size' in custom_parameters[col]:
                temp_x_window = custom_parameters[col]['x_win_size']
            if 'separation' in custom_parameters[col]:
                temp_sep = custom_parameters[col]['separation']
        # Anchor x_start from the y_start
        x_start = y_start - temp_x_window - temp_sep
        new_df = df[[col]][x_start:x_start + temp_x_window]
        feature_window = pd.concat([feature_window, new_df], axis='columns')
    return feature_window


def segment_dataset(df, time_col,
                    x_win_size=pd.Timedelta('28 days'),
                    y_win_size=pd.Timedelta(1, unit='d'),
                    shift=pd.Timedelta(14, unit='h')):
    """
    Segments the data set based on the parameters that are passed in.

    :param df: the data frame to segment into windows, must be indexed by datetime
    :param time_col: the name of the time column in the dataset
    :param x_win_size: Timedelta for the size of feature windows.
    :param y_win_size: Timedelta for the size of target windows.
    :param shift: Timedelta for the amount to shift windows by.

    :return: An array of Dataframes windowed for features and targets
    """
    warnings.warn("segment_dataset is depricated, use extract_windows instead.",
                  DeprecationWarning)
    segments = []
    targets = []
    start = df[time_col][0]
    end = df[time_col][len(df[time_col]) - 1]
    offset = pd.Timedelta(1, unit='s')  # removes overlap between x and y since indexing is inclusive
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


def create_time_of_day(x):
    """
    creates a time Category that will correlate to the Datetime object that is passed in
    
    :param x: the Datetime object to create the time of day for
    
    :return: the string representing the time of day.
    """
    retval = ''
    if x.hour >= 22 or x.hour <= 4:
        retval = 'night'
    elif x.hour <= 6:
        retval = 'dawn'
    elif x.hour <= 10:
        retval = 'morning'
    elif x.hour <= 14:
        retval = 'afternoon'
    elif x.hour <= 18:
        retval = 'dusk'
    else:
        retval = 'evening'
    return retval


def round_time(dt, round_to=900):
    """
    Round a date time object to any time lapse in seconds
    :param dt: the datetime.datetime object to be rounded
    :param round_to: The closes number of seconds to round to, default 15 minutes
    :return: The rounded datetime object to the roundTo time
    """
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds + round_to / 2) // round_to * round_to
    return dt + datetime.timedelta(0, rounding - seconds, -dt.microsecond)


def extract_weather_data(filename):
    """
    extracts the weather data from the file name
    :param filename: the file name that contains the ISD weather station data
    :return: an array of entries from the file
    """
    with open(filename) as file:
        contents = file.readlines()
    contents = [SurfaceStationReading(x) for x in contents]
    return contents


def _apply_dictionary(value, dict):
    """

    Helper method to be used when bucketing columns. This function applies the dictionary (dict) to the specific
    'value' value in the. The dictionary must have all basis covered or check in order to be completed.
    :param value: The value to have to categorize
    :param dict: The dictionary to be applied.
    :return: the category of the value
    """

    for key in dict.keys():
        ret_val = dict[key](value)
        if ret_val is not None:
            return ret_val


def bucket_column(df, column, dictionary, name=''):
    """
    Buckets the specific column in the DataFrame

    :param df: the DataFrame to bucket
    :param column: the column of the DataFrame to bucket
    :param dictionary: the dictionary with the conditionals within. all conditions met to guarantee the working as
                        intended
    :param name: the name to be appended to the column.

    :return: the altered DataFrame with a new column of 'column' (passed in) + '_bucket' if the name of the column was
            was not specified; Otherwise just the name that was passed in.
    """

    if name != '':
        df[name] = df[column].apply(_apply_dictionary, args=(dictionary,)).astype('category')
    else:
        df[column + '_bucket'] = df[column].apply(_apply_dictionary, args=(dictionary,)).astype('category')
