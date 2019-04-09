from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import recall_score, precision_score, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc
import pandas as pd


def add_target_column(data_frames, target_column='BGA-Phycocyanin RFU',
                      new_target_name='bloom', threshold=2):
    """
    Adds a target categorical column to the given list of data frames.

    :param data_frames: List of data frames to add column to.
    :param target_column: Name of column to create the categorical column from.
    :param new_target_name: Name of the new target column.
    :param threshold: Threshold to set each value of the new target with.
    """
    for df in data_frames:
        df[new_target_name] = df[target_column].apply(
            lambda x: 1 if x > threshold else 0)


def import_df_data(files, drop_columns=[]):
    data_frames = []
    for file in files:
        df = pd.read_csv(file)
        for column in drop_columns:
            if column in df.columns:
                df = df.drop(columns=[column])
        data_frames.append(df)
    return data_frames


def is_numerical(series):
    """
    Checks if a series is made up of numerical data or not.

    :param series: Series object to be checked
    :return: True if the data are integers or floats, False otherwise
    """
    return series.dtype == 'int64' or series.dtype == 'float64' \
           or series.dtype == 'int32' or series.dtype == 'float32'


def split_numerical_categorical(df):
    """
    Splits a given DataFrame into numerical and categorical dataframes.

    :param df: DataFrame to split.
    :return: Tuple of: (numerical DataFrame, categorical DataFrame)
    """
    num_columns = []
    cat_columns = []
    for column in df.columns:
        if is_numerical(df[column]):
            num_columns.append(column)
        else:
            cat_columns.append(column)
    return df[num_columns], df[cat_columns]


def remove_matching_strings(original, to_remove):
    """
    Removes a list of strings from the original list while maintaining the order.

    :param original: list to remove from.
    :param to_remove: list to check if exists in original.
    :return: list of strings in "original" not in "to_remove"
    """
    result = []
    for s in original:
        if s not in to_remove:
            result.append(s)
    return result


def get_matching_strings(a, b):
    """
    Gets a list of strings that are in both lists a and b.

    :param a: List of strings to check against b.
    :param b: List of strings to check against a.
    :return: List of matching strings.
    """
    matches = []
    for string_a in a:
        for string_b in b:
            if string_a == string_b:
                matches.append(string_a)
    return matches


def scale_columns(train, test, scaler=StandardScaler()):
    """
    Scales the columns in the train and test set
    
    :param train: The Dataframe of numerical columns that will be used for training models
    :param test: The Dataframe of numerical of columns that will be used for scaling models
    :param scaler: the scalar that will be used defaults to StandardScaler()
    
    :return: ret_train, ret_test tuple of the scaled values
    """

    ret_train = scaler.fit_transform(train)
    ret_test = scaler.transform(test)
    return ret_train, ret_test


def alter_columns(train, test, op):
    """
    Alters the columns based on the numpy mathematical operation that is passed in

    :param train: the training Numpy array to have the columns altered
    :param test: the testing Numpy array to have the columns altered
    :param op: the functools.partial(Numpy mathematical) operation to do on the Data

    :return: train, test with the altered values for the NP array
    """
    return op(train), op(test)


def impute_columns(train, test, imputer):
    """
    Imputes the columns based on the DataFrame that is passed in
    
    :param train: The  Data frame that will be used for training
    :param test: The Data frame that will be used for testin
    :param imputer: The imputer that will be used.
    
    :return: ret_train, ret_test tuple of the imputed values
    """
    ret_train = imputer.fit_transform(train)
    ret_test = imputer.transform(test)

    return ret_train, ret_test


def create_numpy_arrays(training_df, testing_df, x_columns, y_column,
                        null_model=False):
    """
    Creates training and testing numpy arrays with scalled and imputed data from the given data
    frames. Will creates dummy columns for categorical data (non int or float type) data.

    :param training_df: Data frame to create the training data from.
    :param testing_df: Data frame to create the testing data from.
    :param x_columns: Columns to be used as inputs.
    :param y_column: Target column to be predicted.
    :param null_model: Flag if this should create a null model or not.
    :return: x_train, y_train, x_test, y_test tuple of numpy arrays.
    """
    # Split dataframes by categorical and numerical data
    df_train_num, df_train_cat = split_numerical_categorical(training_df)
    df_test_num, df_test_cat = split_numerical_categorical(testing_df)

    # Select only the x columns in the numerical data frame.
    x_columns_num = list(set(x_columns).intersection(df_train_num.columns))

    x_train = df_train_num[x_columns_num].astype('float64').values
    x_test = df_test_num[x_columns_num].astype('float64').values

    y_train = df_train_cat if y_column in df_train_cat.columns else df_train_num
    y_test = df_test_cat if y_column in df_test_cat.columns else df_test_num
    y_train = y_train[y_column].astype('float64').values
    y_test = y_test[y_column].astype('float64').values

    if len(x_columns_num) > 0:
        # Scale the numerical data
        scaler = StandardScaler()
        x_train, x_test = scale_columns(x_train, x_test, scaler)

        # impute missing values
        imputer = SimpleImputer(missing_values=np.nan)
        x_train, x_test = impute_columns(x_train, x_test, imputer)

    # add categorical columns
    if len(df_train_cat.columns) > 0 and len(df_test_cat.columns) > 0:
        # Select only x columns in the categorical data frame.
        x_columns_cat = list(set(x_columns).intersection(df_train_cat.columns))
        if len(x_columns_cat) > 0:
            x_train_cat = pd.get_dummies(df_train_cat[x_columns_cat])
            x_test_cat = pd.get_dummies(df_test_cat[x_columns_cat])
            # have to make sure that all columns are in both datasets
            test_col = x_test_cat.columns
            train_col = x_train_cat.columns
            for col in train_col:
                if col not in test_col:
                    x_test_cat[col] = 0
            for col in test_col:
                if col not in train_col:
                    x_train_cat[col] = 0

            x_train_cat = x_train_cat.values.astype('float64')
            x_test_cat = x_test_cat.values.astype('float64')
            x_train = np.hstack([x_train, x_train_cat])
            x_test = np.hstack([x_test, x_test_cat])

    if null_model:
        # Create null model if required
        x_train = np.zeros(x_train.shape)
        x_test = np.zeros(x_test.shape)

    return x_train, y_train, x_test, y_test


def train_model(model, training_df, testing_df, x_columns, y_column, null_model=False,
                mathop=None):
    """
    Trains a Scikit-Learn classification model on the given training and testing data frames.

    :param model: Scikit-Learn classification model
    :param training_df: Data frame to create the training data from.
    :param testing_df: Data frame to create the testing data from.
    :param x_columns: Columns to be used as inputs.
    :param y_column: Target column to be predicted.
    :param max_iter: Max iterations for training.
    :param null_model: Whether to train a null model or not.
    :param mathop: the functools.partial(Numpy mathematical) operation to do on the Data

    :return: tuple of accuracy, recall, precision, and confusing matrix metrics,

    as well as predictions made, predictions probabilities, and the model itself.
    """
    # Create training and testing numpy arrays
    x_train, y_train, x_test, y_test = create_numpy_arrays(training_df,
                                                           testing_df,
                                                           x_columns,
                                                           y_column,
                                                           null_model=null_model)

    if mathop is not None:
        x_train, x_test = alter_columns(x_train, x_test, mathop)
    # Train the model
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    predictions_prob = model.predict_proba(x_test)
    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    return accuracy, recall, precision, cm, predictions, predictions_prob, model


def roc_plot(actual, predictions):
    """
    Plots a ROC curve.
    :param actual: Array of actual target values
    :param predictions: Predictions made by the model.
    :return: matplotlib ROC plot.
    """
    fpr, tpr, thresholds = roc_curve(actual, predictions)
    roc_auc = auc(fpr, tpr)
    print("Model AUC: %0.4f" % roc_auc)
    plt.title("ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot(fpr, tpr)


def sort_columns_by_metric(model, training_df, testing_df, x_columns, y_column,
                           optimize_accuracy=True, optimize_recall=False, optimize_precision=False,
                           mathop=None):
    """
    Trains and sorts each column in the x_columns by accuracy, recall, or precision

    :param model: Scikit-Learn classification model
    :param training_df: Data frame to create the training data from.
    :param testing_df: Data frame to create the testing data from.
    :param x_columns: Columns to be used as inputs.
    :param y_column: Target column to be predicted.
    :param optimize_accuracy: True if you wish optimize by accuracy (default is True)
    :param optimize_recall: True if you wish optimize by recall (default is False)
    :param optimize_precision: True if you wish optimize by precision (default is False)
    :param mathop: the functools.partial(Numpy mathematical) operation to do on the Data

    :return: List of sorted column names
    """
    models = {}
    for column in x_columns:
        print("Training model with:", column)
        accuracy, recall, precision, cm, _, _, _ = train_model(
            model, training_df, testing_df, [column], y_column, mathop=mathop)
        models[column] = (accuracy * optimize_accuracy) + (recall * optimize_recall) + (precision * optimize_precision)
        print("Accuracy", accuracy)
        print("Recall:", recall)
        print("Precision", precision)
        print("Confusion Matrix:\n", cm)
        print()

    # sort columns by the sum of selected metrics first
    sorted_columns = sorted(models, key=models.get, reverse=True)
    for column in sorted_columns:
        print("{} metric value: {}".format(column, models[column]))
    return sorted_columns


def greedy_model(model, training_df, testing_df, x_columns, y_column, sorted_columns, base_columns=[], mathop=None):
    """
    Creates a greedy model based on columns which only improve recall.

    :param model: Scikit-Learn classification model
    :param training_df: Data frame to create the training data from.
    :param testing_df: Data frame to create the testing data from.
    :param x_columns: Columns to be used as inputs.
    :param y_column: Target column to be predicted.
    :param sorted_columns: List of sorted columns by recall.
    :param base_columns: Base columns to start the greedy model with.
    :param mathop: the functools.partial(Numpy mathematical) operation to do on the Data

    :return: tuple of recall, precision, and confusing matrix metrics,
    as well as predictions made, predictions probabilities, and the model itself.
    """
    # Start with a base null model
    if len(base_columns) > 0:
        accuracy, recall, precision, cm, predictions, predictions_prob, model = train_model(
            model, training_df, testing_df, base_columns, y_column, mathop=mathop)
    else:
        accuracy, recall, precision, cm, predictions, predictions_prob, model = train_model(
            model, training_df, testing_df, x_columns, y_column, null_model=True, mathop=mathop)
    greedy_columns = base_columns
    # Remove the base columns from the greedy columns
    print('greedy_columns:', greedy_columns)
    print('sorted_columns:', sorted_columns)
    sorted_columns = remove_matching_strings(sorted_columns, greedy_columns)
    print('adjusted sorted_columns:', sorted_columns)

    for column in sorted_columns:
        temp_columns = greedy_columns + [column]
        print("Training model with:", temp_columns)
        temp_accuracy, temp_recall, temp_precision, temp_cm, temp_pred, temp_pred_prob, \
        temp_model = train_model(model, training_df, testing_df, temp_columns, y_column, mathop=mathop)
        print("Test model accuracy:", temp_accuracy)
        print("Test model recall:", temp_recall)
        print("Test model precision:", temp_precision)
        if temp_accuracy > accuracy:
            print("\nUpdating greedy model")
            greedy_columns = temp_columns
            accuracy = temp_accuracy
            recall = temp_recall
            precision = temp_precision
            cm = temp_cm
            predictions = temp_pred
            predictions_prob = temp_pred_prob
            model = temp_model
        print()

    print("Final greedy columns:", greedy_columns)
    print("Final greedy accuracy", accuracy)
    print("Final greedy recall:", recall)
    print("Final greedy precision:", precision)
    print("Final greedy confusion matrix:\n", cm)
    return accuracy, recall, precision, cm, predictions, predictions_prob, model


def cross_validate(model, df_early, df_late, x_columns, y_column, mathop=None):
    """
    Cross validate the early and late DataFrames with each other.

    :param model: the model that is going to be used in training.
    :param df_early: The already cleaned, ready to train DataFrame of the earlier year
    :param df_late: The already cleaned, ready to train DataFrame of the later year
    :param x_columns: the columns that are going to be used to train on
    :param y_column: the target column name
    :param mathop: the math operation if needed

    :return: a dictionary of results with keys representing the trainset_testset
    """

    results = {}

    accuracy, recall, precision, cm, predictions, predictions_prob, _ = train_model(model, df_late, df_early,
                                                                                    x_columns, y_column,
                                                                                    null_model=False,
                                                                                    mathop=mathop)
    results['dflate_dfearly'] = (accuracy, recall, precision)
    accuracy, recall, precision, cm, predictions, predictions_prob, _ = train_model(model, df_early, df_late,
                                                                                    x_columns, y_column,
                                                                                    null_model=False,
                                                                                    mathop=mathop)
    results['dfearly_dflate'] = (accuracy, recall, precision)
    return results
