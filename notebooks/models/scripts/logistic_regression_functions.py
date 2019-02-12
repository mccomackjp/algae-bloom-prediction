from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
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

    x_train = df_train_num[x_columns].astype('float64').values
    x_test = df_test_num[x_columns].astype('float64').values

    y_train = df_train_cat if y_column in df_train_cat.columns else df_train_num
    y_test = df_test_cat if y_column in df_test_cat.columns else df_test_num
    y_train = y_train[y_column].astype('float64').values
    y_test = y_test[y_column].astype('float64').values

    # Scale the numerical data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # impute missing values
    imputer = SimpleImputer(missing_values=np.nan)
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)

    # add categorical columns
    if len(df_train_cat.columns) > 0 and len(df_test_cat.columns) > 0:
        cat_columns = get_matching_strings(x_columns, df_train_cat.columns)
        if len(cat_columns) > 0:
            x_train_cat = pd.get_dummies(df_train_cat[cat_columns]).astype('float64').values
            x_test_cat = pd.get_dummies(df_test_cat[cat_columns]).astype('float64').values
            x_train = np.hstack([x_train, x_train_cat])
            x_test = np.hstack([x_test, x_test_cat])

    if null_model:
        # Create null model if required
        x_train = np.zeros(x_train.shape)
        x_test = np.zeros(x_test.shape)

    return x_train, y_train, x_test, y_test


def train_model(training_df, testing_df, x_columns, y_column, max_iter=25000,
                null_model=False):
    """
    Trains a linear regression model on the given training and testing data frames.

    :param training_df: Data frame to create the training data from.
    :param testing_df: Data frame to create the testing data from.
    :param x_columns: Columns to be used as inputs.
    :param y_column: Target column to be predicted.
    :param max_iter: Max iterations for training.
    :param null_model: Whether to train a null model or not.
    :return: tuple of accuracy, recall, precision, and confusing matrix metrics,
    as well as predictions made, predictions probabilities, and the model itself.
    """
    # Create training and testing numpy arrays
    x_train, y_train, x_test, y_test = create_numpy_arrays(training_df,
                                                           testing_df,
                                                           x_columns,
                                                           y_column,
                                                           null_model)

    # Train the model
    model = SGDClassifier(max_iter=max_iter, loss="log")
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
    :return: mMtplotlib ROC plot.
    """
    fpr, tpr, thresholds = roc_curve(actual, predictions)
    roc_auc = auc(fpr, tpr)
    print("Model AUC: %0.4f" % roc_auc)
    plt.title("ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot(fpr, tpr)


def sort_columns_by_accuracy(training_df, testing_df, x_columns, y_column):
    """
    Trains and sorts each column in the x_columns by accuracy
    :param training_df: Data frame to create the training data from.
    :param testing_df: Data frame to create the testing data from.
    :param x_columns: Columns to be used as inputs.
    :param y_column: Target column to be predicted.
    :return: List of sorted column names
    """
    models = {}
    for column in x_columns:
        print("Training model with:", column)
        accuracy, recall, precision, cm, _, _, _ = train_model(training_df, testing_df, [column], y_column)
        models[column] = accuracy
        print("Accuracy", accuracy)
        print("Recall:", recall)
        print("Precision", precision)
        print("Confusion Matrix:\n", cm)
        print()

    # sort columns by best recall first
    sorted_columns = sorted(models, key=models.get, reverse=True)
    for column in sorted_columns:
        print("{} accuracy: {}".format(column, models[column]))
    return sorted_columns


def greedy_model(training_df, testing_df, x_columns, y_column, sorted_columns):
    """
    Creates a greedy model based on columns which only improve recall.

    :param training_df: Data frame to create the training data from.
    :param testing_df: Data frame to create the testing data from.
    :param x_columns: Columns to be used as inputs.
    :param y_column: Target column to be predicted.
    :param sorted_columns: List of sorted columns by recall.
    :return: tuple of recall, precision, and confusing matrix metrics,
    as well as predictions made, predictions probabilities, and the model itself.
    """
    # Start with a base null model
    accuracy, recall, precision, cm, predictions, predictions_prob, model = train_model(
        training_df, testing_df, x_columns, y_column, null_model=True)
    greedy_columns = []
    for column in sorted_columns:
        temp_columns = greedy_columns + [column]
        print("Training model with:", temp_columns)
        temp_accuracy, temp_recall, temp_precision, temp_cm, temp_pred, temp_pred_prob, \
            temp_model = train_model(training_df, testing_df, temp_columns, y_column)
        print("Test model recall:", temp_recall)
        print("Test model precision:", temp_precision)
        if temp_recall > recall:
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
    print("Final greedy precision:", recall)
    print("Final greedy confusion matrix:\n", cm)
    return accuracy, recall, precision, cm, predictions, predictions_prob, model
