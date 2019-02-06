from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
import pandas as pd


def add_target_column(data_frames, target_column='BGA-Phycocyanin RFU',
                      new_target_name='bloom', threshold=2):
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
    """Checks if a series is made up of numerical data or not.

    Args:
        series: Series object to be checked

    Returns:
        True if the data are integers or floats, False otherwise

    """
    return series.dtype == 'int64' or series.dtype == 'float64' \
           or series.dtype == 'int32' or series.dtype == 'float32'


def split_numerical_categorical(df):
    """Splits a given DataFrame into numerical and categorical dataframes.

    Args:
        df: DataFrame to split.

    Returns:
        Tuple of: (numerical DataFrame, categorical DataFrame)
    """
    num_columns = []
    cat_columns = []
    for column in df.columns:
        if is_numerical(df[column]):
            num_columns.append(column)
        else:
            cat_columns.append(column)
    return df[num_columns], df[cat_columns]


def create_numpy_arrays(training_df, testing_df, x_columns, y_column,
                        null_model=False):
    # Split dataframes by categorical and numerical data
    df_train_num, df_train_cat = split_numerical_categorical(training_df)
    df_test_num, df_test_cat = split_numerical_categorical(testing_df)

    x_train = df_train_num[x_columns].astype('float64').values
    y_train = df_train_num[y_column].astype('float64').values

    x_test = df_test_num[x_columns].astype('float64').values
    y_test = df_test_num[y_column].astype('float64').values

    # Scale the numerical data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # add categorical columns
    if len(df_train_cat.columns) > 0 and len(df_test_cat.columns) > 0:
        x_train_cat = pd.get_dummies(df_train_cat).astype('float64').values
        x_test_cat = pd.get_dummies(df_test_cat).astype('float64').values
        x_train = np.hstack([x_train, x_train_cat])
        x_test = np.hstack([x_test, x_test_cat])

    if null_model:
        # Create null model if required
        x_train = np.zeroes(x_train.shape)
        x_test = np.zeroes(x_test.shape)
    else:
        # impute missing values
        imputer = SimpleImputer(missing_values=np.nan)
        x_train = imputer.fit_transform(x_train)
        x_test = imputer.transform(x_test)

    return x_train, y_train, x_test, y_test


def train_model(training_df, testing_df, x_columns, y_column, max_iter=25000,
                null_model=False):
    """Trains a SGD Classifier model.

    training_df: dataframe of training data.
    testing_df: dataframe of testing data.
    x_columns: list of numerical feature column names.
    y_column: target string name.

    returns a tuple of:
    The trained model, predicted values, predicted probabilities, model accuracy
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
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    return recall, precision, cm, predictions_prob, model


def roc_plot(actual, predictions):
    fpr, tpr, thresholds = roc_curve(actual, predictions)
    roc_auc = auc(fpr, tpr)
    plt.title("ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot(fpr, tpr)


def sort_columns_by_recall(training_df, testing_df, x_columns, y_column):
    """Sorts a list of columns by their recall in descending order.

    Args:
        df: DataFrame to train with.
        regressor: model to train.

    Returns:
        a sorted list of column names

    """
    models = {}
    for column in x_columns:
        print("Training model with:", column)
        recall, precision, cm, _, _ = train_model(training_df, testing_df, [column], y_column)
        models[column] = recall
        print("Recall:", recall)
        print("Precision", precision)
        print("Confusion Matrix:\n", cm)
        print()

    # sort columns by best accuracy first
    sorted_columns = sorted(models, key=models.get, reverse=True)
    for column in sorted_columns:
        print("{} recall: {}".format(column, models[column]))
    return sorted_columns


def greedy_model(training_df, testing_df, x_columns, y_column, sorted_columns):
    """Trains a given regressor model with the given dataframe on the target
    columns using a greedy algorithm based on the sorted columns.

    Args:
        df: DataFrame to get data from.
        target_columns: column names to train on.
        sorted_columns: columns sorted by rmse to create greedy model with.
        regressor: scikit-learn Regressor object to train.
        multitask: Boolean determining if we are using a multitask Regressor.

    Returns:
       Tuple of: trained Regressor model, root mean squared error of the model.

    """
    recall, precision, cm, predictions_prob, model
    greedy_columns = []
    greedy_accuracy = -1
    greedy_predictions = None
    greedy_predictions_prob = None
    greedy_model = None
    for column in sorted_columns:
        temp_columns = greedy_columns + [column]
        print("Training model with:", temp_columns)
        temp_model, temp_predictions, temp_cm, temp_pred_prob = train_model(
            df_train,
            df_test,
            temp_columns,
            y_column)
        print("Test model accuracy:", temp_accuracy)
        if temp_accuracy > greedy_accuracy:
            print("\nUpdating greedy model")
            greedy_columns = temp_columns
            greedy_accuracy = temp_accuracy
            greedy_predictions = temp_predictions
            greedy_predictions_prob = temp_prob
            model = temp_model
        print()

    print("Final greedy columns:", greedy_columns)
    print("Final greedy accuracy:", greedy_accuracy)
    return (greedy_model, greedy_rmse)