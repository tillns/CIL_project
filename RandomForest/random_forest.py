"""Random Forest

This file trains a random forest classifier and evaluates it against a test set. It also predicts
the scores for a query data set and saves them to a file.

"""

from argparse import ArgumentParser
from random_forest_utils import get_train_data, get_test_data, get_query_data
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics
import joblib
import os
import csv

parser = ArgumentParser()
parser.add_argument("--data-directory", required=True, help="Required. The directory where the dataset is stored.")
parser.add_argument("--numpy-directory", required=True, help="Required. The directory where the numpy data is stored or should be stored. This directory doesn't have to exists yet.")
parser.add_argument("--num_features", default=10, type=int, help="Optional. The number of features per image taken into consideration. Default is 10")
parser.add_argument("--split_ratio", default=0.9, type=float, help="Optional. The train-test split ratio of the data to be stored. The value has to be between 0 and 1. Default is 0.9")
parser.add_argument("--random-seed", default=1234, type=int, help="Optional. A integer to seed the random number generator. Default is 1234")

def train_model(arguments):
    """Trains a random forest classifier

    Creates and trains a random forest classifier. It is then used to predict the scores on a test set and is saved to disk for later use.

    Parameters
    ----------
    arguments : argparse-arguments
        The given command line arguments

    """

    numpy_data_directory = arguments.numpy_directory
    data_directory = arguments.data_directory
    num_features = arguments.num_features
    split_ratio = arguments.split_ratio

    train_features, train_labels = get_train_data(numpy_data_directory, data_directory, num_features, split_ratio)

    base_model = sklearn.ensemble.RandomForestRegressor(criterion="mae", oob_score=True, random_state = arguments.random_seed)
    ml_model = sklearn.model_selection.GridSearchCV(base_model, {"n_estimators": [5, 10, 50, 100]}, verbose=5, scoring='neg_mean_absolute_error')

    print("Fitting...")
    ml_model.fit(train_features, train_labels)
    print("Finished fitting!")

    # save model to directory
    joblib.dump(ml_model, _model_file_path(numpy_data_directory, num_features, split_ratio))

    test_features, test_labels = get_test_data(numpy_data_directory, data_directory, num_features, split_ratio)

    print("Predicting test set...")
    predictions = ml_model.predict(test_features)
    print("Finished Predicting, mean absolute error: {}".format(sklearn.metrics.mean_absolute_error(test_labels, predictions)))

def create_query_file(arguments):
    """Uses a trained classifier to predict the query results

    Parameters
    ----------
    arguments : argparse-arguments
        The given command line arguments

    """

    numpy_data_directory = arguments.numpy_directory
    data_directory = arguments.data_directory
    num_features = arguments.num_features
    split_ratio = arguments.split_ratio

    ml_model = joblib.load(_model_file_path(numpy_data_directory, num_features, split_ratio))
    query_features, query_ids = get_query_data(numpy_data_directory, data_directory, num_features)

    print("Predicting query set...")
    query_predict = ml_model.predict(query_features)
    csv_data = zip(query_ids, query_predict)

    with open(_query_result_path(numpy_data_directory, num_features, split_ratio), 'w') as query_file:
        writer = csv.writer(query_file)
        writer.writerow(["Id", "Predicted"])
        writer.writerows(csv_data)

    print("Query results writen to {}".format(_query_result_path(numpy_data_directory, num_features, split_ratio)))

def _model_file_path(numpy_data_directory, num_features, split_ratio):
    """Determines the path for the file of the trained model

    Parameters
    ----------
    numpy_data_directory : str
        The directory where the numpy data is stored or should be stored
    num_features : int
        The number of features per image
    split_ratio : float
        The train-test split ratio of the data to be stored

    Returns
    -------
    file_path : str
        Path where the trained model can be stored or loaded

    """

    model_path = os.path.join(numpy_data_directory, "fitted_models")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    file_name = "random_forest_{}_{}.sav".format(num_features, split_ratio)
    return os.path.join(model_path, file_name)

def _query_result_path(numpy_data_directory, num_features, split_ratio):
    """Determines the path for the file of the query results

    Parameters
    ----------
    numpy_data_directory : str
        The directory where the numpy data is stored or should be stored
    num_features : int
        The number of features per image
    split_ratio : float
        The train-test split ratio of the data to be stored

    Returns
    -------
    file_path : str
        Path where the query results should be stored

    """

    query_results_path = os.path.join(numpy_data_directory, "query_results")
    if not os.path.exists(query_results_path):
        os.makedirs(query_results_path)

    file_name = "random_forest_query_results_{}_{}.csv".format(num_features, split_ratio)
    return os.path.join(query_results_path, file_name)

if __name__ == "__main__":
    arguments = parser.parse_args()

    train_model(arguments)
    create_query_file(arguments)
