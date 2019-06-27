"""Random Forest

This file trains a random forest classifier and evaluates it against a test set. It also predicts
the scores for a query data set and saves them to a file.

"""

from argparse import ArgumentParser
from random_forest_utils import get_train_data, get_test_data, get_query_data, get_train_and_test_data
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics
import joblib
import os
import csv
import yaml
from datetime import datetime
from shutil import copyfile

parser = ArgumentParser()
parser.add_argument("--data-directory", required=True, help="Required. The directory where the dataset is stored.")
parser.add_argument("--numpy-directory", required=True, help="Required. The directory where the numpy data is stored or should be stored. This directory doesn't have to exists yet.")
#parser.add_argument("--num-features", default=0, type=int, help="Optional. The number of features per image taken into consideration. If no number is given, the correct number is concluded from the configuration file")
#parser.add_argument("--split-ratio", default=0.9, type=float, help="Optional. The train-test split ratio of the data to be stored. The value has to be between 0 and 1. Default is 0.9")
parser.add_argument("--random-seed", default=1234, type=int, help="Optional. A integer to seed the random number generator. Default is 1234")

def _find_num_features(arguments):
    arguments.num_features = 0
    for roi_type, type_conf in conf['ROI_options'].items():
        if type_conf['include']:
            plus = type_conf['num_bins']
            if isinstance(plus, list):
                plus = sum(plus)
            if roi_type == "quarter_img":
                plus *= 4
            arguments.num_features += plus

def train_model(arguments, dump_directory):
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

    train_features, train_labels, test_features, test_labels = \
        get_train_and_test_data(numpy_data_directory, data_directory, num_features, split_ratio,
                                arguments.num_imgs_to_load)


    base_model = sklearn.ensemble.RandomForestRegressor(criterion="mae", oob_score=True, random_state = arguments.random_seed)
    ml_model = sklearn.model_selection.GridSearchCV(base_model, {"n_estimators": [5, 10, 50, 100]}, verbose=5, scoring='neg_mean_absolute_error')

    print("Fitting...")
    ml_model.fit(train_features, train_labels)
    print("Finished fitting!")

    # save model to directory
    joblib.dump(ml_model, _model_file_path(dump_directory, num_features, split_ratio))


    print("Predicting test set...")
    predictions = ml_model.predict(test_features)
    mae = sklearn.metrics.mean_absolute_error(test_labels, predictions)
    print("Finished Predicting, mean absolute error: {}".format(mae))
    file = open("{}.txt".format(os.path.join(dump_directory, "loss")), "w")
    file.write("MAE: {}".format(mae))
    file.close()

def create_query_file(arguments, dump_directory):
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

    ml_model = joblib.load(_model_file_path(dump_directory, num_features, split_ratio))
    query_features, query_ids = get_query_data(numpy_data_directory, data_directory, num_features, split_ratio)

    print("Predicting query set...")
    query_predict = ml_model.predict(query_features)
    csv_data = zip(query_ids, query_predict)

    with open(_query_result_path(dump_directory, num_features, split_ratio), 'w') as query_file:
        writer = csv.writer(query_file)
        writer.writerow(["Id", "Predicted"])
        writer.writerows(csv_data)

    print("Query results writen to {}".format(_query_result_path(dump_directory, num_features, split_ratio)))

def _model_file_path(dump_directory, num_features, split_ratio):
    """Determines the path for the file of the trained model

    Parameters
    ----------
    dump_directory : str
        The directory where the trained model should be stored
    num_features : int
        The number of features per image
    split_ratio : float
        The train-test split ratio of the data to be stored

    Returns
    -------
    file_path : str
        Path where the trained model can be stored or loaded

    """

    file_name = "random_forest_{}_{}.sav".format(num_features, split_ratio)
    return os.path.join(dump_directory, file_name)

def _query_result_path(dump_directory, num_features, split_ratio):
    """Determines the path for the file of the query results

    Parameters
    ----------
    dump_directory : str
        The directory where query results should be stored
    num_features : int
        The number of features per image
    split_ratio : float
        The train-test split ratio of the data to be stored

    Returns
    -------
    file_path : str
        Path where the query results should be stored

    """

    file_name = "random_forest_query_results_{}_{}.csv".format(num_features, split_ratio)
    return os.path.join(dump_directory, file_name)

if __name__ == "__main__":
    arguments = parser.parse_args()

    dump_directory = os.path.join(arguments.numpy_directory, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(dump_directory)
    copyfile("config.yaml", os.path.join(dump_directory, "config.yaml"))
    with open("config.yaml", 'r') as stream:
        conf = yaml.full_load(stream)
    _find_num_features(arguments)
    arguments.split_ratio = conf['split_ratio']
    arguments.num_imgs_to_load = conf['num_imgs_to_load']
    train_model(arguments, dump_directory)
    create_query_file(arguments, dump_directory)
