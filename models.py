from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVR


def get_classifier(model_name, model_params):
    if model_name == 'LogisticRegression':
        return LogisticRegression(**model_params)

    if model_name == 'KNeighborsClassifier':
        return KNeighborsClassifier(**model_params)

    if model_name == 'RandomForestClassifier':
        return RandomForestClassifier(**model_params)

    if model_name == 'DecisionTreeClassifier':
        return DecisionTreeClassifier(**model_params)

    if model_name == 'LGBMClassifier':
        return LGBMClassifier(**model_params)

    if model_name == 'AdaBoostClassifier':
        return AdaBoostClassifier(**model_params)

    if model_name == 'MLPClassifier':
        return MLPClassifier(**model_params)


def get_regressor(model_name, model_params):
    if model_name == 'LinearRegression':
        return LinearRegression(**model_params)

    if model_name == 'KNeighborsRegressor':
        return KNeighborsRegressor(**model_params)

    if model_name == 'DecisionTreeRegressor':
        return DecisionTreeRegressor(**model_params)

    if model_name == 'SVR':
        return SVR(**model_params)

    if model_name == 'RandomForestRegressor':
        return RandomForestRegressor(**model_params)

    if model_name == 'AdaBoostRegressor':

        return AdaBoostRegressor(**model_params)

    if model_name == 'MLPRegressor':
        return MLPRegressor(**model_params)

    if model_name == 'LGBMRegressor':

        return LGBMRegressor(**model_params)