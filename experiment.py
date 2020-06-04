import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from model import Model
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import StratifiedKFold
#import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')


def execute_experiment(model_name, model_params, fit_params,
                       dataset_name, encoders_list, validation_type,
                       experiment_description, target='classification', N_SPLITS=5):
    dataset_pth = f"./prep_data/{dataset_name}.csv"
    results = {'model_name': model_name, "experiment_description": experiment_description}

    model_validation = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    # encoder_validation = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=2019)

    # load processed dataset
    data = pd.read_csv(dataset_pth)
    data = data.dropna()

    # make train-test split
    cat_cols = [col for col in data.columns if col.startswith("cat")]

    X_train, X_test, y_train, y_test = train_test_split(data.drop("target", axis=1), data["target"], \
                                                        test_size=0.3, shuffle=False)
    X_train, X_test = X_train.reset_index(drop=False), X_test.reset_index(drop=False)
    y_train, y_test = np.array(y_train), np.array(y_test)

    print(f'Current target: {target}')
    print(f'Current validation type : {validation_type}')

    for encoders_tuple in encoders_list:
        try:
            print(f"Current itteration : {encoders_tuple}, {dataset_name}\n")
            time_start = time.time()
            model = Model(model_name=model_name,
                              model_params=model_params,
                              fit_params=fit_params,
                              cat_validation=validation_type,
                              encoders_names=encoders_tuple,
                              cat_cols=cat_cols,
                              model_validation=model_validation,
                              target=target)

            train_score, val_score = model.fit(X_train, y_train)

            y_hat = model.predict(X_test)

            if target == 'classification':
                test_score = roc_auc_score(y_test, y_hat)

            elif target == 'regression':
                #test_score = mean_squared_log_error(y_test, y_hat)
                test_score = mean_squared_error(y_test, y_hat)

            print(f'\tTest score : {np.round(test_score, 6)}')
            print('\n', '*' * 50)

            time_end = time.time()

            results[str(encoders_tuple)] = {"train_score": train_score,
                                            "val_score": val_score,
                                            "test_score": test_score,
                                            "time": time_end - time_start}
        except Exception as err:
            print(err)
            continue
    return results




def run_experiment(dataset_list, target, target_model_params, validation_type, encoders_list, save=False):
    result_list = []
    for dataset_name in dataset_list[target]:
        for model_name, params in target_model_params[target].items():
            model_params = params['model_param']
            fit_params = params['fit_param']
            print(f'\n')
            print(model_name)

            experiment_description = f"Check single encoder, {validation_type} validation"
            results = execute_experiment(model_name, model_params, fit_params, \
                                         dataset_name, encoders_list, validation_type, experiment_description, \
                                         target)
            result_list.append(results)

    if save:
        with open(f'./results/{dataset_name}_{validation_type}.json', 'w') as file:
            json.dump(result_list, file)

    return result_list

if __name__ == '__main__':
    encoders_list = [
        ("HelmertEncoder",),  # non double
        ("SumEncoder",),  # non double
        ("LeaveOneOutEncoder",),
        ("FrequencyEncoder",),
        ("MEstimateEncoder",),
        ("TargetEncoder",),
        ("BackwardDifferenceEncoder",),  # non double
        ("JamesSteinEncoder",),
        ("OrdinalEncoder",),
        ("OneHotEncoder", ),
        ("CatBoostEncoder",),
    ]

    encoders_list_double = [
        ("LeaveOneOutEncoder",),
        ("FrequencyEncoder",),
        ("MEstimateEncoder",),
        ("TargetEncoder",),
        ("JamesSteinEncoder",),
        ("CatBoostEncoder",),

    ]

    target_model_params = {
        'classification':
            {
                'LogisticRegression': {'model_param':
                                           {"solver": "lbfgs", 'max_iter': 2000, "random_state": 42},
                                       'fit_param': {}},
                'KNeighborsClassifier': {'model_param': {"n_neighbors": 45},
                                         'fit_param': {}},
                'RandomForestClassifier': {'model_param': {"n_estimators": 350, "random_state": 42},
                                           'fit_param': {}},
                'DecisionTreeClassifier': {'model_param': {"random_state": 42},
                                           'fit_param': {}},
                'LGBMClassifier': {'model_param': {"metrics": "AUC", "n_estimators": 5000, "learning_rate": 0.02, \
                                                   "random_state": 42},
                                   'fit_param': {'verbose': False, 'early_stopping_rounds': 150}},
                'AdaBoostClassifier': {'model_param': {"n_estimators": 500, "learning_rate": 0.02, "random_state": 42},
                                       'fit_param': {}},
                'MLPClassifier': {'model_param': {"hidden_layer_sizes": (32, 64, 128, 256), "solver": 'adam', \
                                                  "learning_rate_init": 0.02, "random_state": 42,
                                                  'early_stopping': True},
                                  'fit_param': {}}
            },
        'regression':
            {
                'LinearRegression': {'model_param': {'n_jobs': -1},
                                        'fit_param': {}},
                'KNeighborsRegressor': {'model_param': {"n_neighbors": 45, 'n_jobs': -1},
                                        'fit_param': {}},
                'DecisionTreeRegressor': {'model_param': {"random_state": 42, 'max_features':'auto'},
                                          'fit_param': {}},
                'RandomForestRegressor': {'model_param': {"n_estimators": 100, "random_state": 42, 'n_jobs': -1},
                                          'fit_param': {}},
                'MLPRegressor': {'model_param': {"hidden_layer_sizes": (32, 32, 64), "solver": 'adam', \
                                                 "learning_rate_init": 0.02, "random_state": 42,
                                                 'early_stopping': True},
                                 'fit_param': {}},
                'LGBMRegressor': {'model_param': {"metrics": "MSE", "n_estimators": 5000, \
                                                  "learning_rate": 0.02, "random_state": 42,  'n_jobs':- 1,},
                                  'fit_param': {'verbose': False, 'early_stopping_rounds': 1000}},
            }
    }

    dataset_list = {'regression': ['mimic']}

    target = 'regression'
    save_results = True
    validation_type = "None"
    results_none = run_experiment(dataset_list, target, target_model_params, validation_type, encoders_list, save_results)

    validation_type = "Single"
    results_none = run_experiment(dataset_list, target, target_model_params, validation_type, encoders_list, save_results)

    validation_type = "Double"
    results_none = run_experiment(dataset_list, target, target_model_params, validation_type, encoders_list_double, save_results)
