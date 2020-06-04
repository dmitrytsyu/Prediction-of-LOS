import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_squared_log_error
from scipy.stats import rankdata

from encoders import MultipleEncoder, DoubleValidationEncoderNumerical
from encoders import get_single_encoder
from models import get_classifier, get_regressor


class Model:
    def __init__(self, model_name='LGBMClassifier', model_params={}, fit_params={}, cat_validation="None",
                 encoders_names=None, cat_cols=None,
                 model_validation=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                 target='classification'):

        self.model_name = model_name
        self.target = target

        self.model_params = model_params
        self.fit_params = fit_params

        self.cat_validation = cat_validation
        self.encoders_names = encoders_names
        self.cat_cols = cat_cols
        self.model_validation = model_validation

        self.encoders_list = []
        self.models_list = []
        self.scores_list_train = []
        self.scores_list_val = []


    def fit(self, X: pd.DataFrame, y: np.array) -> tuple:

        if self.cat_validation == "None":
            encoder = MultipleEncoder(cols=self.cat_cols, encoders_names_tuple=self.encoders_names)
            X = encoder.fit_transform(X, y)

        for n_fold, (train_idx, val_idx) in enumerate(self.model_validation.split(X, y)):

            X_train, X_val = X.loc[train_idx].reset_index(drop=True), X.loc[val_idx].reset_index(drop=True)
            y_train, y_val = y[train_idx], y[val_idx]

            if self.cat_validation == "Single":
                encoder = MultipleEncoder(cols=self.cat_cols, encoders_names_tuple=self.encoders_names)
                X_train = encoder.fit_transform(X_train, y_train)
                X_val = encoder.transform(X_val)

            if self.cat_validation == "Double":
                encoder = DoubleValidationEncoderNumerical(cols=self.cat_cols, encoders_names_tuple=self.encoders_names)
                X_train = encoder.fit_transform(X_train, y_train)
                X_val = encoder.transform(X_val)
                pass

            self.encoders_list.append(encoder)


            for col in [col for col in X_train.columns if "OrdinalEncoder" in col]:
                X_train[col] = X_train[col].astype("category")
                X_val[col] = X_val[col].astype("category")



            if self.target == 'classification':
                model = get_classifier(self.model_name, self.model_params)

                if self.model_name == 'LGBMClassifier':
                    self.fit_params.update({'eval_set': [(X_train, y_train), (X_val, y_val)]} )

                model.fit(X_train, y_train,  **self.fit_params)


                self.models_list.append(model)

                y_hat = model.predict_proba(X_train)[:, 1]
                score_train = roc_auc_score(y_train, y_hat)
                self.scores_list_train.append(score_train)

                y_hat = model.predict_proba(X_val)[:, 1]
                score_val = roc_auc_score(y_val, y_hat)
                self.scores_list_val.append(score_val)

                print(f"\tAUC on {n_fold} fold train : {np.round(score_train, 4)} ")
                print(f"\tAUC on {n_fold} fold val : {np.round(score_val, 4)}")

            if self.target == 'regression':

                model = get_regressor(self.model_name, self.model_params)


                if self.model_name == 'LGBMRegressor':
                    self.fit_params.update({'eval_set': [(X_train, y_train), (X_val, y_val)]})


                model.fit(X_train, y_train, **self.fit_params)


                self.models_list.append(model)


                y_hat = model.predict(X_train)
                #score_train = mean_squared_log_error(y_train, y_hat)
                score_train = mean_squared_error(y_train, y_hat)
                self.scores_list_train.append(score_train)

                y_hat = model.predict(X_val)

                #score_val = mean_squared_log_error(y_val, y_hat)
                score_val = mean_squared_error(y_val, y_hat)

                self.scores_list_val.append(score_val)

                print(f"\tMSE on {n_fold} fold train : {np.round(score_train, 4)} ")
                print(f"\tMSE on {n_fold} fold val : {np.round(score_val, 4)} ")

        mean_score_train = np.mean(self.scores_list_train)
        mean_score_val = np.mean(self.scores_list_val)

        print(f"\tMean score train : {np.round(mean_score_train, 4)}")
        print(f"\tMean score val : {np.round(mean_score_val, 4)}")

        return mean_score_train, mean_score_val


    # prediction is not available in unranked_pred in regression
    def predict(self, X: pd.DataFrame) -> np.array:

        y_hat = np.zeros(X.shape[0])
        for encoder, model in zip(self.encoders_list, self.models_list):

            X_test = X.copy()
            X_test = encoder.transform(X_test)

            # check for OrdinalEncoder encoding
            for col in [col for col in X_test.columns if "OrdinalEncoder" in col]:
                X_test[col] = X_test[col].astype("category")

            if self.target == 'regression':

                unranked_preds = model.predict(X_test)
                y_hat = unranked_preds

            if self.target == 'classification':
                # print('*'*1000)

                unranked_preds = model.predict_proba(X_test)[:, 1]
                y_hat += rankdata(unranked_preds)

        return y_hat


if __name__ == "__main__":
    # model = LGBMClassifier(**self.model_params)
    # model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)],
    #           verbose=False, early_stopping_rounds=150)
    print("model works")
