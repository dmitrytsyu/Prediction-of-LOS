import numpy as np
import pandas as pd
from typing import List




def save_dict_to_file(dic: dict, path: str, save_raw=False) -> None:
    """
    Save dict values into txt file
    :param dic: Dict with values
    :param path: Path to .txt file
    :return: None
    """

    f = open(path, 'w')
    if save_raw:
        f.write(str(dic))
    else:
        for k, v in dic.items():
            f.write(str(k))
            f.write(str(v))
            f.write("\n\n")
    f.close()

def cat_cols_info(X_train: pd.DataFrame, X_test: pd.DataFrame, cat_cols: List[str]) -> dict:
    """
    Get the main info about cat columns in dataframe, i.e. num of values, uniqueness
    :param X_train: Train dataframe
    :param X_test: Test dataframe
    :param cat_cols: List of categorical columns
    :return: Dict with results
    """

    cc_info = {}

    for col in cat_cols:
        train_values = set(X_train[col])
        number_of_new_test = len(set(X_test[col]) - train_values)
        fraction_of_new_test = np.mean(X_test[col].apply(lambda v: v not in train_values))

        cc_info[col] = {
            "num_uniq_train": X_train[col].nunique(), "num_uniq_test": X_test[col].nunique(),
            "number_of_new_test": number_of_new_test, "fraction_of_new_test": fraction_of_new_test
        }
    return cc_info

if __name__ == "__main__":

    with open('../results/file.txt', 'w') as file:
        file.write('a')
    print("utils works")
