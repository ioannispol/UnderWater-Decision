# classifier_base.py

import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


def load_dataset(file_path):
    return pd.read_csv(file_path)


def encode_columns(df, columns):
    encoders = {column: LabelEncoder().fit(df[column]) for column in columns}
    df = df.apply(lambda col: encoders[col.name].transform(col) if col.name in encoders else col)
    return df, encoders


def get_features_target(df, feature_columns, target_column):
    return df[feature_columns], df[target_column]


def split_dataset(features, target, test_size):
    return train_test_split(features, target, test_size=test_size, random_state=42)


def prediction_model(model, features_test, target_test):
    target_pred = model.predict(features_test)
    return classification_report(target_test, target_pred)


def prediction_accuracy(model, features_test, target_test):
    target_pred = model.predict(features_test)
    return accuracy_score(target_test, target_pred)


def print_and_save_output(output, file_path):
    print(output)
    with open(file_path, 'a') as file:
        file.write(output + '\n')


def save_model(model, file_path):
    pickle.dump(model, open(file_path, 'wb'))


def get_decision_path(model, features_test, sample_id):
    feature_names = features_test.columns
    node_indicator = model.decision_path(features_test)
    leaf_id = model.apply(features_test)
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]

    rules = f'Rules used to predict sample {sample_id}:\n'
    for node_id in node_index:
        if (features_test.iloc[sample_id, model.tree_.feature[node_id]] <= model.tree_.threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        rules += (f"decision node {node_id} : (X[{sample_id}, {feature_names[model.tree_.feature[node_id]]}] = "
                  f"{features_test.iloc[sample_id, model.tree_.feature[node_id]]}) "
                  f"{threshold_sign} {model.tree_.threshold[node_id]}\n")
    rules += f'The sample is predicted to be in leaf node {leaf_id[sample_id]}.\n'
    return rules
