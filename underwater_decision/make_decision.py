from classifier_base import load_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


decision_tree_model = load_model("path_to_decision_tree_model.pkl")
random_forest_model = load_model("path_to_random_forest_model.pkl")
xgboost_model = load_model("path_to_xgboost_model.pkl")


def make_decision(features):
    decision_tree_output = decision_tree_model.predict(features)
    random_forest_output = random_forest_model.predict(features)
    xgboost_output = xgboost_model.predict(features)

    return decision_tree_output, random_forest_output, xgboost_output
