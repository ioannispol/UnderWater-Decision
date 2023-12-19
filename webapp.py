import gradio as gr
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from underwater_decision.decision_tree_train import DecisionTreeModelTrainer, load_dataset, encode_columns, get_features_target, split_dataset
from underwater_decision.xgb_train import train_xgboost_with_grid_search, plot_learning_curve, plot_single_tree

# Load and preprocess the dataset
dataset = load_dataset('data/default_synthetic_dataset1.csv')
dataset, encoders = encode_columns(dataset, ['platform', 'item', 'Recommended_Cleaning_Method'])
feature_columns = ['platform', 'year', 'depthmin', 'depthmax', 'item', 'hardPerc', 'hardmm', 'softPerc', 'softmm', 'Total_Area_Coverage']
target_column = 'Recommended_Cleaning_Method'
features, target = get_features_target(dataset, feature_columns, target_column)
features_train, features_test, target_train, target_test = split_dataset(features, target, test_size=0.3)

# Train the Decision Tree model
model_trainer = DecisionTreeModelTrainer()
model, best_params, best_score = model_trainer.train_with_grid_search(features_train, target_train)

# Train the XGBoost model
xgb_model, _, _ = train_xgboost_with_grid_search(features_train, target_train)

# Define a dictionary to map from model names to models
models = {
    'Decision Tree': model,
    'XGBoost': xgb_model,
}

def predict(model_name, platform, year, depthmin, depthmax, item, hardPerc, hardmm, softPerc, softmm, Total_Area_Coverage):
    # Select the model based on the user's choice
    model = models[model_name]

    # Create a DataFrame from the inputs
    data = pd.DataFrame([[platform, year, depthmin, depthmax, item, hardPerc, hardmm, softPerc, softmm, Total_Area_Coverage]], columns=feature_columns)
    
    # Encode the categorical columns
    data['platform'] = encoders['platform'].transform([data['platform']])
    data['item'] = encoders['item'].transform([data['item']])
    
    # Make the prediction
    prediction = model.predict(data)
    
    # Decode the prediction
    prediction = encoders[target_column].inverse_transform(prediction)

    # Generate the learning curve and tree plot for the XGBoost model
    if model_name == 'XGBoost':
        learning_curve_file = 'learning_curve.png'
        tree_plot_file = 'xgb_tree.png'
        plot_learning_curve(model, "XGBoost Learning Curve", features_train, target_train, cv=5, n_jobs=-1, file_path=learning_curve_file)
        plot_single_tree(model, 0, tree_plot_file)

        return prediction[0], learning_curve_file, tree_plot_file

    return prediction[0], None, None

iface = gr.Interface(fn=predict,
                     inputs = [
    gr.Dropdown(choices=['Decision Tree', 'XGBoost'], label='Model'),
    gr.Textbox(label='platform'),
    gr.Number(label='year'),
    gr.Number(label='depthmin'),
    gr.Number(label='depthmax'),
    gr.Textbox(label='item'),
    gr.Number(label='hardPerc'),
    gr.Number(label='hardmm'),
    gr.Number(label='softPerc'),
    gr.Number(label='softmm'),
    gr.Number(label='Total_Area_Coverage')
],
                      outputs=[
    gr.Textbox(label="Recommended_Cleaning_Method"),
    gr.Image(label="Learning Curve"),
    gr.Image(label="Tree Plot"),
                      ])

iface.launch()