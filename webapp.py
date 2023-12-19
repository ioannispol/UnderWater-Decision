import gradio as gr
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from underwater_decision.decision_tree_train import DecisionTreeModelTrainer, load_dataset, encode_columns, get_features_target, split_dataset

# Load and preprocess the dataset
dataset = load_dataset('data/default_synthetic_dataset.csv')
dataset, encoders = encode_columns(dataset, ['platform', 'item', 'Recommended_Cleaning_Method'])
feature_columns = ['platform', 'year', 'depthmin', 'depthmax', 'item', 'hardPerc', 'hardmm', 'softPerc', 'softmm', 'Total_Area_Coverage']
target_column = 'Recommended_Cleaning_Method'
features, target = get_features_target(dataset, feature_columns, target_column)
features_train, features_test, target_train, target_test = split_dataset(features, target, test_size=0.3)

# Train the Decision Tree model
model_trainer = DecisionTreeModelTrainer()
model, best_params, best_score = model_trainer.train_with_grid_search(features_train, target_train)

def predict(platform, year, depthmin, depthmax, item, hardPerc, hardmm, softPerc, softmm, Total_Area_Coverage):
    # Create a DataFrame from the inputs
    data = pd.DataFrame([[platform, year, depthmin, depthmax, item, hardPerc, hardmm, softPerc, softmm, Total_Area_Coverage]], columns=feature_columns)
    
    # Encode the categorical columns
    data['platform'] = encoders['platform'].transform([data['platform']])
    data['item'] = encoders['item'].transform([data['item']])
    
    # Make the prediction
    prediction = model.predict(data)
    
    # Decode the prediction
    prediction = encoders[target_column].inverse_transform(prediction)
    
    return prediction[0]


iface = gr.Interface(fn=predict,
                     inputs = [
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
                     outputs={"text": "Recommended_Cleaning_Method"})
iface.launch()