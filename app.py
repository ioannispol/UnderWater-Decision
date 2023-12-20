import gradio as gr
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from underwater_unet.model import UNet
from prediction import predict_image, mask_to_image
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

# Load the U-Net models
unet_models = {
    "UW-Unet": "/home/ioannis/Dev/uw-decision/UnderWater-Decision/UnderWaterU-Net/experiment/exp_cx4cf49r/model_epoch_9.pth",
    "UW-Unet1": "/home/ioannis/Dev/uw-decision/UnderWater-Decision/UnderWaterU-Net/experiment/exp_cx4cf49r/model_epoch_5.pth",
    "UW-Unet2": "/home/ioannis/Dev/uw-decision/UnderWater-Decision/UnderWaterU-Net/experiment/exp_cx4cf49r/model_epoch_2.pth"
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a function to load the model
def load_model(model_path):
    model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    return model

loaded_models = {name: load_model(path) for name, path in unet_models.items()}

# Define a function that will take an image input and predict the output using the trained model
def predict(image, model_name):
    model = loaded_models[model_name]  # Get the model based on the model name
    output = predict_image(model, image, device)
    mask_image = mask_to_image(output)

    return mask_image

# Define a function to predict the recommended cleaning method
def predict_cleaning_method(model_name, platform, year, depthmin, depthmax, item, hardPerc, hardmm, softPerc, softmm, Total_Area_Coverage):
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

    return prediction[0]


def combined_function(image, unet_model_name, decision_model_name, platform, year, depthmin, depthmax, item, hardPerc, hardmm, softPerc, softmm, Total_Area_Coverage):
    mask_image = predict(image, unet_model_name)
    recommended_cleaning_method = predict_cleaning_method(decision_model_name, platform, year, depthmin, depthmax, item, hardPerc, hardmm, softPerc, softmm, Total_Area_Coverage)
    return mask_image, recommended_cleaning_method


# Define the Gradio interface
iface = gr.Interface(
    fn=combined_function,
    inputs=[
        gr.Image(),  # Adjust the shape to match your model's input shape
        gr.Dropdown(choices=unet_models, value="UW-Unet", label='U-Net Model'),
        gr.Dropdown(choices=['Decision Tree', 'XGBoost'], label='Decision Model'),
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
        gr.Image(label="Segmentation Mask"),
        gr.Textbox(label="Recommended_Cleaning_Method")
    ],
    examples=[
        ['/home/ioannis/Dev/uw-decision/UnderWater-Decision/UnderWaterU-Net/data/images/076193.jpg', 'UW-Unet', 'Decision Tree', 'platform1', 2022, 10, 20, 'item1', 0.5, 1.0, 0.5, 1.0, 100],
        ['/home/ioannis/Dev/uw-decision/UnderWater-Decision/UnderWaterU-Net/data/images/076350.jpg', 'UW-Unet', 'XGBoost', 'platform2', 2023, 15, 25, 'item2', 0.6, 1.1, 0.6, 1.1, 200]
    ],
    title="Underwater Image Segmentation and Cleaning Recommendation",
    description="Upload an image and select a model to predict the segmentation mask. Then, input the required parameters to get the recommended cleaning method."
)

# Launch the interface
iface.launch(share=True)
