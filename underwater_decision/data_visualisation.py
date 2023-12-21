import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Function to load the data and plot
def plot_data(csv_path):
    # Load the data
    data = pd.read_csv(csv_path)

    # Define the columns you want to plot against Total_Area_Coverage
    columns_to_plot = ["hardPerc", "hardmm", "softPerc", "softmm"]
    # Define a list of colors that will be used for the scatter plots
    colors = list(
        mcolors.TABLEAU_COLORS
    )  # This gives a list of visually distinct colors

    # Create a figure with two subplots
    fig, (ax_scatter, ax_bar) = plt.subplots(
        2, 1, figsize=(12, 16), gridspec_kw={"height_ratios": [3, 1]}
    )

    # Scatter plots in the first subplot
    for column, color in zip(columns_to_plot, colors):
        ax_scatter.scatter(
            data[column],
            data["Total_Area_Coverage"],
            alpha=0.5,
            label=f"{column} vs Total Area Coverage",
            color=color,
        )

    ax_scatter.set_title(
        "Scatter Plots of Fouling Characteristics vs Total Area Coverage"
    )
    ax_scatter.set_xlabel("Measurement Value")
    ax_scatter.set_ylabel("Total Area Coverage")
    ax_scatter.legend(loc="upper left")

    # Bar chart in the second subplot
    method_counts = data["Recommended_Cleaning_Method"].value_counts()
    ax_bar.bar(method_counts.index, method_counts.values, color="gray")
    ax_bar.set_title("Counts of Recommended Cleaning Method")
    ax_bar.set_xlabel("Cleaning Method")
    ax_bar.set_ylabel("Counts")
    ax_bar.tick_params(axis="x", rotation=45)  # Rotate x labels for better readability

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()


# Call the function with the path to your CSV file
plot_data("/workspaces/UnderWater-Decision/data/default_synthetic_dataset.csv")
