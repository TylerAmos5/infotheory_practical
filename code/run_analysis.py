import pandas as pd
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

def load_data(filepath):
    # Load the dataset
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    # 'num': target where value > 0 indicates presence of heart disease
    data['disease'] = (data['num'] > 0).astype(int)
    # Handle missing values by replacing with the mean (questionable)
    data = data.fillna(data.mean())
    return data

def perform_bayesian_inference(data, feature, value):
    # Filter data based on the feature of interest: heart disease
    observed_data = data[data[feature] == value]['disease']
    # Count the num of occurrences / non-occurrences of heart disease
    successes = observed_data.sum()
    failures = observed_data.size - successes
    
    # Prior distribution parameters (randomly chosen)
    alpha_prior = 2
    beta_prior = 8
    
    # Bayesian updating
    alpha_post = alpha_prior + successes
    beta_post = beta_prior + failures
    
    # Plot prior and posterior distributions
    x = np.linspace(0, 1, 100)
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot data
    ax.plot(x, beta.pdf(x, alpha_prior, beta_prior), 'r-', label='Prior Distribution')
    ax.plot(x, beta.pdf(x, alpha_post, beta_post), 'b-', label='Posterior Distribution')

    # Set title and labels with the correct methods
    ax.set_title('Bayesian Updating of Heart Disease Probability')
    ax.set_xlabel('Probability of Heart Disease')
    ax.set_ylabel('Density')

    # Add legend
    ax.legend()

    # Save the plot
    fig.savefig('./output/bayesian_update.png', dpi=300)

    # Close the figure to avoid displaying it
    plt.close(fig)
    return alpha_post, beta_post

if __name__ == "__main__":
    filepath = './data/heart_disease.csv'
    data = load_data(filepath)
    data = preprocess_data(data)
    
    # Check the effect of chest pain (cp) type 4 on heart disease
    feature = 'cp'
    value = 4  # 4 represents intense chest pain
    
    alpha_post, beta_post = perform_bayesian_inference(data, feature, value)
    print("Posterior alpha:", alpha_post)
    print("Posterior beta:", beta_post)
