# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "chardet",
#     "matplotlib",
#     "pandas",
#     "statsmodels",
#     "scikit-learn",
#     "missingno",
#     "python-dotenv",
#     "requests",
#     "seaborn",
# ]
# ///
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import missingno as msno
from scipy import stats
import requests
import os
import chardet

# Set the AIPROXY_TOKEN environment variable if not already set
if "AIPROXY_TOKEN" not in os.environ:
    api_key = input("Please enter your OpenAI API key: ")
    os.environ["AIPROXY_TOKEN"] = api_key

api_key = os.environ["AIPROXY_TOKEN"]

# Function to detect file encoding
def detect_encoding(filename):
    with open(filename, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

# Function to load and clean the dataset
def load_and_clean_data(filename):
    encoding = detect_encoding(filename)
    df = pd.read_csv(filename, encoding=encoding)
    
    # Drop rows with all NaN values
    df.dropna(axis=0, how='all', inplace=True)
    
    # Fill missing values in numeric columns with the mean of the column
    numeric_columns = df.select_dtypes(include='number')
    df[numeric_columns.columns] = numeric_columns.fillna(numeric_columns.mean())
    
    # Handle missing values in non-numeric columns (e.g., fill with 'Unknown')
    non_numeric_columns = df.select_dtypes(exclude='number')
    df[non_numeric_columns.columns] = non_numeric_columns.fillna('Unknown')
    
    return df

# Function to summarize the dataset
def summarize_data(df):
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'types': df.dtypes.to_dict(),
        'descriptive_statistics': df.describe().to_dict(),
        'missing_values': df.isnull().sum().to_dict()
    }
    return summary

# Outlier detection function using Z-Score
def detect_outliers(df):
    numeric_df = df.select_dtypes(include=[np.number])
    z_scores = np.abs(stats.zscore(numeric_df))
    outliers = (z_scores > 3).sum(axis=0)
    outlier_info = {
        column: int(count) for column, count in zip(numeric_df.columns, outliers)
    }
    return outlier_info

# Correlation analysis function
def correlation_analysis(df):
    numeric_df = df.select_dtypes(include='number')
    correlation_matrix = numeric_df.corr()
    return correlation_matrix.to_dict()

# Cluster analysis using KMeans
def perform_clustering(df, n_clusters=3):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_scaled)
    return df, kmeans

# PCA for dimensionality reduction (optional)
def perform_pca(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df_scaled)
    df['PCA1'] = pca_components[:, 0]
    df['PCA2'] = pca_components[:, 1]
    return df

# Function to create visualizations
import seaborn as sns
import matplotlib.pyplot as plt

# Function to create visualizations
def create_visualizations(df):
    # Visualization for missing data
    import missingno as msno
    msno.matrix(df)
    plt.tight_layout()  # Adjust layout
    missing_img = 'missing_data.png'
    plt.savefig(missing_img)
    plt.close()

    # Filter numeric columns for correlation heatmap
    numeric_df = df.select_dtypes(include='number')
    
    if numeric_df.shape[1] > 1:  # Ensure there are more than one numeric column for correlation
        # Correlation heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Matrix")
        correlation_img = 'correlation_matrix.png'
        plt.tight_layout()  # Adjust layout to prevent warnings
        plt.savefig(correlation_img)
        plt.close()
    else:
        correlation_img = None  # If no numeric columns, set as None

    # Cluster visualization (after performing PCA)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='Set1')
    plt.title("Cluster Analysis (PCA)")
    cluster_img = 'cluster_analysis.png'
    plt.tight_layout()  # Adjust layout to prevent warnings
    plt.savefig(cluster_img)
    plt.close()

    return [missing_img, correlation_img, cluster_img] if correlation_img else [missing_img, cluster_img]


# Function to generate GPT-4o-Mini analysis story
def generate_analysis_story(summary, outliers, correlation_matrix):
    # Prepare the prompt with the data
    prompt = f"""
    Given the following dataset summary:
    - Shape: {summary['shape']}
    - Columns: {', '.join(summary['columns'])}
    - Data Types: {summary['types']}
    - Descriptive Statistics: {summary['descriptive_statistics']}
    - Missing values: {summary['missing_values']}

    Additionally, the outliers detected are: {outliers}

    The correlation matrix of the dataset is:
    {correlation_matrix}

    Generate a detailed, insightful analysis of the dataset. Include an interpretation of the statistics, correlations, outliers, and any recommendations for further analysis. Additionally, narrate the findings in a story-like manner to convey the insights effectively.
    """

    # Call OpenAI API (GPT-4 or GPT-4o-Mini) to generate the story
    try:
        response = requests.post(
            "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": prompt},
                ]
            }
        )
        response.raise_for_status()  # Check for request errors
        result = response.json()
        story = result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"Error generating analysis story: {e}")
        return "Error generating analysis story"
    
    return story

# Function to write the README with analysis story and results
def write_readme(summary, outliers, correlation_matrix, visualizations, story, filename):
    with open('README.md', 'w') as f:
        # Header
        f.write(f"# Dataset Analysis of {filename}\n")

        # Dataset Summary
        f.write("\n## Dataset Summary\n")
        f.write(f"- Shape of the dataset: {summary['shape']}\n")
        f.write(f"- Columns: {', '.join(summary['columns'])}\n")
        f.write(f"- Data types:\n{summary['types']}\n")
        f.write(f"- Descriptive statistics:\n{summary['descriptive_statistics']}\n")
        f.write(f"- Missing values per column:\n{summary['missing_values']}\n")

        # Outlier Analysis
        f.write("\n## Outlier Detection\n")
        f.write(f"Outliers detected in each numeric column (Z-score > 3):\n{outliers}\n")

        # Correlation Analysis
        f.write("\n## Correlation Analysis\n")
        f.write(f"Correlation Matrix:\n{correlation_matrix}\n")

        # Narrative Story (from GPT-4o-Mini)
        f.write("\n## Dataset Analysis Story\n")
        f.write(f"{story}\n")

        # Visualizations
        f.write("\n## Visualizations\n")
        for img in visualizations:
            f.write(f"![{img}]({img})\n")

# Main function
def main(filename):
    # Load and clean data
    df = load_and_clean_data(filename)

    # Summarize the data
    summary = summarize_data(df)

    # Outlier detection
    outliers = detect_outliers(df)

    # Correlation analysis
    correlation_matrix = correlation_analysis(df)

    # Perform clustering
    df, kmeans = perform_clustering(df)

    # Perform PCA for visualization
    df = perform_pca(df)

    # Create visualizations
    visualizations = create_visualizations(df)

    # Generate GPT-4o-Mini analysis story
    story = generate_analysis_story(summary, outliers, correlation_matrix)

    # Write README file
    write_readme(summary, outliers, correlation_matrix, visualizations, story, filename)

    print(f"Analysis complete. Results saved in 'README.md'.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
    else:
        main(sys.argv[1])
