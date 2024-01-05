import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def visualize_data(data):   
    # Distribution of trip durations
    plt.figure(figsize=(8, 5))
    sns.histplot(data['trip_duration'], bins=50, kde=True)
    plt.title('Distribution of Trip Durations')
    plt.xlabel('Trip Duration')
    plt.ylabel('Frequency')
    plt.savefig('data_preprocessing_visualizations/trip_durations_distribution.png')
    print("Saved visualization: trip_durations_distribution.png")

    # Distributions of potential features
    features = ['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                 'dropoff_latitude', 'vendor_id']
    for feature in features:
        plt.figure(figsize=(8, 5))
        sns.histplot(data[feature], bins=50, kde=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.savefig(f'data_preprocessing_visualizations/{feature}_distribution.png')
        print(f"Saved visualization: {feature}_distribution.png")

    # Correlations
    correlation_matrix = data[features + ['trip_duration']].corr()
    plt.figure(figsize=(13, 13))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('data_preprocessing_visualizations/correlation_matrix.png')
    print("Saved visualization: correlation_matrix.png")

def preprocess_data():
    # Load dataset
    train_data = pd.read_csv('./nyc-taxi-trip-duration/train.csv')
    test_data = pd.read_csv('./nyc-taxi-trip-duration/test.csv')

    # Process dataset: Here you can add more preprocessing steps as needed
    # For this example, I'll only use the numeric columns
    features = ['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
    
    X = train_data[features]
    y = train_data['trip_duration']

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)
    
    # For test set, we don't have the 'trip_duration' column, so we'll just use the features
    X_test = test_data[features]

    return X_train, y_train, X_val, y_val, X_test
    

'''Used for visualization
if __name__ == "__main__":
    # Load the data
    train_data = pd.read_csv("'./nyc-taxi-trip-duration/train.csv")
    
    # Visualize the data
    #visualize_data(train_data)
'''