import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def load_data(file_path):
    """Load the heart disease dataset."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the data by splitting features and target."""
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

def split_train_test(X, y, test_size=0.3, random_state=42):
    """Split the data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_features(X_train, X_test):
    """Scale the features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def create_neural_network_model(input_dim, hidden_units=64, output_units=1):
    """Create a neural network model."""
    model = Sequential([
        Dense(hidden_units, activation="relu", input_shape=(input_dim,)),
        Dense(output_units, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_model(model, X_train, y_train, epochs=100, batch_size=32):
    """Train the neural network model."""
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model on the test set."""
    predictions = model.predict(X_test)
    y_pred = (predictions > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    
    return accuracy

def main():
    file_path = "heart.csv"
    data = load_data(file_path)
    
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    model = create_neural_network_model(input_dim=X.shape[1], hidden_units=64, output_units=1)
    
    history = train_model(model, X_train_scaled, y_train)
    
    final_accuracy = evaluate_model(model, X_test_scaled, y_test)
    
    print(f"\nFinal Accuracy: {final_accuracy:.2f}%")

if __name__ == "__main__":
    main()
