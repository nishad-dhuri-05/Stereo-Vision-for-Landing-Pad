#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stereovision ANN Model for Pi Camera v1.3
-----------------------------------------
This script builds an Artificial Neural Network to predict true disparity (True D) 
and true height (True H) from stereoscopic vision data using Pi Camera v1.3.

Author: AI Assistant
Date: 2025-03-18
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create results directory
if not os.path.exists('results'):
    os.makedirs('results')

def prepare_data(csv_file_path):
    """Prepare and visualize the dataset from CSV file"""
    # Load data from CSV
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Successfully loaded data from {csv_file_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
    
    # Ensure column names match expected format
    # Map column names if they're different in the CSV
    column_mapping = {
        'True D': 'True_D',
        'True H': 'True_H',
        'Sensor size': 'Sensor_size',
        'Image width': 'Image_width'
    }
    df = df.rename(columns=column_mapping)
    
    # Feature engineering - calculating disparity
    df['Disparity'] = df['R'] - df['L']
    
    # Visualize data relationships
    plt.figure(figsize=(16, 12))
    
    # Plot relationships
    plt.subplot(2, 2, 1)
    sns.regplot(x='Disparity', y='True_D', data=df)
    plt.xlabel('Calculated Disparity (R-L)')
    plt.ylabel('True Disparity')
    plt.title('Calculated vs True Disparity')
    
    plt.subplot(2, 2, 2)
    sns.regplot(x='True_D', y='True_H', data=df)
    plt.xlabel('True Disparity')
    plt.ylabel('True Height')
    plt.title('Disparity vs Height')
    
    plt.subplot(2, 2, 3)
    sns.regplot(x='L', y='R', data=df)
    plt.xlabel('Left Pixel (L)')
    plt.ylabel('Right Pixel (R)')
    plt.title('Left vs Right Pixel Values')
    
    plt.subplot(2, 2, 4)
    plt.plot(df.index, df['Disparity'], 'o-', label='Calculated Disparity')
    plt.plot(df.index, df['True_D'], 'o-', label='True Disparity')
    plt.xlabel('Sample Index')
    plt.ylabel('Disparity Value')
    plt.title('Calculated vs True Disparity')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/data_relationships.png')
    plt.close()
    
    return df

def build_and_train_model(X, y):
    """Build and train the ANN model"""
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build model
    model = Sequential([
        Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(2)  # Output layer for True_D and True_H
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        epochs=300,
        batch_size=4,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/training_history.png')
    plt.close()
    
    return model, scaler, X_test_scaled, y_test

def evaluate_model(model, X_test_scaled, y_test):
    """Evaluate the model and visualize predictions"""
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Convert to DataFrames
    y_pred_df = pd.DataFrame(y_pred, columns=['Predicted_D', 'Predicted_H'])
    y_test_df = y_test.reset_index(drop=True)
    
    # Combine actual and predicted values
    results_df = pd.concat([y_test_df, y_pred_df], axis=1)
    
    # Calculate metrics
    mse_d = mean_squared_error(results_df['True_D'], results_df['Predicted_D'])
    mae_d = mean_absolute_error(results_df['True_D'], results_df['Predicted_D'])
    r2_d = r2_score(results_df['True_D'], results_df['Predicted_D'])
    
    mse_h = mean_squared_error(results_df['True_H'], results_df['Predicted_H'])
    mae_h = mean_absolute_error(results_df['True_H'], results_df['Predicted_H'])
    r2_h = r2_score(results_df['True_H'], results_df['Predicted_H'])
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    print(f"True Disparity (D):")
    print(f"  MSE: {mse_d:.4f}")
    print(f"  MAE: {mae_d:.4f}")
    print(f"  R²: {r2_d:.4f}")
    
    print(f"\nTrue Height (H):")
    print(f"  MSE: {mse_h:.4f}")
    print(f"  MAE: {mae_h:.4f}")
    print(f"  R²: {r2_h:.4f}")
    
    # Visualize predictions
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(results_df['True_D'], results_df['Predicted_D'])
    plt.plot([results_df['True_D'].min(), results_df['True_D'].max()],
             [results_df['True_D'].min(), results_df['True_D'].max()],
             'k--')
    plt.xlabel('True Disparity')
    plt.ylabel('Predicted Disparity')
    plt.title('True vs Predicted Disparity')
    
    plt.subplot(1, 2, 2)
    plt.scatter(results_df['True_H'], results_df['Predicted_H'])
    plt.plot([results_df['True_H'].min(), results_df['True_H'].max()],
             [results_df['True_H'].min(), results_df['True_H'].max()],
             'k--')
    plt.xlabel('True Height')
    plt.ylabel('Predicted Height')
    plt.title('True vs Predicted Height')
    
    plt.tight_layout()
    plt.savefig('results/prediction_performance.png')
    plt.close()
    
    return results_df

def predict_stereo_values(model, scaler, left_px, right_px):
    """Make predictions on new stereo vision data"""
    # Calculate disparity
    disparity = right_px - left_px
    
    # Create feature array
    features = np.array([[left_px, right_px, disparity]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    
    return {
        'Predicted_Disparity': prediction[0][0],
        'Predicted_Height': prediction[0][1]
    }

def main():
    """Main function to run the entire workflow"""
    print("=== Stereovision ANN Model for Pi Camera v1.3 ===\n")
    
    # Get CSV file path from user
    csv_file_path = input("Enter the path to the CSV file containing the input_train data: ")
    
    # Prepare data
    print("Preparing and visualizing data...")
    df = prepare_data(csv_file_path)
    
    if df is None:
        print("Exiting due to data loading error.")
        return
    
    # Define features and targets
    X = df[['L', 'R', 'Disparity']]
    y = df[['True_D', 'True_H']]
    
    # Build and train model
    print("\nBuilding and training ANN model...")
    model, scaler, X_test_scaled, y_test = build_and_train_model(X, y)
    
    # Evaluate model
    print("\nEvaluating model performance...")
    results_df = evaluate_model(model, X_test_scaled, y_test)
    
    # Save model and results
    model.save('results/stereo_vision_model.h5')
    import joblib
    joblib.dump(scaler, 'results/feature_scaler.pkl')
    results_df.to_csv('results/prediction_results.csv', index=False)
    
    # Example predictions
    print("\nExample predictions:")
    examples = [
        (800, 980),    # Similar to first row
        (1000, 1120),  # Middle range
        (1190, 1230)   # Similar to last row
    ]
    
    for left, right in examples:
        prediction = predict_stereo_values(model, scaler, left, right)
        print(f"\nFor L={left}, R={right}:")
        print(f"  Predicted Disparity: {prediction['Predicted_Disparity']:.2f}")
        print(f"  Predicted Height: {prediction['Predicted_Height']:.2f}")
    
    print("\n=== Model training and evaluation complete ===")
    print("Results and model saved in the 'results' directory")

if __name__ == "__main__":
    main()
