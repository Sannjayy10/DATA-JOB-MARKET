import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Conv1D, Dense, Dropout, Flatten, GlobalAveragePooling1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("TensorFlow version:", tf.__version__)

SEQ_LENGTH = 30 # Number of past cycles to look at
FEATURES = ['dv_dt', 'dq_dt', 'power', 'internal_resistance', 'overvoltage', 'soc_extreme', 'final_temp', 'temp_rise']
NORM_FEATURES_FOR_SCORE = ['final_temp', 'temp_rise', 'power', 'internal_resistance', 'overvoltage', 'soc_extreme']

def build_sequences(df):
    """ Builds time-series windows per battery cell (file). """
    X, y = [], []
    for file_name, group in df.groupby('file'):
        group = group.sort_values('cycle_number')
        
        feature_data = group[FEATURES].values
        labels = group['risk_class'].values
        
        for i in range(len(feature_data) - SEQ_LENGTH):
            X.append(feature_data[i : i + SEQ_LENGTH])
            # The label is the risk class of the final cycle in the window
            y.append(labels[i + SEQ_LENGTH])
            
    return np.array(X), np.array(y)

def quantize_and_save(keras_model, filename):
    """ Converts Keras model to Float16 Quantized TFLite """
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    # Enable post-training optimization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Set the target type to float16
    converter.target_spec.supported_types = [tf.float16]
    
    # Enable TF Select Ops for RNNs
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    
    tflite_model = converter.convert()
    
    with open(filename, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved Float16 Quantized TFLite: {filename}")

def main():
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv('extracted_cycle_features.csv')
    
    # Fill any NaNs
    df = df.fillna(0)

    # 2. Normalize the specified features
    scaler = MinMaxScaler()
    df[NORM_FEATURES_FOR_SCORE] = scaler.fit_transform(df[NORM_FEATURES_FOR_SCORE])
    
    # Make sure dv_dt and dq_dt are also scaled for neural network training stability
    full_scaler = MinMaxScaler()
    df[FEATURES] = full_scaler.fit_transform(df[FEATURES])

    # 3. Calculate Risk Score
    # risk = 0.25(t) + 0.20(temp_rise) + 0.20(power) + 0.15(int.resistance) + 0.10(soc_extreme)
    df['risk_score'] = (
        0.25 * df['final_temp'] +
        0.20 * df['temp_rise'] +
        0.20 * df['power'] +
        0.15 * df['internal_resistance'] +
        0.10 * df['soc_extreme']
    )
    
    # 4. Thresholding to Classes
    def assign_class(risk):
        if risk <= 0.4:
            return 0
        elif risk <= 0.7:
            return 1
        else:
            return 2
            
    df['risk_class'] = df['risk_score'].apply(assign_class)
    
    print("\nClass Distribution:")
    print(df['risk_class'].value_counts())
    
    # 5. Build Sequences
    X, y = build_sequences(df)
    
    if len(X) == 0:
        print("Error: No sequences could be built. Data might be too short.")
        return
        
    print(f"Constructed input shape: {X.shape}, labels shape: {y.shape}")
    
    # One-hot encode labels
    y_cat = tf.keras.utils.to_categorical(y, num_classes=3)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)
    
    # Class weights for imbalanced data matching user's prior conversations
    # e.g., 25x focus on risk
    class_weights = {0: 1.0, 1: 5.0, 2: 25.0}
    # But let's check true distribution before forcing
    unique, counts = np.unique(y_train.argmax(axis=1), return_counts=True)
    dict_counts = dict(zip(unique, counts))
    print(f"Training split distribution: {dict_counts}")

    input_shape = (X_train.shape[1], X_train.shape[2])

    models = {}
    
    ### LSTM ###
    print("\n--- Training LSTM ---")
    lstm = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])
    lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    lstm.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1, class_weight=class_weights)
    models['LSTM'] = lstm
    
    ### 1D CNN ###
    print("\n--- Training 1D-CNN ---")
    cnn = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'),
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1, class_weight=class_weights)
    models['1D_CNN'] = cnn
    
    ### GRU ###
    print("\n--- Training GRU ---")
    gru = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])
    gru.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    gru.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1, class_weight=class_weights)
    models['GRU'] = gru
    
    # Evaluate and Export
    model_accuracies = {}
    for name, model in models.items():
        print(f"\n================ Evaluating {name} ================")
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        model_accuracies[name] = acc
        print(f"Test Accuracy: {acc*100:.2f}%")
        
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        print(classification_report(y_true, y_pred, zero_division=0))
        
        # Save Confusion Matrix plot
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{name} Confusion Matrix")
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.savefig(f"{name}_confusion_matrix.png")
        plt.close()
        
        # Save exact Keras `.h5` model
        h5_path = f"model_output_{name}.h5"
        model.save(h5_path)
        print(f"Saved Keras Model: {h5_path}")
        
        # Save TFLite quantized
        tflite_path = f"deploy_{name}_fp16.tflite"
        quantize_and_save(model, tflite_path)
    
    # Print Overall Model Accuracy Summary
    print("\n" + "="*50)
    print("OVERALL MODEL ACCURACY SUMMARY")
    print("="*50)
    for name, acc in model_accuracies.items():
        print(f"{name:12} | Test Accuracy: {acc*100:6.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
