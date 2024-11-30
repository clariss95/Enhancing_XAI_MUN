import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Load preprocessed data
event_features = np.load('event_features.npy')
case_features = np.load('case_features.npy')
labels = np.load('labels.npy')

# Define the sequence length
sequence_length = 2
num_event_features = event_features.shape[2]
num_case_features = case_features.shape[1]

# Training, validation, and testing splits
train_idx = int(.6 * len(event_features))
val_idx = int(.8 * len(event_features))

train_features_event = event_features[:train_idx]
val_features_event = event_features[train_idx:val_idx]
test_features_event = event_features[val_idx:]

train_features_case = case_features[:train_idx]
val_features_case = case_features[train_idx:val_idx]
test_features_case = case_features[val_idx:]

train_labels = labels[:train_idx]
val_labels = labels[train_idx:val_idx]
test_labels = labels[val_idx:]

# Resample the training data using SMOTE
from imblearn.over_sampling import SMOTE
train_features_combined = np.hstack([train_features_event.reshape(train_features_event.shape[0], -1), train_features_case])
smote = SMOTE(sampling_strategy='auto')
train_features_resampled, train_labels_resampled = smote.fit_resample(train_features_combined, train_labels)

# Separate the resampled features back into event and case features
num_samples_train = train_features_resampled.shape[0]
train_features_event_resampled = train_features_resampled[:, :num_event_features * sequence_length].reshape(num_samples_train, sequence_length, num_event_features)
train_features_case_resampled = train_features_resampled[:, num_event_features * sequence_length:]

# Standardize the features
from sklearn.preprocessing import StandardScaler
scaler_event = StandardScaler()
scaler_case = StandardScaler()

train_features_event_resampled = scaler_event.fit_transform(train_features_event_resampled.reshape(-1, num_event_features)).reshape(num_samples_train, sequence_length, num_event_features)
train_features_case_resampled = scaler_case.fit_transform(train_features_case_resampled)

val_features_event = scaler_event.transform(val_features_event.reshape(-1, num_event_features)).reshape(val_features_event.shape[0], sequence_length, num_event_features)
val_features_case = scaler_case.transform(val_features_case)

test_features_event = scaler_event.transform(test_features_event.reshape(-1, num_event_features)).reshape(test_features_event.shape[0], sequence_length, num_event_features)
test_features_case = scaler_case.transform(test_features_case)

# Create TensorFlow datasets
train_dataset_resampled = tf.data.Dataset.from_tensor_slices(((train_features_event_resampled, train_features_case_resampled), train_labels_resampled)).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices(((val_features_event, val_features_case), val_labels)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices(((test_features_event, test_features_case), test_labels)).batch(32)


# Define the baseline model structure (if not already saved)
def build_baseline_model():
    event_input = Input(shape=(sequence_length, num_event_features), name='event_input')
    lstm_out = LSTM(64, return_sequences=True)(event_input)  # Baseline: LSTM Units 1: 64
    lstm_out = Dropout(0.5)(lstm_out)  # Baseline: Dropout Rate: 0.5
    lstm_out = LSTM(32, return_sequences=False)(lstm_out)  # Baseline: LSTM Units 2: 32
    lstm_out = Dropout(0.5)(lstm_out)  # Baseline: Dropout Rate: 0.5
    lstm_out = BatchNormalization()(lstm_out)

    case_input = Input(shape=(num_case_features,), name='case_input')
    combined = Concatenate()([lstm_out, case_input])
    output = Dense(1, activation='sigmoid')(combined)

    model = Model(inputs=[event_input, case_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Baseline Learning Rate: 0.0001
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

try:
    # Load the pre-trained baseline model
    baseline_model = load_model('baseline_model.h5')
    print("Baseline model loaded successfully!")
except:
    # If not already saved, build and retrain the baseline model
    print("Baseline model not found, training it again.")
    baseline_model = build_baseline_model()

    # Callbacks for training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
        ModelCheckpoint('baseline_model.h5', save_best_only=True, monitor='val_loss')
    ]

    # Train the baseline model
    history = baseline_model.fit(
        train_dataset_resampled,
        validation_data=val_dataset,
        epochs=20,
        callbacks=callbacks
    )

# Evaluate the baseline model on the test set
test_loss_baseline, test_accuracy_baseline = baseline_model.evaluate(test_dataset)
print(f"Baseline Model - Test Loss: {test_loss_baseline}, Test Accuracy: {test_accuracy_baseline}")

# Save the test results for the baseline
with open('baseline_test_results.txt', 'w') as f:
    f.write(f"Test Loss: {test_loss_baseline}\n")
    f.write(f"Test Accuracy: {test_accuracy_baseline}\n")
