
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from sklearn.model_selection import train_test_split
import optuna
from optuna.samplers import TPESampler

# Create dummy data for demonstration (replace with your dataset)
def get_dummy_data(n_samples=1000, seq_len=100, n_features=1, n_classes=2):
    X = np.random.rand(n_samples, seq_len, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, size=n_samples)
    y = tf.keras.utils.to_categorical(y, num_classes=n_classes)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function for Optuna
def objective(trial):
    # Load data
    X_train, X_val, y_train, y_val = get_dummy_data()

    # Suggest hyperparameters
    n_filters = trial.suggest_int("n_filters", 16, 128, step=16)
    kernel_size = trial.suggest_int("kernel_size", 3, 7)
    stride = trial.suggest_int("stride", 1, 2)
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])

    # Build model
    model = Sequential()
    model.add(Conv1D(filters=n_filters,
                     kernel_size=kernel_size,
                     strides=stride,
                     activation='relu',
                     input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    # Choose optimizer
    optimizer_dict = {
        "Adam": Adam(learning_rate=learning_rate),
        "RMSprop": RMSprop(learning_rate=learning_rate),
        "SGD": SGD(learning_rate=learning_rate)
    }

    model.compile(optimizer=optimizer_dict[optimizer_name],
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=10,
                        batch_size=batch_size,
                        verbose=0)

    # Evaluate
    val_accuracy = history.history['val_accuracy'][-1]
    return 1.0 - val_accuracy  # minimize (1 - accuracy)

# Create study with multivariate TPE sampler
sampler = TPESampler(multivariate=True, seed=42)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=30)

# Show best trial
print("Best trial:")
trial = study.best_trial
print(f"  Value (1 - accuracy): {trial.value}")
print("  Params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
