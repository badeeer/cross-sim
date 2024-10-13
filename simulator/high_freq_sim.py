


import sys
import os
from pathlib import Path

# Add the parent directory of High_freq_sim to the Python path
current_dir = Path(__file__).resolve().parent
cross_sim_dir = current_dir.parent
sys.path.append(str(cross_sim_dir))

# Add the simulator directory to the Python path
simulator_dir = cross_sim_dir / "simulator"
sys.path.append(str(simulator_dir))

# Verify the existence of Octave_implem.py and raise a specific error if not found
octave_implem_path = simulator_dir / "Octave_implem.py"
if not octave_implem_path.exists():
    raise FileNotFoundError(f"Octave_implem.py not found at expected path: {octave_implem_path}")

# Verify the existence of prepare_models.py and raise a specific error if not found
prepare_models_path = simulator_dir / "prepare_models.py"
if not prepare_models_path.exists():
    raise FileNotFoundError(f"prepare_models.py not found at expected path: {prepare_models_path}")

# Directly import OctaveResNet from Octave_implem.py
import importlib.util
spec = importlib.util.spec_from_file_location("Octave_implem", octave_implem_path)
Octave_implem = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Octave_implem)
OctaveResNet = Octave_implem.OctaveResNet

# Import necessary functions from prepare_models
from prepare_models import define_compile_model, augment

import tensorflow as tf
import numpy as np
from typing import Tuple

# Import CrossSimParameters if it exists, otherwise create a dummy class
try:
    from simulator.parameters.crosssim_parameters import CrossSimParameters
except ImportError:
    print("Warning: CrossSimParameters not found. Using a dummy class.")
    class CrossSimParameters:
        def __init__(self):
            self.simulation = type('obj', (object,), {'use_high_freq_sim': True})

class HighFreqSimCore:
    def __init__(self, params: CrossSimParameters):
        self.params = params
        self.use_high_freq_sim = getattr(params.simulation, 'use_high_freq_sim', True)

        # Define the high-frequency processor
        conv1 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same')
        original_model = tf.keras.Sequential([
            conv1,
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        ])
        low_freq_processor = tf.keras.layers.Conv2D(32, kernel_size=3, padding='same')

        # Initialize OctaveResNet
        self.high_freq_processor = OctaveResNet(
            original_model=original_model,
            low_freq_processor=low_freq_processor,
            conv1=conv1,
            high_freq_processor=tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
            ])
        )

        # Load and preprocess CIFAR-10 dataset
        (x_train, y_train), (x_test, y_test) = self.load_cifar10_data()
        self.x_train, self.y_train, self.x_val, self.y_val = self.prepare_data_splits(x_train, y_train, validation_split=0.1)

        # Prepare datasets
        self.batch_size = 32
        self.train_dataset = self.prepare_tf_dataset(self.x_train, self.y_train)
        self.validation_dataset = self.prepare_tf_dataset(self.x_val, self.y_val, validation=True)

        # Early stopping callbacks
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                mode='min'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                mode='max'
            )
        ]

    def load_cifar10_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        return (x_train, y_train), (x_test, y_test)

    def prepare_data_splits(self, x_train: np.ndarray, y_train: np.ndarray, validation_split: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        num_val_samples = int(validation_split * len(x_train))
        return (x_train[:-num_val_samples], y_train[:-num_val_samples], 
                x_train[-num_val_samples:], y_train[-num_val_samples:])

    def prepare_tf_dataset(self, x_data: np.ndarray, y_data: np.ndarray, validation: bool = False) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        if not validation:
            dataset = dataset.shuffle(1024).map(augment)  # Using the imported augment function
        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def train(self, epochs: int = 50) -> None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.high_freq_processor.compile(optimizer=optimizer,
                                         loss='sparse_categorical_crossentropy',
                                         metrics=['accuracy'])
        
        history = self.high_freq_processor.fit(
            self.train_dataset,
            epochs=epochs,
            validation_data=self.validation_dataset,
            callbacks=self.callbacks
        )
        
        # Print final metrics
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        print(f"Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

    def process(self, x: tf.Tensor) -> tf.Tensor:
        return self.high_freq_processor(x)

def integrate_high_freq_sim(core_class):
    original_call = core_class.__call__

    def new_call(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, 'high_freq_sim'):
            self.high_freq_sim = HighFreqSimCore(self.params)
        x = self.high_freq_sim.process(x)
        return original_call(self, x)

    core_class.__call__ = new_call
    return core_class

if __name__ == "__main__":
    try:
        params = CrossSimParameters()
        high_freq_sim = HighFreqSimCore(params)
        high_freq_sim.train(epochs=10)  # Train for 10 epochs as a test
        print("HighFreqSimCore module loaded and tested successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise