

import sys
import os
from pathlib import Path

# Add the parent directory of the current file to the Python path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Add the root directory to the Python path
root_dir = parent_dir.parent  # Assuming the script is in simulator/cores/
sys.path.append(str(root_dir))

# Import statements using relative imports
from backend.backend import ComputeBackend
from prepare_models import define_compile_model, augment
from Octave_implem import OctaveResNet
from parameters.crosssim_parameters import CrossSimParameters  

import tensorflow as tf
import numpy as np
import numpy.typing as npt

# Initialize ComputeBackend
xp = ComputeBackend()

class LowFreqSimCore:
    def __init__(self, params: CrossSimParameters):
        """
        Initialize the LowFreqSimCore with CrossSimParameters.
        
        Parameters:
        - params: CrossSimParameters object containing the parameters for simulation.
        """
        self.params = params
        self.use_low_freq_sim = getattr(params.simulation, 'use_low_freq_sim', False)

        # Initialize OctaveResNet with the appropriate parameters
        conv1 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same')
        original_model = tf.keras.Sequential([
            conv1,
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        ])
        high_freq_processor = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same')

        # Initialize OctaveResNet with the low-frequency processor
        self.low_freq_processor = OctaveResNet(
            original_model=original_model,
            high_freq_processor=high_freq_processor,
            conv1=conv1,
            low_freq_processor=tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
            ])
        )

        # Load and preprocess CIFAR-10 dataset for low-frequency simulation
        (x_train, y_train), (x_test, y_test) = self.load_cifar10_data()
        self.x_train, self.y_train, self.x_val, self.y_val = self.prepare_data_splits(x_train, y_train, validation_split=0.1)

        # Prepare the dataset for training and validation
        self.batch_size = 4
        self.train_dataset = self.prepare_tf_dataset(self.x_train, self.y_train)
        self.validation_dataset = self.prepare_tf_dataset(self.x_val, self.y_val, validation=False)

    def load_cifar10_data(self):
        """
        Load CIFAR-10 dataset.
        
        Returns:
        - Tuple of train and test datasets.
        """
        return tf.keras.datasets.cifar10.load_data()

    def prepare_data_splits(self, x_train, y_train, validation_split=0.1):
        """
        Split the training data into training and validation datasets.
        
        Parameters:
        - x_train: Training images.
        - y_train: Training labels.
        - validation_split: Percentage of data to be used for validation.
        
        Returns:
        - Training and validation datasets.
        """
        num_val = int(validation_split * len(x_train))
        x_val, y_val = x_train[:num_val], y_train[:num_val]
        x_train, y_train = x_train[num_val:], y_train[num_val:]
        return x_train, y_train, x_val, y_val

    def prepare_tf_dataset(self, x, y, validation=True):
        """
        Prepare TensorFlow dataset.
        
        Parameters:
        - x: Images.
        - y: Labels.
        - validation: Boolean to indicate if this is for validation.
        
        Returns:
        - Prepared TensorFlow dataset.
        """
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if validation:
            dataset = dataset.batch(self.batch_size)
        else:
            dataset = dataset.batch(self.batch_size).shuffle(buffer_size=1024)
        return dataset

    # Additional methods for processing can be added here.

# Function to integrate LowFreqSimCore with AnalogCore or any other core
def integrate_low_freq_sim(core_class):
    """
    Integrate LowFreqSimCore into an existing core class.

    Parameters:
    - core_class: The core class to integrate with LowFreqSimCore.

    Returns:
    - The modified core class with low-frequency simulation integration.
    """
    original_call = core_class.__call__

    def new_call(self, x: npt.ArrayLike) -> npt.ArrayLike:
        if not hasattr(self, 'low_freq_sim'):
            self.low_freq_sim = LowFreqSimCore(self.params)
        x = self.low_freq_sim.process(x)
        return original_call(self, x)

    core_class.__call__ = new_call
    return core_class

if __name__ == "__main__":
    print("LowFreqSimCore module loaded successfully.")

    
