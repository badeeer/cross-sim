
# File: cross-sim/simulator/cores/octave_conv_integration.py


import sys
import os
from pathlib import Path

# Add the parent directory of cross-sim to the Python path
current_dir = Path(__file__).resolve().parent
cross_sim_parent = current_dir.parent.parent
sys.path.append(str(cross_sim_parent))

# Add the High_freq_sim directory to the Python path
high_freq_sim_dir = cross_sim_parent / 'High_freq_sim'
sys.path.append(str(high_freq_sim_dir))

# Add the simulator directory to the Python path
simulator_dir = cross_sim_parent / 'simulator'
sys.path.append(str(simulator_dir))

# Now we can import all necessary modules
from high_freq_sim import HighFreqSimCore
from cores.low_freq_sim_core import LowFreqSimCore
from parameters.crosssim_parameters import CrossSimParameters

import tensorflow as tf
import numpy as np
from typing import Tuple

class OctaveConvIntegration:
    def __init__(self, params: CrossSimParameters, alpha: float = 0.5):
        """
        Initialize OctaveConvIntegration.
        
        Args:
            params (CrossSimParameters): Parameters for the simulation
            alpha (float): Ratio of low-frequency channels (default: 0.5)
        """
        self.params = params
        self.alpha = alpha
        self.low_freq_core = LowFreqSimCore(params)
        self.high_freq_core = HighFreqSimCore(params)

    def split_frequencies(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Split input tensor into high and low frequency components.
        
        Args:
            x (tf.Tensor): Input tensor
        
        Returns:
            Tuple[tf.Tensor, tf.Tensor]: High and low frequency tensors
        """
        channels = x.shape[-1]
        low_channels = int(channels * self.alpha)
        
        x_high = x[..., :channels-low_channels]
        x_low = x[..., channels-low_channels:]
        
        return x_high, x_low

    def upsample(self, x: tf.Tensor) -> tf.Tensor:
        """
        Upsample the input tensor.
        
        Args:
            x (tf.Tensor): Input tensor
        
        Returns:
            tf.Tensor: Upsampled tensor
        """
        return tf.image.resize(x, tf.shape(x)[1:3] * 2, method='nearest')

    def combine_frequencies(self, x_high: tf.Tensor, x_low: tf.Tensor) -> tf.Tensor:
        """
        Combine high and low frequency outputs.
        
        Args:
            x_high (tf.Tensor): High-frequency tensor
            x_low (tf.Tensor): Low-frequency tensor
        
        Returns:
            tf.Tensor: Combined tensor
        """
        x_low_upsampled = self.upsample(x_low)
        return tf.concat([x_high, x_low_upsampled], axis=-1)

    def process(self, x: tf.Tensor) -> tf.Tensor:
        """
        Process input tensor using octave convolution.
        
        Args:
            x (tf.Tensor): Input tensor
        
        Returns:
            tf.Tensor: Processed tensor
        """
        x_high, x_low = self.split_frequencies(x)
        x_high_processed = self.high_freq_core.process(x_high)
        x_low_processed = self.low_freq_core.process(x_low)
        return self.combine_frequencies(x_high_processed, x_low_processed)

    def train(self, x_train, y_train, x_val, y_val, epochs=1, batch_size=32):
        """
        Train both high and low frequency models for a single epoch.
        
        Args:
            x_train (tf.Tensor): Training data
            y_train (tf.Tensor): Training labels
            x_val (tf.Tensor): Validation data
            y_val (tf.Tensor): Validation labels
            epochs (int): Number of epochs (default: 1)
            batch_size (int): Batch size for training
        
        Returns:
            tuple: Training history for high and low frequency models
        """
        x_train_high, x_train_low = self.split_frequencies(x_train)
        x_val_high, x_val_low = self.split_frequencies(x_val)

        high_freq_history = self.high_freq_core.train(
            x_train_high, y_train,
            x_val_high, y_val,
            epochs=epochs,
            batch_size=batch_size
        )

        low_freq_history = self.low_freq_core.train(
            x_train_low, y_train,
            x_val_low, y_val,
            epochs=epochs,
            batch_size=batch_size
        )

        return high_freq_history, low_freq_history

if __name__ == "__main__":
    params = CrossSimParameters()
    octave_conv = OctaveConvIntegration(params, alpha=0.25)
    
    input_tensor = tf.random.normal([1000, 32, 32, 64])
    labels = tf.random.uniform([1000], minval=0, maxval=10, dtype=tf.int32)

    split = 800
    x_train, x_val = input_tensor[:split], input_tensor[split:]
    y_train, y_val = labels[:split], labels[split:]
    
    high_freq_history, low_freq_history = octave_conv.train(x_train, y_train, x_val, y_val, epochs=1)
    
    print("High Frequency Training completed. Final loss:", high_freq_history.history['loss'][-1])
    print("Low Frequency Training completed. Final loss:", low_freq_history.history['loss'][-1])
    
    output = octave_conv.process(input_tensor[:1])
    
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)