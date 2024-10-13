import tensorflow as tf
from typing import Tuple
from simulator.parameters.crosssim_parameters import CrossSimParameters
from low_freq_sim_core import LowFreqSimCore
from High_freq_sim.high_freq_sim import High_freq_sim

class IntraFrequencyOps:
    def __init__(self, params: CrossSimParameters, alpha: float = 0.5, upsampling_scale: int = 2, downsampling_scale: int = 2):
        """
        Initialize IntraFrequencyOps.

        Args:
            params (CrossSimParameters): Parameters for the simulation.
            alpha (float): Ratio of low-frequency channels (default: 0.5).
            upsampling_scale (int): Scale factor for upsampling (default: 2).
            downsampling_scale (int): Scale factor for downsampling (default: 2).
        """
        self.params = params
        self.alpha = alpha
        self.upsampling_scale = upsampling_scale
        self.downsampling_scale = downsampling_scale
        self.low_freq_core = LowFreqSimCore(params)
        self.high_freq_core = High_freq_sim(params)

    def split_frequencies(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Split input tensor into high and low frequency components.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: High and low frequency tensors.
        """
        channels = x.shape[-1]
        low_channels = int(channels * self.alpha)

        x_high = x[..., :channels-low_channels]
        x_low = x[..., channels-low_channels:]

        return x_high, x_low

    def upsample(self, x: tf.Tensor) -> tf.Tensor:
        """
        Upsample the input tensor by a given scale factor.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Upsampled tensor.
        """
        new_size = tf.shape(x)[1:3] * self.upsampling_scale
        return tf.image.resize(x, new_size, method='nearest')

    def downsample(self, x: tf.Tensor) -> tf.Tensor:
        """
        Downsample the input tensor by a given scale factor.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Downsampled tensor.
        """
        new_size = tf.shape(x)[1:3] // self.downsampling_scale
        return tf.image.resize(x, new_size, method='bilinear')

    def process_low_freq(self, x: tf.Tensor) -> tf.Tensor:
        """
        Process low-frequency component using Cross-SIM.

        Args:
            x (tf.Tensor): Low-frequency tensor.

        Returns:
            tf.Tensor: Processed tensor.
        """
        return self.low_freq_core.process(x)

    def process_high_freq(self, x: tf.Tensor) -> tf.Tensor:
        """
        Process high-frequency component using CPU/GPU.

        Args:
            x (tf.Tensor): High-frequency tensor.

        Returns:
            tf.Tensor: Processed tensor.
        """
        return self.high_freq_core.process(x)

    def combine_frequencies(self, x_high: tf.Tensor, x_low: tf.Tensor) -> tf.Tensor:
        """
        Combine high and low frequency outputs.

        Args:
            x_high (tf.Tensor): High-frequency tensor.
            x_low (tf.Tensor): Low-frequency tensor.

        Returns:
            tf.Tensor: Combined tensor.
        """
        x_low_upsampled = self.upsample(x_low)
        return tf.concat([x_high, x_low_upsampled], axis=-1)

    def process(self, x: tf.Tensor) -> tf.Tensor:
        """
        Process input tensor using octave convolution.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Processed tensor.
        """
        x_high, x_low = self.split_frequencies(x)
        x_low_processed = self.process_low_freq(x_low)
        x_high_processed = self.process_high_freq(x_high)
        return self.combine_frequencies(x_high_processed, x_low_processed)

# Example usage
if __name__ == "__main__":
    # Create sample parameters (adjust as needed based on CrossSimParameters structure)
    params = CrossSimParameters()
    
    # Create IntraFrequencyOps instance
    intra_freq_ops = IntraFrequencyOps(params, alpha=0.25)
    
    # Create a sample input tensor
    input_tensor = tf.random.normal([1, 32, 32, 64])
    
    # Process the input using octave convolution
    output = intra_freq_ops.process(input_tensor)
    
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)
