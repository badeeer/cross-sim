
import tensorflow as tf
from prepare_models import define_compile_model, augment

class OctaveResNet(tf.keras.Model):
    def __init__(self, original_model, high_freq_processor, conv1, low_freq_processor):
        super(OctaveResNet, self).__init__()
        self.original_model = original_model
        self.high_freq_processor = high_freq_processor
        self.conv1 = conv1
        self.low_freq_processor = low_freq_processor

    def call(self, inputs):
        # Apply initial convolution
        x = self.conv1(inputs)

        # Split into high and low frequency components
        high_freq = self.high_freq_processor(x)
        low_freq = tf.nn.avg_pool2d(x, ksize=2, strides=2, padding='SAME')

        # Process low frequency
        low_freq = self.low_freq_processor(low_freq)

        # Upsample low frequency to match high frequency
        low_freq_upsampled = tf.image.resize(low_freq, tf.shape(high_freq)[1:3])

        # Combine high and low frequency components
        combined = tf.concat([high_freq, low_freq_upsampled], axis=-1)

        # Apply the rest of the original model
        output = self.original_model(combined)

        return output

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

def split_data(x_train, y_train, validation_split=0.1):
    num_val_samples = int(validation_split * len(x_train))
    return (x_train[:-num_val_samples], y_train[:-num_val_samples], 
            x_train[-num_val_samples:], y_train[-num_val_samples:])

def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train, y_train, x_val, y_val = split_data(x_train, y_train)
    
    model = define_compile_model()

    batch_size = 4
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(1024).map(augment).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

    # Fit model for exactly 5 epochs
    model.fit(train_dataset, validation_data=val_dataset, epochs=5, verbose=1)

    # Evaluate final accuracy on the test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()