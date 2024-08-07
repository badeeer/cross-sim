# simulator/custom_inference.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
import os

def load_cifar10_test():
    (_, _), (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype('float32') / 255
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

def infer(model, test_data):
    loss, accuracy = model.evaluate(test_data, verbose=0)
    print(f'Accuracy: {accuracy * 100:.2f}%')

def create_resnet50_model(input_shape=(32, 32, 3), num_classes=10):
    base_model = tf.keras.applications.ResNet50(
        include_top=False, weights=None, input_shape=input_shape
    )
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    return model

if __name__ == "__main__":
    print("Loading CIFAR-10 test dataset...")
    cifar10_test = load_cifar10_test()
    
    model_weights_path = 'best_model.keras'
    
    if not os.path.exists(model_weights_path):
        print(f"Model weights file not found: {model_weights_path}")
    else:
        try:
            print(f"\nLoading model architecture and weights from {model_weights_path}...")
            model = create_resnet50_model()
            model.load_weights(model_weights_path)
            infer(model, cifar10_test)
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An error occurred while processing the model: {e}")
