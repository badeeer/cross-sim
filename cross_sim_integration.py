import json
import sys
import os
import tensorflow as tf

# Add the current directory to the Python path to import custom_inference
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import custom_inference as cusifn  # Import custom_inference functions

def load_cross_sim_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def save_cross_sim_config(config_path, config):
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def apply_cross_sim_config(model, cross_sim_config):
    # Modify the model's layers to use the analog cores
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer.kernel_initializer = 'zeros'
            layer.bias_initializer = 'zeros'
            layer.use_bias = cross_sim_config['convolution']['bias_row']
            # Add any other necessary modifications to the layer
    return model

def main():
    # Load Cross-SIM configuration
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'ISAAC.json')
    cross_sim_config = load_cross_sim_config(config_path)

    # Load the custom model and CIFAR-10 test dataset
    model_config_path = os.path.join(os.path.dirname(__file__), 'best_model', 'config.json')
    model_weights_path = os.path.join(os.path.dirname(__file__), 'best_model', 'model.weights.h5')

    try:
        model = cusifn.load_custom_model(model_config_path, model_weights_path)
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return
    
    cifar10_test = cusifn.load_cifar10_test()

    # Apply Cross-SIM configurations to the model
    model = apply_cross_sim_config(model, cross_sim_config)

    # Run inference with Cross-SIM configuration
    print("Running inference on the model...")
    accuracy = cusifn.infer(model, cifar10_test)

    # Ensure the directory exists
    simulator_dir = os.path.join(os.path.dirname(__file__), 'simulator')
    if not os.path.exists(simulator_dir):
        os.makedirs(simulator_dir)

    # Update and save configuration with results
    results_path = os.path.join(simulator_dir, 'inference_results.json')
    
    try:
        with open(results_path, 'r') as results_file:
            results = json.load(results_file)
    except FileNotFoundError:
        print(f"Inference results file not found at {results_path}.")
        return
    except json.JSONDecodeError:
        print("Error decoding inference results file.")
        return

    if 'accuracy' in results:
        cross_sim_config['results'] = {
            'accuracy': results['accuracy'] * 100,
            'model': 'ResNet-50',
            'dataset': 'CIFAR-10'
        }
        # Save updated configuration
        save_cross_sim_config(config_path, cross_sim_config)
        print(f"Results saved to {config_path}")
        print(f"Accuracy: {results['accuracy'] * 100:.2f}%")
    else:
        print("Unable to obtain accuracy result.")

if __name__ == "__main__":
    main()