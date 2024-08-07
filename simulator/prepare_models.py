
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, UpSampling2D, Flatten, Input
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np
import math

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Perform the train-validation split
validation_split = 0.1
num_validation_samples = int(validation_split * len(x_train))
x_train, x_val = x_train[:-num_validation_samples], x_train[-num_validation_samples:]
y_train, y_val = y_train[:-num_validation_samples], y_train[-num_validation_samples:]

# Data augmentation function
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, label

# Create TensorFlow datasets
batch_size = 64

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).map(augment).batch(batch_size).repeat()

validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
validation_dataset = validation_dataset.batch(batch_size).repeat()

steps_per_epoch = math.ceil(len(x_train) / batch_size)
validation_steps = math.ceil(len(x_val) / batch_size)

print(f"Total training samples: {len(x_train)}")
print(f"Total validation samples: {len(x_val)}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

def feature_extractor(inputs):
    feature_extractor = ResNet50V2(input_shape=(224, 224, 3),
                                   include_top=False,
                                   weights='imagenet')(inputs)
    return feature_extractor

def classifier(inputs):
    x = GlobalAveragePooling2D()(inputs)
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(10, activation="softmax", name="classification")(x)
    return x

def final_model(inputs):
    resize = UpSampling2D(size=(7, 7))(inputs)
    resnet_feature_extractor = feature_extractor(resize)
    classification_output = classifier(resnet_feature_extractor)
    return classification_output

def define_compile_model():
    inputs = Input(shape=(32, 32, 3))
    classification_output = final_model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=classification_output)
    
    model.compile(optimizer=SGD(), 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

model = define_compile_model()
model.summary()

# Callbacks
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=5,
    callbacks=[checkpoint, reduce_lr, early_stopping],
    verbose=1
)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Fine-tuning
for layer in model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_fine = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=5,
    callbacks=[checkpoint, reduce_lr, early_stopping],
    verbose=1
)

# Final evaluation
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Final test accuracy after fine-tuning: {test_acc:.4f}")