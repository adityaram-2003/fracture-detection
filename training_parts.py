import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Load images to build and train the model
def load_path(path):
    """
    Load X-ray dataset
    """
    dataset = []
    for folder in os.listdir(path):
        folder = os.path.join(path, folder)
        if os.path.isdir(folder):
            for body in os.listdir(folder):
                path_p = os.path.join(folder, body)
                for id_p in os.listdir(path_p):
                    patient_id = id_p
                    path_id = os.path.join(path_p, id_p)
                    for lab in os.listdir(path_id):
                        if lab.split('_')[-1] == 'positive':
                            label = 'fractured'
                        elif lab.split('_')[-1] == 'negative':
                            label = 'Not Fractured'
                        path_l = os.path.join(path_id, lab)
                        for img in os.listdir(path_l):
                            img_path = os.path.join(path_l, img)
                            dataset.append(
                                {
                                    'label': body,
                                    'image_path': img_path
                                }
                            )
    return dataset

# Load data from path
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(THIS_FOLDER, 'Dataset')
data = load_path(image_dir)
labels = []
filepaths = []

# Add labels for DataFrame for each category: 0-Elbow, 1-Hand, 2-Shoulder
Labels = ["Elbow", "Hand", "Shoulder"]
for row in data:
    labels.append(row['label'])
    filepaths.append(row['image_path'])

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

images = pd.concat([filepaths, labels], axis=1)

# Split dataset: 10% test, 90% train (90% train will split to 20% validation and 80% train)
train_df, test_df = train_test_split(images, train_size=0.9, shuffle=True, random_state=1)

# Image generators for training, validation, and testing
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    validation_split=0.2)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=64,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=64,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

# Load pre-trained ResNet50 model
pretrained_model = tf.keras.applications.resnet50.ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg')

# Freeze layers to use pre-trained weights
pretrained_model.trainable = False

# Add custom dense layers for classification
inputs = pretrained_model.input
x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dense(50, activation='relu')(x)
outputs = tf.keras.layers.Dense(len(Labels), activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(train_images, validation_data=val_images, epochs=25, callbacks=[callbacks])

# Save the trained model
model.save(os.path.join(THIS_FOLDER, "weights/ResNet50_BodyParts.h5"))

# Evaluate the model on test data
results = model.evaluate(test_images, verbose=0)
print("Test Loss:", results[0])
print("Test Accuracy:", results[1])

# Plot training history (accuracy and loss)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(THIS_FOLDER, "plots/accuracy_plot.png"))
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(THIS_FOLDER, "plots/loss_plot.png"))
plt.show()
