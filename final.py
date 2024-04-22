import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Load data
X_train = np.load("C:/Users/admin/Downloads/X_train.npy")
Y_train = np.load("C:/Users/admin/Downloads/Y_train.npy")
X_test = np.load("C:/Users/admin/Downloads/X_test.npy")

# Preprocess data
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

num_classes = 2  # Binary classification: benign or malignant
Y_train = np.eye(num_classes)[Y_train]  # One-hot encoding

# Build the model
model = Sequential()

# Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten layer
model.add(Flatten())

# Dense layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile
model.compile(optimizer='adagrad', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, Y_train, epochs=10, batch_size=64)

# Make predictions
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

np.savetxt("C:/Users/admin/Downloads/rishi_jain.csv", np.column_stack((predicted_labels, X_test.reshape(X_test.shape[0], -1))), delimiter=",")

# Display sample images.
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[i])
    plt.title("Predicted: Benign" if predicted_labels[i] == 0 else "Predicted: Malignant")
    plt.axis('off')
plt.show()