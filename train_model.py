import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# load data
with open("clean_data/images.p", "rb") as f:
    X = pickle.load(f)

with open("clean_data/labels.p", "rb") as f:
    y = pickle.load(f)

# convert labels to numbers
unique_labels = np.unique(y)
label_dict = {name: i for i, name in enumerate(unique_labels)}
y = np.array([label_dict[name] for name in y])

# reshape and normalize
X = X.reshape(-1, 100, 100, 1)
X = X / 255.0

# model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100,100,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(unique_labels), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# train
model.fit(X, y, epochs=5, batch_size=16)

# save model
model.save("final_model.h5")

print("✅ final_model.h5 created")
