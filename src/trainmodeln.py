from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actio:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(
                os.path.join(
                    NUM_PATH, action, str(sequence), "{}.npy".format(frame_num)
                ),
                allow_pickle=True,
            )
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])


# Convert lists in sequences to numpy arrays
sequences_arrays = [np.array(seq) for seq in sequences]

# Determine the shapes of all elements in sequences
shapes = [seq.shape for seq in sequences_arrays]

# Find the most common shape
most_common_shape = max(set(shapes), key=shapes.count)

# Identify elements with shapes different from the most common shape
indices_with_different_shape = [
    i for i, shape in enumerate(shapes) if shape != most_common_shape
]

# Remove sequences with different shapes from X and their corresponding labels from y
X_clean = [
    sequences_arrays[i]
    for i in range(len(sequences_arrays))
    if i not in indices_with_different_shape
]
y_clean = [
    labels[i] for i in range(len(labels)) if i not in indices_with_different_shape
]


# Convert sequences_clean to a numpy array
X = np.array(X_clean)
y = to_categorical(y_clean).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join("Logs")
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(1, 63)))
model.add(LSTM(128, return_sequences=True, activation="relu"))
model.add(LSTM(64, return_sequences=False, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(actions.shape[0], activation="softmax"))
res = [0.7, 0.2, 0.1]

model.compile(
    optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
)
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])
model.summary()

model_json = model.to_json()
with open("nmodel.json", "w") as json_file:
    json_file.write(model_json)
model.save("nmodel.h5")
print("saved")
