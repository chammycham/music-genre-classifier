#%% Early stopping 
#%%
import os
import random
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Define functions to process audio and Mel spectrograms
def find_mel_spectrogram(path, fft_size=2048, hop_size=None, window_size=None):
    try:
        signal, sample_rate = sf.read(path)
    except RuntimeError as e:
        print(f"Error reading {path}: {e}")
        return None

    if not window_size:
        window_size = fft_size

    if not hop_size:
        hop_size = window_size // 4

    mel = librosa.feature.melspectrogram(
        y=signal, 
        sr=sample_rate, 
        hop_length=hop_size,
        n_fft=fft_size
    )
    
    spectrogram = np.abs(mel)
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
    return spectrogram_db

def pad_or_truncate_spectrogram(spectrogram, target_shape=(128, 1293)):
    if spectrogram is None:
        return None
    target_height, target_width = target_shape
    height, width = spectrogram.shape

    # Pad or truncate height
    if height < target_height:
        pad_height = target_height - height
        spectrogram = np.pad(spectrogram, ((0, pad_height), (0, 0)), mode='constant')
    elif height > target_height:
        spectrogram = spectrogram[:target_height, :]

    # Pad or truncate width
    if width < target_width:
        pad_width = target_width - width
        spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    elif width > target_width:
        spectrogram = spectrogram[:, :target_width]

    return spectrogram

genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
base_path = #path to GTZN dataset here
target_shape = (128, 1293)
genre_map = {genre: idx for idx, genre in enumerate(genres)}

data = []
for genre in genres:
    genre_path = os.path.join(base_path, genre)
    files = [f for f in os.listdir(genre_path) if f.endswith(".wav")]
    selected_files = random.sample(files, 50)

    for file in selected_files:
        file_path = os.path.join(genre_path, file)
        mel_spectrogram = find_mel_spectrogram(file_path)
        mel_spectrogram_resized = pad_or_truncate_spectrogram(mel_spectrogram, target_shape)
        genre_index = genre_map[genre]
        data.append((genre_index, mel_spectrogram_resized))

df = pd.DataFrame(data, columns=["genre", "spectrogram"])
df["genre"] = df["genre"].apply(lambda x: to_categorical(x, num_classes=len(genres)))

shapes = [spectrogram.shape for spectrogram in df["spectrogram"]]
assert len(set(shapes)) == 1, "Spectrograms are not uniform in shape."

X = np.array(df["spectrogram"].tolist())
X = (X - np.min(X)) / (np.max(X) - np.min(X))  # Normalize to [0, 1]
X = X[..., np.newaxis]  # Add channel dimension
y = np.array(df["genre"].tolist())

# Split into train, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

# Verify shapes
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
#%%
input_shape = (128, 1293, 1)  # Spectrogram dimensions + channel
num_classes = len(genres)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    #Dropout(0.2),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    #Dropout(0.2),

    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    #Dropout(0.2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
#%%
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    verbose=1
)

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {test_accuracy:.2f}")
#%%
import matplotlib.pyplot as plt

plt.figure(dpi=600)
plt.xlim(0, 20)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(dpi=600)
plt.xlim(0, 20)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class indices
y_true = np.argmax(y_test, axis=1)        # Convert one-hot encoded labels to class indices

cm = confusion_matrix(y_true, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genres)
plt.figure(dpi=600)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.show()
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred, target_names=genres))
# %%
