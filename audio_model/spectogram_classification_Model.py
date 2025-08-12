import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB1
import random
import matplotlib.pyplot as plt

# --- 1. Load Metadata ---
metadata_path = '/kaggle/input/met-223/op.xlsx'
df = pd.read_excel(metadata_path)
df = df.drop_duplicates().drop_duplicates(subset=['image_file_name'])

# --- 2. Prepare labels ---
le = LabelEncoder()
df['Regime_enc'] = le.fit_transform(df['Regime'])
num_classes = len(le.classes_)

# --- 3. Load spectrogram images ---
SPEC_SIZE = (64, 128)

def load_spec_image(path):
    img = load_img(path, target_size=SPEC_SIZE, color_mode='rgb')
    arr = img_to_array(img) / 255.0
    return arr

# --- 4. Spectrogram Augmentation ---
def augment_spec(arr, time_mask_width=16, freq_mask_height=8, noise_std=0.02):
    spec = arr.copy()
    t = spec.shape[1]
    t0 = random.randint(0, t - time_mask_width)
    spec[:, t0:t0 + time_mask_width, :] = 0
    f = spec.shape[0]
    f0 = random.randint(0, f - freq_mask_height)
    spec[f0:f0 + freq_mask_height, :, :] = 0
    noise = np.random.normal(loc=0.0, scale=noise_std, size=spec.shape)
    spec = np.clip(spec + noise, 0.0, 1.0)
    return spec

specs = []
labels = []
for _, row in df.iterrows():
    base = load_spec_image(row['image_file_name'])
    specs.append(base)
    labels.append(row['Regime_enc'])
    for _ in range(2):  # 2 augmentations per image
        specs.append(augment_spec(base))
        labels.append(row['Regime_enc'])

X_spec = np.array(specs)
y = np.array(labels)

# --- 5. Tabular features ---
tab_features = df[['T', 'We']].values
X_tab = np.repeat(tab_features, len(specs) // len(tab_features), axis=0)
X_tab = StandardScaler().fit_transform(X_tab)
y_cat = to_categorical(y, num_classes=num_classes)

# --- 6. Split Data ---
X_spec_trainval, X_spec_test, X_tab_trainval, X_tab_test, y_trainval, y_test = train_test_split(
    X_spec, X_tab, y_cat, test_size=0.2, random_state=42, stratify=y_cat)
X_spec_train, X_spec_val, X_tab_train, X_tab_val, y_train, y_val = train_test_split(
    X_spec_trainval, X_tab_trainval, y_trainval,
    test_size=0.1, random_state=42, stratify=y_trainval)

# --- 7. Build CNN Model with Transfer Learning (EfficientNetB0) ---
base_cnn = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(64, 128, 3))
base_cnn.trainable = False  # Freeze base model initially

spec_input = Input(shape=(64, 128, 3), name='spec_input')
x = base_cnn(spec_input, training=False)  # Important for transfer learning
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)

tab_input = Input(shape=(2,), name='tab_input')
y_tab = Dense(32, activation='relu')(tab_input)
y_tab = BatchNormalization()(y_tab)
y_tab = Dense(16, activation='relu')(y_tab)

combined = Concatenate()([x, y_tab])
zz = Dense(64, activation='relu')(combined)
zz = Dropout(0.3)(zz)
output = Dense(num_classes, activation='softmax')(zz)

model = Model(inputs=[spec_input, tab_input], outputs=output)

# --- 8. Inverse-Class-Weighting ---
total = len(y)
class_counts = np.bincount(y)
class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}

# --- 9. Compile ---
optimizer = Adam(learning_rate=1e-3)


model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 10. Train Model ---
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

history = model.fit(
    [X_spec_train, X_tab_train], y_train,
    validation_data=([X_spec_val, X_tab_val], y_val),
    epochs=100,
    batch_size=16,
    class_weight=class_weights,
    callbacks=[early_stop, lr_reducer]
)

# --- 11. Evaluate ---
for split, (Xs, Xt, ys) in zip(['Train', 'Val', 'Test'],
    [(X_spec_train, X_tab_train, y_train),
     (X_spec_val, X_tab_val, y_val),
     (X_spec_test, X_tab_test, y_test)]):
    loss, acc = model.evaluate([Xs, Xt], ys, verbose=0)
    print(f"{split} Accuracy: {acc*100:.2f}%")

# --- 12. Plot Loss Curves ---
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()
