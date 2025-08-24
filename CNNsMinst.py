# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 14:12:49 2025

@author: AlvaroRivera-Eraso
"""

##############################################################################
# NOTE: The code in this project is presented in separate sections, as it was
# executed incrementally at different stages of the workflow. 
# To reproduce the results from scratch, please follow the sequence outlined
# in the report, running each code block in the order explicitly described.
##############################################################################


#%%
import os, base64
import zipfile
from sklearn.model_selection import KFold
import optuna
import numpy as np
import random
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from datetime import datetime
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam
from contextlib import redirect_stdout
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd


#%%
# ========== Local Path Setup ==========

zip_path = r"C:/Users/AlvaroRivera-Eraso/Documents/HULL/d.zip"
extract_to = r"C:/Users/AlvaroRivera-Eraso/Documents/HULL/triple_minst_data"

# Unzip if not already unzipped
if not os.path.exists(extract_to):
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


#%%

def load_data(root_dir, image_size=(84, 84)):
    images = []
    labels = []

    label_folders = sorted([
        f for f in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, f)) and f.isdigit()
    ])

    for label_folder in tqdm(label_folders, desc="Load images", unit="folder"):
        folder_path = os.path.join(root_dir, label_folder)
        label_digits = [int(d) for d in label_folder]
        image_files = sorted(os.listdir(folder_path))

        for img_file in tqdm(image_files, desc=f"Loading {label_folder}", leave=False):
            img_path = os.path.join(folder_path, img_file)
            try:
                img = load_img(img_path, color_mode='grayscale', target_size=image_size)
                img_array = img_to_array(img) / 255.0  # normalize to [0,1]
                images.append(img_array)
                labels.append(label_digits)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    X = np.array(images).astype('float32')
    Y = np.array(labels).astype('int32')
    return X, Y

def preprocess_data(path, num_classes=10):
    X, Y = load_data(path)
    y1 = to_categorical(Y[:, 0], num_classes)
    y2 = to_categorical(Y[:, 1], num_classes)
    y3 = to_categorical(Y[:, 2], num_classes)
    return X, y1, y2, y3


#%%

#base_path = r"C:/Users/AlvaroRivera-Eraso/Documents/HULL/triple_minst_data/triple_mnist"

#x_train, y1_train, y2_train, y3_train = preprocess_data(os.path.join(base_path, "train"))
#x_val,   y1_val,   y2_val,   y3_val   = preprocess_data(os.path.join(base_path, "val"))
#x_test,  y1_test,  y2_test,  y3_test  = preprocess_data(os.path.join(base_path, "test"))




#%%

def count_images_per_label(root_dir):
    label_counts = {}

    label_folders = sorted([
        f for f in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, f)) and f.isdigit()
    ])

    for label_folder in label_folders:
        folder_path = os.path.join(root_dir, label_folder)
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        label_counts[label_folder] = len(image_files)

    return label_counts

def plot_label_distribution(label_counts, title="Label Distribution"):
    labels = list(label_counts.keys())
    counts = list(label_counts.values())

    plt.figure(figsize=(14, 6))
    plt.bar(labels, counts, color='skyblue', edgecolor='black')
    plt.xticks(rotation=90)
    plt.xlabel("Label (Digits)")
    plt.ylabel("Number of Images")
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

base_path = r"C:/Users/AlvaroRivera-Eraso/Documents/HULL/triple_minst_data/triple_mnist"

for split in ['train', 'val', 'test']:
    print(f"\nAnalyzing {split}...")
    split_path = os.path.join(base_path, split)
    label_counts = count_images_per_label(split_path)
    plot_label_distribution(label_counts, title=f"{split.capitalize()} Label Distribution")


#%%

# ========= SUBSET LOADER =========
def load_subset_data(root_dir, fraction=0.3, image_size=(84, 84), num_classes=10, seed=42):
    images, labels = [], []

    label_folders = sorted([
        f for f in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, f)) and f.isdigit()
    ])

    for label_folder in tqdm(label_folders, desc="Loading images"):
        folder_path = os.path.join(root_dir, label_folder)
        label_digits = [int(d) for d in label_folder]
        image_files = sorted(os.listdir(folder_path))
        np.random.seed(seed)
        image_files = np.random.choice(image_files, int(len(image_files) * fraction), replace=False)

        for img_file in image_files:
            try:
                img_path = os.path.join(folder_path, img_file)
                img = load_img(img_path, color_mode='grayscale', target_size=image_size)
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label_digits)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    X = np.array(images).astype('float32')
    Y = np.array(labels).astype('int32')

    # Shuffle before returning
    X, Y = shuffle(X, Y, random_state=seed)

    y1 = to_categorical(Y[:, 0], num_classes)
    y2 = to_categorical(Y[:, 1], num_classes)
    y3 = to_categorical(Y[:, 2], num_classes)
    return X, y1, y2, y3


#%%

# ========= CNN MODEL =========
def build_triple_digit_model(input_shape=(84, 84, 1), num_classes=10, dropout_rate=0.3):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    out1 = Dense(num_classes, activation='softmax', name='digit1')(x)
    out2 = Dense(num_classes, activation='softmax', name='digit2')(x)
    out3 = Dense(num_classes, activation='softmax', name='digit3')(x)

    model = Model(inputs=inputs, outputs=[out1, out2, out3])
    return model

# ========= EXECUTION =========
base_path = r"C:/Users/AlvaroRivera-Eraso/Documents/HULL/triple_minst_data/triple_mnist"
train_path = os.path.join(base_path, "train")
val_path = os.path.join(base_path, "val")

# Load 30% subset for quick test
x_train, y1_train, y2_train, y3_train = load_subset_data(train_path, fraction=0.3)
x_val,   y1_val,   y2_val,   y3_val   = load_subset_data(val_path, fraction=0.3)

# Build model
model = build_triple_digit_model()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', 'accuracy', 'accuracy']
)


# Train model
history = model.fit(
    x_train, [y1_train, y2_train, y3_train],
    validation_data=(x_val, [y1_val, y2_val, y3_val]),
    epochs=10,
    batch_size=32
)





#%%

test_path = os.path.join(base_path, "test")
x_test, y1_test, y2_test, y3_test = load_subset_data(test_path, fraction=1.0)

test_results = model.evaluate(x_test, [y1_test, y2_test, y3_test])
print("Test loss and accuracy per digit:", test_results)

# Evaluate on validation
val_loss = model.evaluate(x_val, [y1_val, y2_val, y3_val])
print("Validation Loss & Acc:", val_loss)

# Plot training history
plt.plot(history.history['digit1_accuracy'], label='Digit1')
plt.plot(history.history['digit2_accuracy'], label='Digit2')
plt.plot(history.history['digit3_accuracy'], label='Digit3')
plt.title("Training Accuracy per Digit")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


#%%
model.save("triple_digit_model_30pct.h5")

#%%

# Load subset
x_train, y1_train, y2_train, y3_train = load_subset_data(train_path, fraction=0.3)
x_val,   y1_val,   y2_val,   y3_val   = load_subset_data(val_path, fraction=0.3)

# One-hot to labels for F1 calculation
y1_val_lbl = y1_val.argmax(axis=1)
y2_val_lbl = y2_val.argmax(axis=1)
y3_val_lbl = y3_val.argmax(axis=1)

def build_model(lr, dropout):
    inputs = Input(shape=(84, 84, 1))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    
    out1 = Dense(10, activation='softmax', name='digit1')(x)
    out2 = Dense(10, activation='softmax', name='digit2')(x)
    out3 = Dense(10, activation='softmax', name='digit3')(x)
    
    model = Model(inputs=inputs, outputs=[out1, out2, out3])
    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy')
    return model

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    dropout = trial.suggest_uniform('dropout', 0.2, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    model = build_model(lr, dropout)
    model.fit(x_train, [y1_train, y2_train, y3_train], batch_size=batch_size, epochs=5, verbose=0)

    preds = model.predict(x_val)
    preds1, preds2, preds3 = preds[0].argmax(axis=1), preds[1].argmax(axis=1), preds[2].argmax(axis=1)

    f1_1 = f1_score(y1_val_lbl, preds1, average='macro')
    f1_2 = f1_score(y2_val_lbl, preds2, average='macro')
    f1_3 = f1_score(y3_val_lbl, preds3, average='macro')

    avg_f1 = (f1_1 + f1_2 + f1_3) / 3
    return avg_f1

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print(" Best trial:")
print(study.best_trial.params)
print(f"Best macro F1: {study.best_value:.4f}")


#%%

x_train, y1_train, y2_train, y3_train = load_subset_data(train_path, fraction=1.0)
x_val,   y1_val,   y2_val,   y3_val   = load_subset_data(val_path,   fraction=1.0)


best_lr       = 4.585767605125491e-04   # from Optuna
best_dropout  = 0.32171964891544136
best_batch    = 16

model = build_triple_digit_model(dropout_rate=best_dropout)
model.compile(
    optimizer=Adam(learning_rate=best_lr),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'accuracy', 'accuracy']
)


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early   = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
chkpt   = ModelCheckpoint('best_full_model.h5', save_best_only=True, monitor='val_loss')

history = model.fit(
    x_train, [y1_train, y2_train, y3_train],
    validation_data=(x_val, [y1_val, y2_val, y3_val]),
    epochs=30,
    batch_size=best_batch,
    callbacks=[early, chkpt]
)

#%%
x_test, y1_test, y2_test, y3_test = load_subset_data(test_path, fraction=1.0)
model.load_weights('best_full_model.h5')   # if EarlyStopping triggered
test_metrics = model.evaluate(x_test, [y1_test, y2_test, y3_test])
print("Final test metrics:", test_metrics)

model.save("triple_digit_full_final.h5")

#%%
model.save("best_full_model.keras")


#%%
best_params = study.best_trial.params
# ==== SAVE ALL ARTIFACTS IN ONE SHOT ====
import os, json, pickle, yaml, datetime, joblib
import pandas as pd
from contextlib import redirect_stdout

ART_DIR = "artifacts"
os.makedirs(ART_DIR, exist_ok=True)

# 1️Model (Keras native format = weights+arch)
model_path = os.path.join(ART_DIR, "best_full_model.keras")
model.save(model_path)       # automatically the EarlyStopping best if you loaded weights

# 2️Model summary to .txt for quick reference
with open(os.path.join(ART_DIR, "best_full_model_summary.txt"), "w") as f:
    with redirect_stdout(f):
        model.summary()

# 3️Training history
with open(os.path.join(ART_DIR, "history.pkl"), "wb") as f:
    pickle.dump(history.history, f)
pd.DataFrame(history.history).to_csv(os.path.join(ART_DIR, "history.csv"), index=False)

# 4️Best hyper-parameters
with open(os.path.join(ART_DIR, "best_params.json"), "w") as f:
    json.dump(best_params, f, indent=2)

# 5️Optuna study (comment out if you didn’t keep `study`)
joblib.dump(study, os.path.join(ART_DIR, "study.pkl"))

# 6️Minimal run config for provenance
config = {
    "date": str(datetime.datetime.now()),
    "epochs_ran": len(history.history["loss"]),
    "batch_size": best_params["batch_size"],
    "train_path": train_path,
    "val_path": val_path,
    "test_path": test_path,
}
with open(os.path.join(ART_DIR, "config.yaml"), "w") as f:
    yaml.dump(config, f)

print(f" All artifacts saved to: {os.path.abspath(ART_DIR)}")


#%%
from tensorflow.keras.models import load_model
import pickle, pandas as pd, joblib, json, yaml
import os

# Point to your artifact folder
ART_DIR = r"C:\Users\AlvaroRivera-Eraso\Documents\HULL\artifacts"

# 1️Load the model
model_path = os.path.join(ART_DIR, "best_full_model.keras")
model = load_model(model_path, compile=False)

# 2️Reload hyperparameters and compile the model
with open(os.path.join(ART_DIR, "best_params.json")) as f:
    best_params = json.load(f)

model.compile(
    optimizer=Adam(learning_rate=best_params["lr"]),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'accuracy', 'accuracy']
)

# 3️Reload training histor
hist_path = os.path.join(ART_DIR, "history.pkl")
with open(hist_path, "rb") as f:
    hist = pickle.load(f)

pd.DataFrame(hist).plot(title="Training History")

# 4️Reload Optuna study (optional, only if you saved study.pkl)
study_path = os.path.join(ART_DIR, "study.pkl")
if os.path.exists(study_path):
    study = joblib.load(study_path)
    print("Optuna study reloaded. Best trial:", study.best_trial.params)
else:
    print("Optuna study.pkl not found in artifacts directory.")


#%%


# =========================
# 0) Imports & Reproducibility
# =========================

# ---- Set seeds for determinism ----
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# 1) Paths & unzip 
# =========================
ZIP_PATH = r"C:\Users\AlvaroRivera-Eraso\Documents\HULL\d.zip"
EXTRACT_TO = r"C:\Users\AlvaroRivera-Eraso\Documents\HULL\triple_minst_data"
BASE_PATH  = os.path.join(EXTRACT_TO, "triple_mnist")

ART_DIR    = r"C:\Users\AlvaroRivera-Eraso\Documents\HULL\artifacts"
os.makedirs(ART_DIR, exist_ok=True)

if not os.path.exists(BASE_PATH):
    os.makedirs(EXTRACT_TO, exist_ok=True)
    if os.path.exists(ZIP_PATH):
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_TO)
    else:
        raise FileNotFoundError(f"Dataset ZIP not found at: {ZIP_PATH}")

TRAIN_PATH = os.path.join(BASE_PATH, "train")
VAL_PATH   = os.path.join(BASE_PATH, "val")
TEST_PATH  = os.path.join(BASE_PATH, "test")

# =========================
# 2) Data loading utilities
# =========================
def list_digit_folders(root_dir):
    return sorted([
        f for f in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, f)) and f.isdigit()
    ])

def load_data(root_dir, image_size=(84,84)):
    images, labels = [], []
    label_folders = list_digit_folders(root_dir)
    for label_folder in tqdm(label_folders, desc=f"Load images from {os.path.basename(root_dir)}", unit="folder"):
        folder_path = os.path.join(root_dir, label_folder)
        label_digits = [int(d) for d in label_folder]
        image_files = sorted(os.listdir(folder_path))
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            if not img_file.lower().endswith(('.png','.jpg','.jpeg')):
                continue
            try:
                img = load_img(img_path, color_mode='grayscale', target_size=image_size)
                img_array = img_to_array(img) / 255.0  # normalize
                images.append(img_array)
                labels.append(label_digits)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    X = np.array(images, dtype='float32')              # (N,84,84,1)
    Y = np.array(labels, dtype='int32')                # (N,3)
    return X, Y

def load_subset_data(root_dir, fraction=0.3, image_size=(84,84), num_classes=10, seed=SEED):
    np.random.seed(seed)  # set once here (moved from inside the loop)
    images, labels = [], []
    label_folders = list_digit_folders(root_dir)
    for label_folder in tqdm(label_folders, desc=f"Subsetting {os.path.basename(root_dir)}"):
        folder_path = os.path.join(root_dir, label_folder)
        label_digits = [int(d) for d in label_folder]
        image_files = [f for f in sorted(os.listdir(folder_path)) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        k = max(1, int(len(image_files) * fraction))
        chosen = np.random.choice(image_files, k, replace=False)
        for img_file in chosen:
            try:
                img_path = os.path.join(folder_path, img_file)
                img = load_img(img_path, color_mode='grayscale', target_size=image_size)
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label_digits)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    X = np.array(images, dtype='float32')
    Y = np.array(labels, dtype='int32')
    X, Y = shuffle(X, Y, random_state=seed)

    y1 = to_categorical(Y[:,0], num_classes)
    y2 = to_categorical(Y[:,1], num_classes)
    y3 = to_categorical(Y[:,2], num_classes)
    return X, y1, y2, y3

def count_images_per_label(root_dir):
    counts = {}
    for label_folder in list_digit_folders(root_dir):
        folder_path = os.path.join(root_dir, label_folder)
        n = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        counts[label_folder] = n
    return counts

def save_label_distribution(root_dir, title, out_png):
    label_counts = count_images_per_label(root_dir)
    labels = list(label_counts.keys())
    counts = [label_counts[k] for k in labels]

    plt.figure(figsize=(14, 5))
    plt.bar(labels, counts, edgecolor='black')
    plt.xticks(rotation=90)
    plt.xlabel("3-digit label")
    plt.ylabel("Images")
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()
    return label_counts

# =========================
# 3) Model definition
# =========================
def build_triple_digit_model(input_shape=(84,84,1), num_classes=10, dropout_rate=0.3,
                             filters1=32, filters2=64, dense_units=128):
    inputs = Input(shape=input_shape)
    x = Conv2D(filters1, (3,3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(filters2, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    out1 = Dense(num_classes, activation='softmax', name='digit1')(x)
    out2 = Dense(num_classes, activation='softmax', name='digit2')(x)
    out3 = Dense(num_classes, activation='softmax', name='digit3')(x)
    model = Model(inputs=inputs, outputs=[out1, out2, out3])
    return model

# =========================
# 4) Visualizations: learning curves & confusion matrices
# =========================
def save_learning_curves(history, out_prefix):
    # Loss
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_prefix + "_loss.png", dpi=140)
    plt.close()

    # Accuracies per head
    for head in ['digit1', 'digit2', 'digit3']:
        tkey, vkey = f'{head}_accuracy', f'val_{head}_accuracy'
        if tkey in history.history and vkey in history.history:
            plt.figure()
            plt.plot(history.history[tkey], label=f'train_{head}_acc')
            plt.plot(history.history[vkey], label=f'val_{head}_acc')
            plt.title(f'Accuracy - {head}')
            plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
            plt.tight_layout()
            plt.savefig(out_prefix + f"_{head}_acc.png", dpi=140)
            plt.close()

def save_confusion_matrix(t, p, name, out_png):
    cm = confusion_matrix(t, p, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(f'Confusion Matrix - {name}')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return cm

# =========================
# 5) Dataset overview plots
# =========================
train_dist_png = os.path.join(ART_DIR, "train_label_distribution.png")
val_dist_png   = os.path.join(ART_DIR, "val_label_distribution.png")
test_dist_png  = os.path.join(ART_DIR, "test_label_distribution.png")

train_counts = save_label_distribution(TRAIN_PATH, "Train Label Distribution", train_dist_png)
val_counts   = save_label_distribution(VAL_PATH,   "Val Label Distribution",   val_dist_png)
test_counts  = save_label_distribution(TEST_PATH,  "Test Label Distribution",  test_dist_png)

# =========================
# 6) Hyperparameter tuning with Optuna (3-fold CV on TRAIN)
#     (To keep runtime reasonable, we tune on a subset fraction)
# =========================
HPO_FRACTION     = 0.3     # use 30% of TRAIN for HPO
HPO_EPOCHS       = 5
HPO_FOLDS        = 3
N_TRIALS         = 20

X_hpo, y1_hpo, y2_hpo, y3_hpo = load_subset_data(TRAIN_PATH, fraction=HPO_FRACTION)
t1_hpo = np.argmax(y1_hpo, axis=1)
t2_hpo = np.argmax(y2_hpo, axis=1)
t3_hpo = np.argmax(y3_hpo, axis=1)

def objective(trial):
    # Modern Optuna API
    lr          = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    dropout     = trial.suggest_float('dropout', 0.2, 0.5)
    batch_size  = trial.suggest_categorical('batch_size', [16, 32, 64])
    filters1    = trial.suggest_categorical('filters1', [32, 48, 64])
    filters2    = trial.suggest_categorical('filters2', [64, 96, 128])
    dense_units = trial.suggest_categorical('dense_units', [128, 192, 256])

    kf = KFold(n_splits=HPO_FOLDS, shuffle=True, random_state=SEED)
    f1_scores = []

    for train_idx, val_idx in kf.split(X_hpo):
        X_tr, X_va = X_hpo[train_idx], X_hpo[val_idx]
        y1_tr, y1_va = y1_hpo[train_idx], y1_hpo[val_idx]
        y2_tr, y2_va = y2_hpo[train_idx], y2_hpo[val_idx]
        y3_tr, y3_va = y3_hpo[train_idx], y3_hpo[val_idx]

        model = build_triple_digit_model(dropout_rate=dropout,
                                         filters1=filters1, filters2=filters2, dense_units=dense_units)
        model.compile(optimizer=Adam(learning_rate=lr),
                      loss='categorical_crossentropy')

        model.fit(X_tr, [y1_tr, y2_tr, y3_tr],
                  validation_data=(X_va, [y1_va, y2_va, y3_va]),
                  epochs=HPO_EPOCHS,
                  batch_size=batch_size,
                  verbose=0)

        preds = model.predict(X_va, verbose=0)
        p1, p2, p3 = [np.argmax(p, axis=1) for p in preds]
        t1, t2, t3 = [np.argmax(y, axis=1) for y in [y1_va, y2_va, y3_va]]

        f1_1 = f1_score(t1, p1, average='macro')
        f1_2 = f1_score(t2, p2, average='macro')
        f1_3 = f1_score(t3, p3, average='macro')
        f1_scores.append((f1_1 + f1_2 + f1_3) / 3.0)

        # Free up memory
        tf.keras.backend.clear_session()

    return float(np.mean(f1_scores))

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=N_TRIALS)

best_params = study.best_trial.params
print("Best params:", best_params)
print(f"Best (CV) macro-F1: {study.best_value:.4f}")

# Save study & params
joblib.dump(study, os.path.join(ART_DIR, "study.pkl"))
with open(os.path.join(ART_DIR, "best_params.json"), "w") as f:
    json.dump(best_params, f, indent=2)

# =========================
# 7) Final training with best hyperparams
# =========================
# Load full train/val for final training
X_train, y1_train, y2_train, y3_train = load_subset_data(TRAIN_PATH, fraction=1.0)
X_val,   y1_val,   y2_val,   y3_val   = load_subset_data(VAL_PATH,   fraction=1.0)
X_test,  y1_test,  y2_test,  y3_test  = load_subset_data(TEST_PATH,  fraction=1.0)

best_lr          = float(best_params['lr'])
best_dropout     = float(best_params['dropout'])
best_batch       = int(best_params['batch_size'])
best_filters1    = int(best_params['filters1'])
best_filters2    = int(best_params['filters2'])
best_dense_units = int(best_params['dense_units'])

model = build_triple_digit_model(dropout_rate=best_dropout,
                                 filters1=best_filters1, filters2=best_filters2,
                                 dense_units=best_dense_units)
model.compile(optimizer=Adam(learning_rate=best_lr),
              loss='categorical_crossentropy',
              metrics=['accuracy','accuracy','accuracy'])

early   = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
chkpt   = ModelCheckpoint(os.path.join(ART_DIR, 'best_full_model.h5'),
                          save_best_only=True, monitor='val_loss')

history = model.fit(
    X_train, [y1_train, y2_train, y3_train],
    validation_data=(X_val, [y1_val, y2_val, y3_val]),
    epochs=30,
    batch_size=best_batch,
    callbacks=[early, chkpt],
    verbose=1
)

# Save native Keras format
model_path_keras = os.path.join(ART_DIR, "best_full_model.keras")
model.save(model_path_keras)

# Save model summary
with open(os.path.join(ART_DIR, "best_full_model_summary.txt"), "w") as f:
    with redirect_stdout(f):
        model.summary()

# Save history
with open(os.path.join(ART_DIR, "history.pkl"), "wb") as f:
    pickle.dump(history.history, f)
pd.DataFrame(history.history).to_csv(os.path.join(ART_DIR, "history.csv"), index=False)

# Save config
run_config = {
    "date": str(datetime.datetime.now()),
    "epochs_ran": len(history.history["loss"]),
    "batch_size": best_batch,
    "train_path": TRAIN_PATH,
    "val_path": VAL_PATH,
    "test_path": TEST_PATH,
    "seed": SEED,
    "best_params": best_params
}
with open(os.path.join(ART_DIR, "config.yaml"), "w") as f:
    yaml.dump(run_config, f)

# =========================
# 8) Final evaluation (Test) + visualizations
# =========================
# Ensure we use the best checkpoint weights if early stopped
if os.path.exists(os.path.join(ART_DIR, 'best_full_model.h5')):
    model.load_weights(os.path.join(ART_DIR, 'best_full_model.h5'))

# Keras evaluate (losses + accuracies)
test_eval = model.evaluate(X_test, [y1_test, y2_test, y3_test], verbose=0)
# unpack for readability
# [total_loss, loss1, loss2, loss3, acc1, acc2, acc3]
eval_dict = {
    "total_loss": test_eval[0],
    "digit1_loss": test_eval[1],
    "digit2_loss": test_eval[2],
    "digit3_loss": test_eval[3],
    "digit1_acc":  test_eval[4],
    "digit2_acc":  test_eval[5],
    "digit3_acc":  test_eval[6],
}
print("Test metrics (Keras evaluate):", eval_dict)

# Predictions for F1/Exact match & confusion matrices
preds = model.predict(X_test, verbose=0)
p1, p2, p3 = [np.argmax(p, axis=1) for p in preds]
t1, t2, t3 = [np.argmax(y, axis=1) for y in [y1_test, y2_test, y3_test]]

f1_1 = f1_score(t1, p1, average='macro')
f1_2 = f1_score(t2, p2, average='macro')
f1_3 = f1_score(t3, p3, average='macro')
avg_f1 = float((f1_1 + f1_2 + f1_3) / 3.0)

exact_match = float(np.mean((t1==p1) & (t2==p2) & (t3==p3)))

metrics_json = {
    "keras_evaluate": eval_dict,
    "macro_f1_digit1": f1_1,
    "macro_f1_digit2": f1_2,
    "macro_f1_digit3": f1_3,
    "macro_f1_avg": avg_f1,
    "exact_match_accuracy": exact_match
}
with open(os.path.join(ART_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_json, f, indent=2)

# Learning curves
save_learning_curves(history, os.path.join(ART_DIR, "learning"))

# Confusion matrices
cm1 = save_confusion_matrix(t1, p1, "digit1", os.path.join(ART_DIR, "cm_digit1.png"))
cm2 = save_confusion_matrix(t2, p2, "digit2", os.path.join(ART_DIR, "cm_digit2.png"))
cm3 = save_confusion_matrix(t3, p3, "digit3", os.path.join(ART_DIR, "cm_digit3.png"))

# Classification reports (saved as text)
with open(os.path.join(ART_DIR, "classification_reports.txt"), "w") as f:
    for name, t, p in [('digit1', t1, p1), ('digit2', t2, p2), ('digit3', t3, p3)]:
        f.write(f"\n=== {name} ===\n")
        f.write(classification_report(t, p, digits=4))

# Optional: Save predictions CSV
preds_df = pd.DataFrame({
    "true_d1": t1, "pred_d1": p1,
    "true_d2": t2, "pred_d2": p2,
    "true_d3": t3, "pred_d3": p3
})
preds_df.to_csv(os.path.join(ART_DIR, "test_predictions.csv"), index=False)

print(f" All artifacts saved to: {os.path.abspath(ART_DIR)}")


#############################################################################
#FROM HERE EVERYTHING ARE OUTPUTS#
#############################################################################

Python 3.11.12 | packaged by conda-forge | (main, Apr 10 2025, 22:09:00) [MSC v.1943 64 bit (AMD64)]
Type "copyright", "credits" or "license" for more information.

IPython 8.36.0 -- An enhanced Interactive Python. Type '?' for help.

%runcell -i 1 C:/Users/AlvaroRivera-Eraso/Documents/HULL/CNNsMinst.py
C:\Users\AlvaroRivera-Eraso\AppData\Local\spyder-6\envs\spyder-runtime\Lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
2025-08-23 05:42:06.884981: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-08-23 05:42:12.248849: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

%runcell -i 2 C:/Users/AlvaroRivera-Eraso/Documents/HULL/CNNsMinst.py

%runcell -i 3 C:/Users/AlvaroRivera-Eraso/Documents/HULL/CNNsMinst.py

%runcell -i 14 C:/Users/AlvaroRivera-Eraso/Documents/HULL/CNNsMinst.py
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
File c:\users\alvarorivera-eraso\documents\hull\cnnsminst.py:405
    402 ART_DIR = "artifacts"
    404 # 1️⃣ Model
--> 405 model = load_model(os.path.join(ART_DIR, "best_full_model.keras"), compile=False)
    407 # 2️⃣ Compile (if you’ll evaluate / continue training)
    408 with open(os.path.join(ART_DIR, "best_params.json")) as f:

File ~\AppData\Local\spyder-6\envs\spyder-runtime\Lib\site-packages\keras\src\saving\saving_api.py:200, in load_model(filepath, custom_objects, compile, safe_mode)
    196     return legacy_h5_format.load_model_from_hdf5(
    197         filepath, custom_objects=custom_objects, compile=compile
    198     )
    199 elif str(filepath).endswith(".keras"):
--> 200     raise ValueError(
    201         f"File not found: filepath={filepath}. "
    202         "Please ensure the file is an accessible `.keras` "
    203         "zip file."
    204     )
    205 else:
    206     raise ValueError(
    207         f"File format not supported: filepath={filepath}. "
    208         "Keras 3 only supports V3 `.keras` files and "
   (...)
    217         "might have a different name)."
    218     )

ValueError: File not found: filepath=artifacts\best_full_model.keras. Please ensure the file is an accessible `.keras` zip file.

%runcell -i 15 C:/Users/AlvaroRivera-Eraso/Documents/HULL/CNNsMinst.py
False
False

%runcell -i 14 C:/Users/AlvaroRivera-Eraso/Documents/HULL/CNNsMinst.py
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
File c:\users\alvarorivera-eraso\documents\hull\cnnsminst.py:405
    402 ART_DIR = "artifacts"
    404 # 1️⃣ Model
--> 405 model = load_model(os.path.join(ART_DIR, "best_full_model.keras"), compile=False)
    407 # 2️⃣ Compile (if you’ll evaluate / continue training)
    408 with open(os.path.join(ART_DIR, "best_params.json")) as f:

File ~\AppData\Local\spyder-6\envs\spyder-runtime\Lib\site-packages\keras\src\saving\saving_api.py:200, in load_model(filepath, custom_objects, compile, safe_mode)
    196     return legacy_h5_format.load_model_from_hdf5(
    197         filepath, custom_objects=custom_objects, compile=compile
    198     )
    199 elif str(filepath).endswith(".keras"):
--> 200     raise ValueError(
    201         f"File not found: filepath={filepath}. "
    202         "Please ensure the file is an accessible `.keras` "
    203         "zip file."
    204     )
    205 else:
    206     raise ValueError(
    207         f"File format not supported: filepath={filepath}. "
    208         "Keras 3 only supports V3 `.keras` files and "
   (...)
    217         "might have a different name)."
    218     )

ValueError: File not found: filepath=artifacts\best_full_model.keras. Please ensure the file is an accessible `.keras` zip file.

%runcell -i 14 C:/Users/AlvaroRivera-Eraso/Documents/HULL/CNNsMinst.py
2025-08-23 06:14:31.538103: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Optuna study reloaded. Best trial: {'lr': 0.0004585767605125491, 'dropout': 0.32171964891544136, 'batch_size': 16}

 
Important
Figures are displayed in the Plots pane by default. To make them also appear inline in the console, you need to uncheck "Mute inline plotting" under the options menu of Plots.
 

%runcell -i 15 C:/Users/AlvaroRivera-Eraso/Documents/HULL/CNNsMinst.py
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
File c:\users\alvarorivera-eraso\documents\hull\cnnsminst.py:439
    436 #%%
    438 import pickle
--> 439 hist = pickle.load(open("artifacts/history.pkl", "rb"))
    440 print(hist.keys())   # should have 'loss', 'digit1_accuracy', ...

File ~\AppData\Local\spyder-6\envs\spyder-runtime\Lib\site-packages\IPython\core\interactiveshell.py:324, in _modified_open(file, *args, **kwargs)
    317 if file in {0, 1, 2}:
    318     raise ValueError(
    319         f"IPython won't let you open fd={file} by default "
    320         "as it is likely to crash IPython. If you know what you are doing, "
    321         "you can use builtins' open."
    322     )
--> 324 return io_open(file, *args, **kwargs)

FileNotFoundError: [Errno 2] No such file or directory: 'artifacts/history.pkl'

%runcell -i 15 C:/Users/AlvaroRivera-Eraso/Documents/HULL/CNNsMinst.py
dict_keys(['digit1_accuracy', 'digit1_loss', 'digit2_accuracy', 'digit2_loss', 'digit3_accuracy', 'digit3_loss', 'loss', 'val_digit1_accuracy', 'val_digit1_loss', 'val_digit2_accuracy', 'val_digit2_loss', 'val_digit3_accuracy', 'val_digit3_loss', 'val_loss'])

%runcell -i 15 C:/Users/AlvaroRivera-Eraso/Documents/HULL/CNNsMinst.py
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
File c:\users\alvarorivera-eraso\documents\hull\cnnsminst.py:437
    433     print(" Optuna study.pkl not found in artifacts directory.")
    436 #%%
--> 437 print(best_params())

TypeError: 'dict' object is not callable

%runcell -i 15 C:/Users/AlvaroRivera-Eraso/Documents/HULL/CNNsMinst.py
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
File c:\users\alvarorivera-eraso\documents\hull\cnnsminst.py:437
    433     print("⚠️ Optuna study.pkl not found in artifacts directory.")
    436 #%%
--> 437 print("Dropout rate:", [l.rate for l in loaded.layers if "dropout" in l.name])

NameError: name 'loaded' is not defined

%runcell -i 15 C:/Users/AlvaroRivera-Eraso/Documents/HULL/CNNsMinst.py
Model: "functional_22"
┌─────────────────────┬───────────────────┬────────────┬───────────────────┐
│ Layer (type)        │ Output Shape      │    Param # │ Connected to      │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ input_layer_22      │ (None, 84, 84, 1) │          0 │ -                 │
│ (InputLayer)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_44 (Conv2D)  │ (None, 84, 84,    │        320 │ input_layer_22[0… │
│                     │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_44    │ (None, 42, 42,    │          0 │ conv2d_44[0][0]   │
│ (MaxPooling2D)      │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_45 (Conv2D)  │ (None, 42, 42,    │     18,496 │ max_pooling2d_44… │
│                     │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_45    │ (None, 21, 21,    │          0 │ conv2d_45[0][0]   │
│ (MaxPooling2D)      │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ flatten_22          │ (None, 28224)     │          0 │ max_pooling2d_45… │
│ (Flatten)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_22 (Dense)    │ (None, 128)       │  3,612,800 │ flatten_22[0][0]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_22          │ (None, 128)       │          0 │ dense_22[0][0]    │
│ (Dropout)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ digit1 (Dense)      │ (None, 10)        │      1,290 │ dropout_22[0][0]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ digit2 (Dense)      │ (None, 10)        │      1,290 │ dropout_22[0][0]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ digit3 (Dense)      │ (None, 10)        │      1,290 │ dropout_22[0][0]  │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
 Total params: 3,635,486 (13.87 MB)
 Trainable params: 3,635,486 (13.87 MB)
 Non-trainable params: 0 (0.00 B)

%runcell -i 15 C:/Users/AlvaroRivera-Eraso/Documents/HULL/CNNsMinst.py
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
File c:\users\alvarorivera-eraso\documents\hull\cnnsminst.py:441
    439 random.seed(SEED)
    440 np.random.seed(SEED)
--> 441 tf.random.set_seed(SEED)
    443 # =========================
    444 # 1) Paths & unzip 
    445 # =========================
    446 ZIP_PATH = r"C:\Users\AlvaroRivera-Eraso\Documents\HULL\d.zip"

NameError: name 'tf' is not defined

%runcell -i 1 C:/Users/AlvaroRivera-Eraso/Documents/HULL/CNNsMinst.py

%runcell -i 15 C:/Users/AlvaroRivera-Eraso/Documents/HULL/CNNsMinst.py
Subsetting train: 100%|██████████| 640/640 [00:30<00:00, 20.85it/s]
[I 2025-08-23 07:02:23,823] A new study created in memory with name: no-name-a43c0e9c-05ff-480e-a6db-dc82482e55ad
WARNING:tensorflow:From C:\Users\AlvaroRivera-Eraso\AppData\Local\spyder-6\envs\spyder-runtime\Lib\site-packages\keras\src\backend\common\global_state.py:82: The name tf.reset_default_graph is deprecated. Please use tf.compat.v1.reset_default_graph instead.

[I 2025-08-23 07:18:50,480] Trial 0 finished with value: 0.23215664448142026 and parameters: {'lr': 0.006683054641341619, 'dropout': 0.21835834726926442, 'batch_size': 64, 'filters1': 32, 'filters2': 128, 'dense_units': 128}. Best is trial 0 with value: 0.23215664448142026.
[I 2025-08-23 07:34:55,102] Trial 1 finished with value: 0.8843093663711161 and parameters: {'lr': 0.0015172396385387968, 'dropout': 0.40595116979751167, 'batch_size': 64, 'filters1': 64, 'filters2': 64, 'dense_units': 128}. Best is trial 1 with value: 0.8843093663711161.
[I 2025-08-23 07:56:31,923] Trial 2 finished with value: 0.01998353074502882 and parameters: {'lr': 0.003988610876880425, 'dropout': 0.35959977736716753, 'batch_size': 64, 'filters1': 64, 'filters2': 96, 'dense_units': 192}. Best is trial 1 with value: 0.8843093663711161.
[I 2025-08-23 08:15:36,744] Trial 3 finished with value: 0.019378998496702136 and parameters: {'lr': 0.008137181814133032, 'dropout': 0.4908592883547912, 'batch_size': 16, 'filters1': 32, 'filters2': 64, 'dense_units': 256}. Best is trial 1 with value: 0.8843093663711161.
[I 2025-08-23 08:48:28,347] Trial 4 finished with value: 0.8890238259503098 and parameters: {'lr': 0.0002657262358979879, 'dropout': 0.24045447236066408, 'batch_size': 16, 'filters1': 64, 'filters2': 96, 'dense_units': 256}. Best is trial 4 with value: 0.8890238259503098.
[I 2025-08-23 09:26:33,486] Trial 5 finished with value: 0.9041748936944117 and parameters: {'lr': 0.00039716424556440836, 'dropout': 0.3727908573665352, 'batch_size': 16, 'filters1': 64, 'filters2': 128, 'dense_units': 192}. Best is trial 5 with value: 0.9041748936944117.
[I 2025-08-23 09:53:24,631] Trial 6 finished with value: 0.5181759556859148 and parameters: {'lr': 0.002926632717127466, 'dropout': 0.22099455781897975, 'batch_size': 32, 'filters1': 64, 'filters2': 96, 'dense_units': 192}. Best is trial 5 with value: 0.9041748936944117.
[I 2025-08-23 10:09:37,003] Trial 7 finished with value: 0.9023911862580368 and parameters: {'lr': 0.0006807540449723056, 'dropout': 0.47455981532984476, 'batch_size': 32, 'filters1': 32, 'filters2': 96, 'dense_units': 192}. Best is trial 5 with value: 0.9041748936944117.
[I 2025-08-23 10:28:14,406] Trial 8 finished with value: 0.01963657418780462 and parameters: {'lr': 0.00896826988805844, 'dropout': 0.3217542237707596, 'batch_size': 64, 'filters1': 48, 'filters2': 96, 'dense_units': 256}. Best is trial 5 with value: 0.9041748936944117.
[I 2025-08-23 10:58:04,030] Trial 9 finished with value: 0.8776107117034107 and parameters: {'lr': 0.0014868327611156974, 'dropout': 0.22790547116265414, 'batch_size': 64, 'filters1': 64, 'filters2': 128, 'dense_units': 256}. Best is trial 5 with value: 0.9041748936944117.
[I 2025-08-23 11:30:46,830] Trial 10 finished with value: 0.8742434048119919 and parameters: {'lr': 0.00015699276041953488, 'dropout': 0.3104036988852514, 'batch_size': 16, 'filters1': 48, 'filters2': 128, 'dense_units': 192}. Best is trial 5 with value: 0.9041748936944117.
[I 2025-08-23 11:51:31,570] Trial 11 finished with value: 0.9031327964792527 and parameters: {'lr': 0.0005650562470514947, 'dropout': 0.4952670792003919, 'batch_size': 32, 'filters1': 32, 'filters2': 128, 'dense_units': 192}. Best is trial 5 with value: 0.9041748936944117.
[I 2025-08-23 12:11:56,024] Trial 12 finished with value: 0.8988195147360613 and parameters: {'lr': 0.0004909254595478861, 'dropout': 0.422443801250427, 'batch_size': 32, 'filters1': 32, 'filters2': 128, 'dense_units': 192}. Best is trial 5 with value: 0.9041748936944117.
[I 2025-08-23 12:32:20,836] Trial 13 finished with value: 0.8863752048080173 and parameters: {'lr': 0.00030695357773444535, 'dropout': 0.43691580541529057, 'batch_size': 32, 'filters1': 32, 'filters2': 128, 'dense_units': 192}. Best is trial 5 with value: 0.9041748936944117.
[I 2025-08-23 13:06:01,976] Trial 14 finished with value: 0.8627283787886747 and parameters: {'lr': 0.0001240050921005089, 'dropout': 0.375458374904442, 'batch_size': 16, 'filters1': 48, 'filters2': 128, 'dense_units': 192}. Best is trial 5 with value: 0.9041748936944117.
[I 2025-08-23 13:38:08,937] Trial 15 finished with value: 0.8981006251980456 and parameters: {'lr': 0.0008513313009412768, 'dropout': 0.285833838716171, 'batch_size': 32, 'filters1': 64, 'filters2': 128, 'dense_units': 192}. Best is trial 5 with value: 0.9041748936944117.
[I 2025-08-23 14:03:03,791] Trial 16 finished with value: 0.899471187883283 and parameters: {'lr': 0.00039676588371600786, 'dropout': 0.45827992179164934, 'batch_size': 16, 'filters1': 32, 'filters2': 128, 'dense_units': 128}. Best is trial 5 with value: 0.9041748936944117.
[I 2025-08-23 14:16:29,503] Trial 17 finished with value: 0.8519024753197685 and parameters: {'lr': 0.00022419394926901477, 'dropout': 0.39093405483615606, 'batch_size': 32, 'filters1': 32, 'filters2': 64, 'dense_units': 192}. Best is trial 5 with value: 0.9041748936944117.
[I 2025-08-23 14:53:28,817] Trial 18 finished with value: 0.8812198568850316 and parameters: {'lr': 0.0012154140979933092, 'dropout': 0.26850629861505937, 'batch_size': 16, 'filters1': 64, 'filters2': 128, 'dense_units': 192}. Best is trial 5 with value: 0.9041748936944117.
[I 2025-08-23 15:16:00,468] Trial 19 finished with value: 0.8994782536907397 and parameters: {'lr': 0.0005648622396590301, 'dropout': 0.3296939108621456, 'batch_size': 32, 'filters1': 48, 'filters2': 128, 'dense_units': 128}. Best is trial 5 with value: 0.9041748936944117.
Best params: {'lr': 0.00039716424556440836, 'dropout': 0.3727908573665352, 'batch_size': 16, 'filters1': 64, 'filters2': 128, 'dense_units': 192}
Best (CV) macro-F1: 0.9042
Subsetting train: 100%|██████████| 640/640 [01:16<00:00,  8.37it/s]
Subsetting val: 100%|██████████| 160/160 [00:21<00:00,  7.30it/s]
Subsetting test: 100%|██████████| 200/200 [00:26<00:00,  7.53it/s]
Epoch 1/30
4000/4000 ━━━━━━━━━━━━━━━━━━━━ 0s 174ms/step - digit1_accuracy: 0.5832 - digit1_loss: 1.1681 - digit2_accuracy: 0.5895 - digit2_loss: 1.1487 - digit3_accuracy: 0.5892 - digit3_loss: 1.1561 - loss: 3.4729WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4000/4000 ━━━━━━━━━━━━━━━━━━━━ 742s 185ms/step - digit1_accuracy: 0.5833 - digit1_loss: 1.1680 - digit2_accuracy: 0.5896 - digit2_loss: 1.1486 - digit3_accuracy: 0.5893 - digit3_loss: 1.1560 - loss: 3.4726 - val_digit1_accuracy: 0.9305 - val_digit1_loss: 0.2324 - val_digit2_accuracy: 0.9352 - val_digit2_loss: 0.2212 - val_digit3_accuracy: 0.9268 - val_digit3_loss: 0.2387 - val_loss: 0.6922
Epoch 2/30
4000/4000 ━━━━━━━━━━━━━━━━━━━━ 0s 176ms/step - digit1_accuracy: 0.9045 - digit1_loss: 0.2976 - digit2_accuracy: 0.9032 - digit2_loss: 0.2997 - digit3_accuracy: 0.9062 - digit3_loss: 0.2884 - loss: 0.8856WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4000/4000 ━━━━━━━━━━━━━━━━━━━━ 746s 186ms/step - digit1_accuracy: 0.9045 - digit1_loss: 0.2976 - digit2_accuracy: 0.9032 - digit2_loss: 0.2996 - digit3_accuracy: 0.9062 - digit3_loss: 0.2884 - loss: 0.8856 - val_digit1_accuracy: 0.9496 - val_digit1_loss: 0.1602 - val_digit2_accuracy: 0.9499 - val_digit2_loss: 0.1655 - val_digit3_accuracy: 0.9496 - val_digit3_loss: 0.1656 - val_loss: 0.4912
Epoch 3/30
4000/4000 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - digit1_accuracy: 0.9409 - digit1_loss: 0.1781 - digit2_accuracy: 0.9410 - digit2_loss: 0.1792 - digit3_accuracy: 0.9393 - digit3_loss: 0.1833 - loss: 0.5405WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4000/4000 ━━━━━━━━━━━━━━━━━━━━ 718s 180ms/step - digit1_accuracy: 0.9409 - digit1_loss: 0.1781 - digit2_accuracy: 0.9411 - digit2_loss: 0.1792 - digit3_accuracy: 0.9393 - digit3_loss: 0.1833 - loss: 0.5405 - val_digit1_accuracy: 0.9544 - val_digit1_loss: 0.1450 - val_digit2_accuracy: 0.9569 - val_digit2_loss: 0.1422 - val_digit3_accuracy: 0.9565 - val_digit3_loss: 0.1419 - val_loss: 0.4292
Epoch 4/30
4000/4000 ━━━━━━━━━━━━━━━━━━━━ 0s 169ms/step - digit1_accuracy: 0.9602 - digit1_loss: 0.1215 - digit2_accuracy: 0.9591 - digit2_loss: 0.1218 - digit3_accuracy: 0.9578 - digit3_loss: 0.1226 - loss: 0.3660WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4000/4000 ━━━━━━━━━━━━━━━━━━━━ 715s 179ms/step - digit1_accuracy: 0.9602 - digit1_loss: 0.1215 - digit2_accuracy: 0.9591 - digit2_loss: 0.1218 - digit3_accuracy: 0.9578 - digit3_loss: 0.1226 - loss: 0.3660 - val_digit1_accuracy: 0.9557 - val_digit1_loss: 0.1478 - val_digit2_accuracy: 0.9603 - val_digit2_loss: 0.1352 - val_digit3_accuracy: 0.9605 - val_digit3_loss: 0.1336 - val_loss: 0.4166
Epoch 5/30**********************************************
4000/4000 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - digit1_accuracy: 0.9689 - digit1_loss: 0.0904 - digit2_accuracy: 0.9680 - digit2_loss: 0.0943 - digit3_accuracy: 0.9706 - digit3_loss: 0.0890 - loss: 0.2738WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
4000/4000 ━━━━━━━━━━━━━━━━━━━━ 719s 180ms/step - digit1_accuracy: 0.9689 - digit1_loss: 0.0904 - digit2_accuracy: 0.9680 - digit2_loss: 0.0943 - digit3_accuracy: 0.9706 - digit3_loss: 0.0890 - loss: 0.2738 - val_digit1_accuracy: 0.9595 - val_digit1_loss: 0.1391 - val_digit2_accuracy: 0.9603 - val_digit2_loss: 0.1391 - val_digit3_accuracy: 0.9599 - val_digit3_loss: 0.1381 - val_loss: 0.4162
Epoch 6/30
4000/4000 ━━━━━━━━━━━━━━━━━━━━ 717s 179ms/step - digit1_accuracy: 0.9737 - digit1_loss: 0.0765 - digit2_accuracy: 0.9753 - digit2_loss: 0.0712 - digit3_accuracy: 0.9747 - digit3_loss: 0.0732 - loss: 0.2210 - val_digit1_accuracy: 0.9618 - val_digit1_loss: 0.1419 - val_digit2_accuracy: 0.9601 - val_digit2_loss: 0.1410 - val_digit3_accuracy: 0.9593 - val_digit3_loss: 0.1465 - val_loss: 0.4294
Epoch 7/30
4000/4000 ━━━━━━━━━━━━━━━━━━━━ 717s 179ms/step - digit1_accuracy: 0.9812 - digit1_loss: 0.0552 - digit2_accuracy: 0.9784 - digit2_loss: 0.0610 - digit3_accuracy: 0.9797 - digit3_loss: 0.0577 - loss: 0.1739 - val_digit1_accuracy: 0.9601 - val_digit1_loss: 0.1520 - val_digit2_accuracy: 0.9641 - val_digit2_loss: 0.1359 - val_digit3_accuracy: 0.9629 - val_digit3_loss: 0.1386 - val_loss: 0.4265
Epoch 8/30
4000/4000 ━━━━━━━━━━━━━━━━━━━━ 717s 179ms/step - digit1_accuracy: 0.9830 - digit1_loss: 0.0502 - digit2_accuracy: 0.9821 - digit2_loss: 0.0504 - digit3_accuracy: 0.9825 - digit3_loss: 0.0511 - loss: 0.1517 - val_digit1_accuracy: 0.9608 - val_digit1_loss: 0.1511 - val_digit2_accuracy: 0.9634 - val_digit2_loss: 0.1420 - val_digit3_accuracy: 0.9622 - val_digit3_loss: 0.1463 - val_loss: 0.4394
Epoch 9/30
4000/4000 ━━━━━━━━━━━━━━━━━━━━ 731s 183ms/step - digit1_accuracy: 0.9833 - digit1_loss: 0.0472 - digit2_accuracy: 0.9844 - digit2_loss: 0.0460 - digit3_accuracy: 0.9846 - digit3_loss: 0.0432 - loss: 0.1364 - val_digit1_accuracy: 0.9636 - val_digit1_loss: 0.1484 - val_digit2_accuracy: 0.9614 - val_digit2_loss: 0.1571 - val_digit3_accuracy: 0.9658 - val_digit3_loss: 0.1399 - val_loss: 0.4454
Epoch 10/30
4000/4000 ━━━━━━━━━━━━━━━━━━━━ 718s 180ms/step - digit1_accuracy: 0.9867 - digit1_loss: 0.0415 - digit2_accuracy: 0.9857 - digit2_loss: 0.0423 - digit3_accuracy: 0.9864 - digit3_loss: 0.0413 - loss: 0.1251 - val_digit1_accuracy: 0.9636 - val_digit1_loss: 0.1502 - val_digit2_accuracy: 0.9663 - val_digit2_loss: 0.1452 - val_digit3_accuracy: 0.9643 - val_digit3_loss: 0.1455 - val_loss: 0.4409
---------------------------------------------------------------------------



