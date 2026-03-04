import os
import glob
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import tensorflow as tf
import joblib

# ------------------------- CONFIG -------------------------
train_base_dir = r"C:\Users\adith\Documents\EduVision\DAISEE\DAiSEE\DataSet\Train"
test_base_dir = r"C:\Users\adith\Documents\EduVision\DAISEE\DAiSEE\DataSet\Test"
train_labels_file = r"C:\Users\adith\Documents\EduVision\DAISEE\DAiSEE\Labels\TrainLabels.csv"
test_labels_file = r"C:\Users\adith\Documents\EduVision\DAISEE\DAiSEE\Labels\TestLabels.csv"
num_frames = 16
frame_size = (224, 224)
num_classes = 4  # Engagement levels 0-3

# ------------------- ATTENTION LAYER (DEFINE FIRST) -------------------
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

    def get_config(self):
        return super(Attention, self).get_config()

# ------------------- HELPER FUNCTIONS -------------------
def collect_video_paths(base_dir):
    video_paths = glob.glob(os.path.join(base_dir, "*", "*", "*.avi"))
    return {os.path.basename(path): path for path in video_paths}

def video_to_frames(video_path, num_frames=num_frames, size=frame_size, augment=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error opening video: {video_path}")
        return np.array([])
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"❌ No frames in video: {video_path}")
        cap.release()
        return np.array([])
        
    frame_idxs = np.linspace(0, total_frames-1, num_frames).astype(int)
    frames = []
    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if augment:
                if np.random.rand() > 0.5:
                    frame = cv2.flip(frame, 1)  # horizontal flip
                factor = 0.8 + np.random.rand() * 0.4  # brightness
                frame = np.clip(frame * factor, 0, 255).astype(np.uint8)
            frames.append(frame)
    cap.release()
    
    if len(frames) == 0:
        print(f"❌ No frames extracted from: {video_path}")
        return np.array([])
        
    return np.array(frames)

# ---------------- STEP 1: LOAD RESNET MODEL ----------------
print("🔧 Loading ResNet50 model...")
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

def extract_features(frames):
    frames = preprocess_input(frames.astype("float32"))
    features = resnet_model.predict(frames, verbose=0)
    return features

# ---------------- STEP 2: PREPARE DATA ----------------
def prepare_data(df_labels, clipid_to_path, augment=False):
    X, y_eng, y_bor, y_conf, y_fru = [], [], [], [], []
    df_labels["video_path"] = df_labels["ClipID"].map(clipid_to_path)
    df_labels = df_labels.dropna(subset=["video_path"])
    print(f"📊 Total videos collected: {len(df_labels)}")

    successful_videos = 0
    for idx, row in tqdm(df_labels.iterrows(), total=len(df_labels), desc="Processing videos", unit="video"):
        frames = video_to_frames(row["video_path"], augment=augment)
        if len(frames) == 0:
            continue
        features = extract_features(frames)
        X.append(features)
        y_eng.append(row["Engagement"])
        y_bor.append(row["Boredom"])
        y_conf.append(row["Confusion"])
        y_fru.append(row["Frustration"])
        successful_videos += 1
        
    print(f"✅ Successfully processed {successful_videos}/{len(df_labels)} videos")
    return np.array(X), np.array(y_eng), np.array(y_bor), np.array(y_conf), np.array(y_fru)

# ---------------- STEP 3: LOAD LABELS ----------------
print("📋 Loading labels...")
df_train_labels = pd.read_csv(train_labels_file)
df_test_labels = pd.read_csv(test_labels_file)
df_train_labels.columns = df_train_labels.columns.str.strip()
df_test_labels.columns = df_test_labels.columns.str.strip()

print("🎬 Collecting video paths...")
train_clipid_to_path = collect_video_paths(train_base_dir)
test_clipid_to_path = collect_video_paths(test_base_dir)

print(f"📁 Found {len(train_clipid_to_path)} training videos")
print(f"📁 Found {len(test_clipid_to_path)} test videos")

# ---------------- STEP 4: PREPARE TRAIN DATA ----------------
print("\n🔄 Preparing training data...")
X_train, y_eng_train, y_bor_train, y_conf_train, y_fru_train = prepare_data(df_train_labels, train_clipid_to_path, augment=True)
print(f"✅ X_train shape: {X_train.shape}")

# ---------------- STEP 5: FEATURE NORMALIZATION + PCA ----------------
print("\n⚙️ Applying feature normalization and PCA...")
X_train_flat = X_train.reshape(-1, X_train.shape[2])
scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)

pca = PCA(n_components=512)
X_train_flat = pca.fit_transform(X_train_flat)
X_train = X_train_flat.reshape(-1, num_frames, 512)

print("🔄 Preparing test data...")
X_test, y_eng_test, y_bor_test, y_conf_test, y_fru_test = prepare_data(df_test_labels, test_clipid_to_path)
X_test_flat = X_test.reshape(-1, X_test.shape[2])
X_test_flat = scaler.transform(X_test_flat)
X_test_flat = pca.transform(X_test_flat)
X_test = X_test_flat.reshape(-1, num_frames, 512)
print(f"✅ X_test shape: {X_test.shape}")

# ---------------- STEP 6: BUILD MODEL ----------------
print("\n🤖 Building model...")
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
shared_lstm = Bidirectional(LSTM(256, return_sequences=True))(input_layer)
shared_lstm = Dropout(0.5)(shared_lstm)
shared_lstm = Bidirectional(LSTM(128, return_sequences=True))(shared_lstm)
shared_lstm = Attention()(shared_lstm)
shared_lstm = Dropout(0.5)(shared_lstm)
shared_lstm = Dense(128, activation="relu")(shared_lstm)
shared_lstm = Dropout(0.5)(shared_lstm)

out_eng = Dense(num_classes, activation="softmax", name="engagement")(shared_lstm)
out_bor = Dense(num_classes, activation="softmax", name="boredom")(shared_lstm)
out_conf = Dense(num_classes, activation="softmax", name="confusion")(shared_lstm)
out_fru = Dense(num_classes, activation="softmax", name="frustration")(shared_lstm)

model = Model(inputs=input_layer, outputs=[out_eng, out_bor, out_conf, out_fru])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics={"engagement": "accuracy",
             "boredom": "accuracy",
             "confusion": "accuracy",
             "frustration": "accuracy"}
)

print("✅ Model built successfully!")
model.summary()

# ---------------- STEP 7: TRAIN ----------------
print("\n🎯 Starting training...")
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)

history = model.fit(
    X_train,
    [y_eng_train, y_bor_train, y_conf_train, y_fru_train],
    epochs=300,
    batch_size=16,
    validation_split=0.1,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ---------------- STEP 8: SAVE MODEL AND PREPROCESSORS ----------------
print("\n💾 Saving model and preprocessors...")
model.save("resnet_bilstm_attention_48_2.h5")

# Save scaler and PCA
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')

# Save test data for later use
np.save('X_test_preprocessed.npy', X_test)
np.save('y_eng_test.npy', y_eng_test)
np.save('y_bor_test.npy', y_bor_test)
np.save('y_conf_test.npy', y_conf_test)
np.save('y_fru_test.npy', y_fru_test)

print("✅ Model saved as resnet_bilstm_attention_48.h5")
print("✅ Preprocessors saved as scaler.pkl and pca.pkl")
print("✅ Test data saved for evaluation")

# ---------------- STEP 9: EVALUATE ----------------
print("\n📊 Evaluating model on test data...")
y_preds = model.predict(X_test, batch_size=8, verbose=1)
y_eng_pred = np.argmax(y_preds[0], axis=1)
y_bor_pred = np.argmax(y_preds[1], axis=1)
y_conf_pred = np.argmax(y_preds[2], axis=1)
y_fru_pred = np.argmax(y_preds[3], axis=1)

eng_acc = accuracy_score(y_eng_test, y_eng_pred)
bor_acc = accuracy_score(y_bor_test, y_bor_pred)
conf_acc = accuracy_score(y_conf_test, y_conf_pred)
fru_acc = accuracy_score(y_fru_test, y_fru_pred)

print("\n" + "="*60)
print("🎯 FINAL RESULTS")
print("="*60)
print(f"✅ Engagement Accuracy:  {eng_acc:.4f}")
print(f"✅ Boredom Accuracy:     {bor_acc:.4f}")
print(f"✅ Confusion Accuracy:   {conf_acc:.4f}")
print(f"✅ Frustration Accuracy: {fru_acc:.4f}")

overall_acc = np.mean([eng_acc, bor_acc, conf_acc, fru_acc])
print(f"\n🔥 Overall Accuracy (avg of 4 tasks): {overall_acc:.4f}")

print("\n" + "="*40)
print("📈 DETAILED CLASSIFICATION REPORTS")
print("="*40)

print("\n📊 Classification Report for Engagement:")
print(classification_report(y_eng_test, y_eng_pred))

print("\n📊 Classification Report for Boredom:")
print(classification_report(y_bor_test, y_bor_pred))

print("\n📊 Classification Report for Confusion:")
print(classification_report(y_conf_test, y_conf_pred))

print("\n📊 Classification Report for Frustration:")
print(classification_report(y_fru_test, y_fru_pred))

print("\n🎉 TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")