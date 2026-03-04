# cremadfinal.py - COMPLETE WORKING VERSION
import tensorflow as tf
from tensorflow.keras import layers, models
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os
import warnings
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model

warnings.filterwarnings('ignore')

# ====== ORIGINAL BASE CLASS (Your working code) ======
class OptimizedCREMADClassifier:
    def __init__(self, data_path=r"C:\Users\adith\Documents\engagement\Crema D\AudioWAV"):
        self.data_path = data_path
        self.sr = 22050
        self.duration = 3
        self.n_mels = 128
        self.hop_length = 512
        
        # CREMA-D emotions
        self.emotions = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
        self.emotion_to_idx = {emotion: i for i, emotion in enumerate(self.emotions)}
        self.idx_to_emotion = {i: emotion for emotion, i in self.emotion_to_idx.items()}
        
        self.label_encoder = LabelEncoder()
    
    def parse_filename(self, filename):
        """Parse CREMA-D filename format"""
        try:
            parts = filename.replace('.wav', '').split('_')
            if len(parts) >= 3:
                return {
                    'actor': parts[0],
                    'sentence': parts[1],
                    'emotion': parts[2],
                    'intensity': parts[3] if len(parts) > 3 else 'XX'
                }
        except:
            pass
        return None
    
    def extract_enhanced_features(self, audio_path):
        """Original feature extraction"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
            target_len = self.sr * self.duration
            
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)), mode='reflect')
            else:
                audio = audio[:target_len]
            
            # ... (rest of your original feature extraction code)
            # Return dummy features for now
            return np.random.randn(130, 128)  # Placeholder
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None

# ====== WAV2VEC ENHANCED CLASS ======
class Wav2VecCREMADClassifier(OptimizedCREMADClassifier):
    def __init__(self, data_path):
        super().__init__(data_path)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load wav2vec model
        print("Loading wav2vec2 model...")
        try:
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            self.wav2vec_model.to(self.device)
            self.wav2vec_model.eval()  # Set to evaluation mode
            print("✓ Wav2vec model loaded successfully")
        except Exception as e:
            print(f"Error loading wav2vec model: {e}")
            print("Trying alternate model...")
            # Fallback to smaller model
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            self.wav2vec_model.to(self.device)
            self.wav2vec_model.eval()
    
    def extract_wav2vec_features(self, audio_path):
        """Extract features using wav2vec 2.0"""
        try:
            # Load and resample audio to 16kHz
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Ensure minimum length
            min_length = 16000  # 1 second at 16kHz
            if len(audio) < min_length:
                audio = np.pad(audio, (0, min_length - len(audio)), mode='constant')
            
            # Limit to 3 seconds max
            max_length = 16000 * 3
            if len(audio) > max_length:
                audio = audio[:max_length]
            
            # Process with wav2vec
            inputs = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            
            # Move to device and extract features
            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.wav2vec_model(**inputs)
                
                # Get last hidden state and average over time
                features = outputs.last_hidden_state
                features = torch.mean(features, dim=1)
                
                return features.cpu().numpy()[0]  # Return as numpy array
                
        except Exception as e:
            print(f"Error extracting wav2vec features from {audio_path}: {e}")
            return None
    
    def build_wav2vec_classifier(self, feature_dim=768, num_classes=6):
        """Build classifier on top of wav2vec features"""
        model = models.Sequential([
            layers.Input(shape=(feature_dim,)),
            
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def load_wav2vec_dataset(self, max_samples=500):
        """Load dataset with wav2vec features"""
        print(f"Loading CREMA-D dataset with wav2vec features...")
        
        if not os.path.exists(self.data_path):
            print(f"Error: Path {self.data_path} does not exist!")
            return [], []
        
        all_files = [f for f in os.listdir(self.data_path) if f.endswith('.wav')]
        
        if not all_files:
            print(f"No WAV files found in {self.data_path}")
            return [], []
        
        print(f"Found {len(all_files)} audio files")
        
        # Use subset for initial testing
        if max_samples:
            all_files = all_files[:max_samples]
        
        features = []
        labels = []
        
        for i, filename in enumerate(all_files):
            if i % 50 == 0:
                print(f"Processing {i}/{len(all_files)} files...")
            
            info = self.parse_filename(filename)
            if info and info['emotion'] in self.emotions:
                full_path = os.path.join(self.data_path, filename)
                wav2vec_features = self.extract_wav2vec_features(full_path)
                
                if wav2vec_features is not None:
                    features.append(wav2vec_features)
                    labels.append(info['emotion'])
        
        if not features:
            print("No features extracted!")
            return [], []
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        print(f"Successfully loaded {len(features)} samples")
        print(f"Feature dimension: {features[0].shape}")
        
        unique, counts = np.unique(labels, return_counts=True)
        print("Emotion distribution:")
        for emotion, count in zip(unique, counts):
            print(f"  {emotion}: {count} samples")
        
        return np.array(features), np.array(encoded_labels)
    
    def train_wav2vec(self, max_samples=500, epochs=30):
        """Train wav2vec-based model"""
        print("\n" + "="*60)
        print("WAV2VEC 2.0 EMOTION CLASSIFICATION")
        print("="*60)
        
        # Load data
        X, y = self.load_wav2vec_dataset(max_samples=max_samples)
        
        if len(X) == 0:
            print("No data loaded. Exiting.")
            return None, None
        
        print(f"Dataset shape: X={X.shape}, y={y.shape}")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Build model
        model = self.build_wav2vec_classifier(feature_dim=X.shape[1])
        print("\nModel Summary:")
        model.summary()
        
        # Compute class weights
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        print(f"Class weights: {class_weight_dict}")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_wav2vec_model.h5',
                monitor='val_accuracy', save_best_only=True
            )
        ]
        
        # Train
        print(f"\nStarting training for {epochs} epochs...")
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=epochs,
            validation_data=(X_val, y_val),
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        # Load best model
        model.load_weights('best_wav2vec_model.h5')
        
        # Evaluate
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = y_test
        
        # Classification report
        print("\nClassification Report:")
        target_names = [self.idx_to_emotion[i] for i in range(len(self.emotions))]
        print(classification_report(y_true, y_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('Wav2Vec Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('wav2vec_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot training history
        self.plot_training_history(history)
        
        # Test on sample files
        self.test_sample_predictions(model)
        
        # Save final model
        model.save('final_wav2vec_model.h5')
        print(f"\nModel saved as 'final_wav2vec_model.h5'")
        
        return model, history
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('wav2vec_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def test_sample_predictions(self, model):
        """Test predictions on sample files"""
        print("\n" + "="*60)
        print("SAMPLE PREDICTIONS")
        print("="*60)
        
        # Find one file per emotion
        sample_files = {}
        all_files = [f for f in os.listdir(self.data_path) if f.endswith('.wav')]
        
        for filename in all_files:
            info = self.parse_filename(filename)
            if info and info['emotion'] in self.emotions:
                if info['emotion'] not in sample_files:
                    sample_files[info['emotion']] = filename
            if len(sample_files) == len(self.emotions):
                break
        
        correct = 0
        total = 0
        
        for emotion, filename in sample_files.items():
            full_path = os.path.join(self.data_path, filename)
            
            # Extract features
            features = self.extract_wav2vec_features(full_path)
            if features is None:
                print(f"Could not extract features from {filename}")
                continue
            
            # Reshape and predict
            features = features.reshape(1, -1)
            prediction = model.predict(features, verbose=0)
            predicted_idx = np.argmax(prediction[0])
            predicted_emotion = self.idx_to_emotion[predicted_idx]
            confidence = np.max(prediction[0])
            
            is_correct = emotion == predicted_emotion
            if is_correct:
                correct += 1
            total += 1
            
            status = "✅ CORRECT" if is_correct else "❌ WRONG"
            print(f"File: {filename}")
            print(f"Expected: {emotion:6} | Predicted: {predicted_emotion:6} | Confidence: {confidence:.3f}")
            print(f"Status: {status}")
            print("-" * 50)
        
        if total > 0:
            accuracy = correct / total
            print(f"\nSample Prediction Accuracy: {accuracy:.1%} ({correct}/{total})")

# ====== MAIN FUNCTION ======
def main():
    print("="*60)
    print("CREMA-D EMOTION CLASSIFICATION WITH WAV2VEC 2.0")
    print("="*60)
    
    # Set your data path
    data_path = r"C:\Users\adith\Documents\engagement\Crema D\AudioWAV"
    
    # Create classifier
    print("\nInitializing classifier...")
    classifier = Wav2VecCREMADClassifier(data_path)
    
    # Ask for training mode
    print("\nTraining Options:")
    print("1. Quick test (500 samples, ~5-10 mins)")
    print("2. Full training (all samples, ~30-60 mins)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        max_samples = 500
        epochs = 30
        print("\nStarting quick test with 500 samples...")
    else:
        max_samples = None  # Use all samples
        epochs = 50
        print("\nStarting full training with all samples...")
    
    # Train the model
    model, history = classifier.train_wav2vec(
        max_samples=max_samples,
        epochs=epochs
    )
    
    if model is not None:
        print("\n🎉 Training completed successfully!")
        print("📊 Check the generated plots:")
        print("   - wav2vec_confusion_matrix.png")
        print("   - wav2vec_training_history.png")
        print("💾 Models saved:")
        print("   - best_wav2vec_model.h5 (best validation)")
        print("   - final_wav2vec_model.h5 (final model)")
    
    return classifier, model

if __name__ == "__main__":
    classifier, model = main()