"""
TERM PROJECT SUBMISSION
Computer Vision Course - Image Classification with Transfer Learning

Student Information:
- Name: [Öğrenci Adı]
- Student ID: [Öğrenci Numarası]
- Course: Computer Vision
- Date: 2025

PROMPTS USED:
1. "You are an AI coding assistant. Your task is to generate a full, production-ready 
   implementation of the term project described in the two PDF files..."
2. [Diğer kullanılan promptlar buraya eklenecek]

Dataset Structure:
/data
    /fracture
        img1.jpg
        img2.jpg
        ...
    /normal
        imgA.jpg
        imgB.jpg
        ...

Methodology:
- Transfer Learning with MobileNetV2 (ImageNet weights)
- 10-fold Stratified Cross-Validation
- Each fold trains a fresh model
- Training must complete under 5 minutes per fold on Google Colab GPU
- Includes: data preprocessing, tf.data pipeline, augmentation, 
  early stopping, reduce LR on plateau, model checkpoint
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications, callbacks, optimizers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import time
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU memory growth error: {e}")

# Configuration
DATA_DIR = '/content/data'
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
NUM_CLASSES = 2
EPOCHS = 50  # Early stopping will prevent overfitting
PATIENCE = 5
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Output file
RESULTS_FILE = 'cv_results.txt'

def log_to_file(message, file=RESULTS_FILE):
    """Write message to both console and file"""
    print(message)
    with open(file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

def load_data(data_dir):
    """
    Load image paths and labels from data directory
    
    Args:
        data_dir: Path to data directory containing 'fracture' and 'normal' subdirectories
        
    Returns:
        image_paths: List of image file paths
        labels: List of labels (0 for normal, 1 for fracture)
    """
    image_paths = []
    labels = []
    
    # Load fracture images (label = 1)
    fracture_dir = os.path.join(data_dir, 'fracture')
    if os.path.exists(fracture_dir):
        for filename in os.listdir(fracture_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(fracture_dir, filename))
                labels.append(1)
    
    # Load normal images (label = 0)
    normal_dir = os.path.join(data_dir, 'normal')
    if os.path.exists(normal_dir):
        for filename in os.listdir(normal_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(normal_dir, filename))
                labels.append(0)
    
    return np.array(image_paths), np.array(labels)

def preprocess_image(image_path, label):
    """
    Load and preprocess a single image
    
    Args:
        image_path: Path to image file
        label: Image label
        
    Returns:
        Preprocessed image tensor and label
    """
    # Read image file
    image_string = tf.io.read_file(image_path)
    # Decode image
    image = tf.image.decode_image(image_string, channels=3)
    # Convert to float32 and normalize to [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Resize image
    image = tf.image.resize(image, IMG_SIZE)
    # MobileNetV2 expects images in [-1, 1] range
    image = (image - 0.5) * 2.0
    return image, label

def augment_image(image, label):
    """
    Apply data augmentation to image
    
    Args:
        image: Image tensor
        label: Image label
        
    Returns:
        Augmented image and label
    """
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    # Random rotation (small angle)
    image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    # Random brightness adjustment
    image = tf.image.random_brightness(image, max_delta=0.2)
    # Random contrast adjustment
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # Ensure values stay in valid range
    image = tf.clip_by_value(image, -1.0, 1.0)
    return image, label

def create_dataset(image_paths, labels, batch_size, shuffle=True, augment=False):
    """
    Create tf.data.Dataset from image paths and labels
    
    Args:
        image_paths: Array of image file paths
        labels: Array of labels
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset
        augment: Whether to apply data augmentation
        
    Returns:
        tf.data.Dataset
    """
    # Create dataset from paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    # Shuffle if needed
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))
    
    # Load and preprocess images
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply augmentation if needed (only for training)
    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    # Prefetch for better performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def create_model():
    """
    Create MobileNetV2-based model for transfer learning
    
    Returns:
        Compiled Keras model
    """
    # Load MobileNetV2 base model (pre-trained on ImageNet)
    base_model = applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    
    # Preprocessing (already done, but keeping for consistency)
    x = base_model(inputs, training=False)
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Add dropout for regularization
    x = layers.Dropout(0.2)(x)
    
    # Dense layer
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Output layer
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_fold(fold_num, train_dataset, val_dataset, num_train, num_val):
    """
    Train model for a single fold
    
    Args:
        fold_num: Fold number (0-9)
        train_dataset: Training dataset
        val_dataset: Validation dataset
        num_train: Number of training samples
        num_val: Number of validation samples
        
    Returns:
        Test accuracy for this fold
    """
    log_to_file(f"\n{'='*60}")
    log_to_file(f"FOLD {fold_num + 1}/10")
    log_to_file(f"{'='*60}")
    log_to_file(f"Training samples: {num_train}")
    log_to_file(f"Validation samples: {num_val}")
    
    # Create fresh model for this fold
    model = create_model()
    
    # Callbacks
    callbacks_list = [
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate on plateau
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        # Model checkpoint
        callbacks.ModelCheckpoint(
            filepath=f'model_fold_{fold_num + 1}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0
        )
    ]
    
    # Train model
    start_time = time.time()
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate(val_dataset, verbose=0)
    
    log_to_file(f"\nFold {fold_num + 1} Results:")
    log_to_file(f"  Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    log_to_file(f"  Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    log_to_file(f"  Validation Loss: {val_loss:.4f}")
    
    # Get predictions for detailed metrics
    y_true = []
    y_pred = []
    
    for images, labels in val_dataset:
        predictions = model.predict(images, verbose=0, batch_size=BATCH_SIZE)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate accuracy
    fold_accuracy = accuracy_score(y_true, y_pred)
    
    # Clean up model from memory
    del model
    tf.keras.backend.clear_session()
    
    log_to_file(f"  Calculated Accuracy: {fold_accuracy:.4f} ({fold_accuracy*100:.2f}%)")
    
    return fold_accuracy

def main():
    """
    Main function to run 10-fold cross-validation
    """
    # Clear results file
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
    
    # Clean up old model files
    for i in range(1, 11):
        model_file = f'model_fold_{i}.h5'
        if os.path.exists(model_file):
            os.remove(model_file)
    
    log_to_file("="*60)
    log_to_file("10-FOLD STRATIFIED CROSS-VALIDATION")
    log_to_file("Image Classification: Fracture vs Normal")
    log_to_file("="*60)
    log_to_file(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_to_file(f"Batch size: {BATCH_SIZE}")
    log_to_file(f"Image size: {IMG_SIZE}")
    log_to_file(f"Learning rate: {LEARNING_RATE}")
    log_to_file(f"Max epochs: {EPOCHS}")
    log_to_file(f"Early stopping patience: {PATIENCE}")
    
    # Load data
    log_to_file("\nLoading data...")
    image_paths, labels = load_data(DATA_DIR)
    
    if len(image_paths) == 0:
        log_to_file("ERROR: No images found in data directory!")
        log_to_file(f"Expected structure: {DATA_DIR}/fracture/ and {DATA_DIR}/normal/")
        return
    
    log_to_file(f"Total images loaded: {len(image_paths)}")
    log_to_file(f"  Fracture images: {np.sum(labels == 1)}")
    log_to_file(f"  Normal images: {np.sum(labels == 0)}")
    
    # 10-fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    fold_accuracies = []
    total_start_time = time.time()
    
    for fold_num, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
        # Split data
        train_paths = image_paths[train_idx]
        train_labels = labels[train_idx]
        val_paths = image_paths[val_idx]
        val_labels = labels[val_idx]
        
        # Create datasets
        train_dataset = create_dataset(
            train_paths, train_labels, 
            BATCH_SIZE, shuffle=True, augment=True
        )
        val_dataset = create_dataset(
            val_paths, val_labels, 
            BATCH_SIZE, shuffle=False, augment=False
        )
        
        # Train model for this fold
        fold_accuracy = train_fold(
            fold_num, train_dataset, val_dataset,
            len(train_paths), len(val_paths)
        )
        
        fold_accuracies.append(fold_accuracy)
    
    total_time = time.time() - total_start_time
    
    # Calculate statistics
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    
    # Final results
    log_to_file("\n" + "="*60)
    log_to_file("FINAL RESULTS")
    log_to_file("="*60)
    log_to_file(f"\nFold Accuracies:")
    for i, acc in enumerate(fold_accuracies):
        log_to_file(f"  Fold {i+1}: {acc:.4f} ({acc*100:.2f}%)")
    
    log_to_file(f"\nOverall Statistics:")
    log_to_file(f"  Mean Accuracy: {mean_accuracy:.4f} ({mean_accuracy*100:.2f}%)")
    log_to_file(f"  Std Deviation: {std_accuracy:.4f} ({std_accuracy*100:.2f}%)")
    log_to_file(f"  Mean ± Std: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    log_to_file(f"  Mean ± Std (%): {mean_accuracy*100:.2f}% ± {std_accuracy*100:.2f}%")
    
    log_to_file(f"\nTotal training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    log_to_file(f"Average time per fold: {total_time/10:.2f} seconds ({total_time/10/60:.2f} minutes)")
    log_to_file(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_to_file("="*60)
    
    print("\n" + "="*60)
    print("CROSS-VALIDATION COMPLETE!")
    print("="*60)
    print(f"Results saved to: {RESULTS_FILE}")
    print(f"Mean Accuracy: {mean_accuracy*100:.2f}% ± {std_accuracy*100:.2f}%")

if __name__ == "__main__":
    # Check GPU availability
    if tf.config.list_physical_devices('GPU'):
        log_to_file("GPU detected and available!")
        log_to_file(f"GPU: {tf.config.list_physical_devices('GPU')[0]}")
    else:
        log_to_file("WARNING: No GPU detected. Training may be slow.")
    
    main()

