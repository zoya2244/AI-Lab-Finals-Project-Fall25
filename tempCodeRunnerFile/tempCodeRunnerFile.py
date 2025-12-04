import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ======================
# CONFIG - UPDATE THIS!
# ======================
DATA_DIR = r"C:\Users\Zoya\Desktop\realorfake detection\dataset"      # 👈 Point to folder containing 'real/' and 'ai/'
IMG_SIZE = (128, 128)          # Smaller = faster + less memory
BATCH_SIZE = 16                # Safe for 8GB RAM
EPOCHS = 30                    # More epochs help small data

# ======================
# DATA GENERATOR
# ======================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    # Strong augmentation to prevent overfitting
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print("Classes:", train_gen.class_indices)  # Should show {'ai': 0, 'real': 1}

# ======================
# BUILD MODEL (Small CNN + Dropout to fight overfitting)
# ======================
model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.GlobalAveragePooling2D(),  # Better than Flatten for small data
    layers.Dropout(0.6),              # High dropout to prevent overfitting
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ======================
# TRAIN
# ======================
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    verbose=1
)

# ======================
# SAVE MODEL
# ======================
model.save("ai_detector_1100.h5")
print("\n✅ Model saved as 'ai_detector_1100.h5'")

# ======================
# PLOT TRAINING CURVES
# ======================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ======================
# PREDICTION FUNCTION (Reusable!)
# ======================
def predict_image(img_path, model_path="ai_detector_1100.h5"):
    from tensorflow.keras.preprocessing import image
    import numpy as np
    
    # Load model if not already loaded
    if isinstance(model_path, str):
        model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    pred = model.predict(img_array, verbose=0)[0][0]
    
    # Interpret: since flow_from_directory sorts classes alphabetically:
    # 'ai' = 0, 'real' = 1
    if pred > 0.5:
        return f"✅ REAL photo (confidence: {pred:.2%})"
    else:
        return f"❌ AI-GENERATED (confidence: {1 - pred:.2%})"

# ======================
# EXAMPLE USAGE
# ======================
# Uncomment to test on your own image
# result = predict_image("test_image.jpg")
# print(result)