# 🍎 Fresh vs Rotten Fruit Classification

A deep learning project that uses a Convolutional Neural Network (CNN) to classify fruit images as **Fresh** or **Rotten**.

The project uses **Transfer Learning with MobileNetV2**, a pretrained CNN, as a feature extractor. The base model is frozen, while a custom classification head is trained for the two fruit-quality classes.

## 🧠 Model Architecture

- MobileNetV2 pretrained on ImageNet
- Data augmentation:
  - Random Flip
  - Random Rotation
  - Random Zoom
- Global Average Pooling
- Dense layer with 128 neurons
- Dropout (0.3)
- Softmax output layer with 2 classes

## ⚙️ Training

The model is trained using:
- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Loss:** Sparse Categorical Crossentropy
- **Batch Size:** 32
- **Maximum Epochs:** 30
- **Early Stopping:** Monitors validation loss

## 🔄 Training Process

```mermaid
flowchart TD
    A["model.fit()"] --> B["Training begins"]
    B --> C["X_train"]
    B --> D["y_train"]
    C --> E["Model learns"]
    D --> E
    E --> F["Epoch 1"]
    F --> G["Validation data"]
    G --> H["Check val_loss"]
    H --> I["Epoch 2"]
    I --> J["..."]
    J --> K{"EarlyStopping?"}
    K -->|No| L["Continue training"]
    L --> F
    K -->|Yes| M["STOP"]
