# 🍎 Fresh vs Rotten Fruit & Vegetable Classification

A deep learning project that uses a Convolutional Neural Network (CNN) to classify fruit and vegetable images as **fresh** or **rotten**, and identifies the produce type (apple, banana, tomato, etc.).

The project uses **Transfer Learning with MobileNetV2**, a pretrained CNN, as a feature extractor. The base model is frozen, while a custom classification head is trained on top.

## 📊 Dataset

[Fresh and Stale Classification](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification) (Kaggle) — ~30,000 images across 9 produce types (apples, banana, bittergourd, capsicum, cucumber, okra, oranges, potato, tomato), each labeled fresh or rotten, giving **18 total classes**.

## 🍒 Model Architecture

- MobileNetV2 pretrained on ImageNet (frozen)
- Data augmentation:
  - Random Flip
  - Random Rotation
  - Random Zoom
- Global Average Pooling
- Dense layer with 128 neurons
- Dropout (0.3)
- Softmax output layer with **18 classes** (9 produce types × fresh/rotten)

## ⚙️ Training

The model is trained using:

- Optimizer: Adam
- Learning Rate: 0.001
- Loss: Sparse Categorical Crossentropy
- Batch Size: 32
- Maximum Epochs: 30
- Early Stopping: Monitors validation loss

### 🔄 Training Process

*(diagram)*

### Fine-tuning experiment

Unfreezing the last 30 layers of MobileNetV2 for fine-tuning was attempted, using a lower learning rate (1e-5). This increased validation loss and widened the train/validation accuracy gap — a sign of overfitting relative to the dataset size — so the frozen-base model was kept as the final version.

## 📈 Results

Achieved **~97% accuracy** on a held-out test set never seen during training or validation, with balanced precision/recall across classes.

## 🖥️ Web App

A Streamlit app (`app.py`) loads the trained model and predicts on any uploaded photo, returning both the produce type and freshness with a confidence score.

### Running locally

\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`

Then open the local URL Streamlit prints (usually `http://localhost:8501`), upload a photo, and click **Predict**.

## 📁 Project Structure

\`\`\`
├── app.py                      # Streamlit web app
├── final_produce_model.keras   # Trained model
├── requirements.txt            # Python dependencies
├── .python-version              # Pinned Python version
├── .streamlit/
│   └── config.toml              # App theme and settings
├── notebook/                    # Training notebook
└── README.md
\`\`\`

## 🛠️ Tech Stack

- TensorFlow / Keras
- OpenCV
- scikit-learn
- Streamlit
- MobileNetV2 (transfer learning)

## 🚀 Future Improvements

- Localize *where* on the fruit the rotten spot is (e.g. Grad-CAM heatmaps or a segmentation model), rather than only classifying the whole image.
- Expand to more produce types.
