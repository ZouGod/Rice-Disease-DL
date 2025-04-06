```markdown
# Rice Disease Recognition - Deep Learning Model

This repository contains the deep learning models for detecting and classifying rice diseases from leaf images.
The system uses convolutional neural networks (CNNs) to achieve high-accuracy disease identification.

```
## 🧠 Model Architecture

### Implemented Models:
1. **EfficientNet-B0** (95% accuracy)
2. **DenseNet-121** (92% accuracy) 
3. **MobileNetV2** (87% accuracy)
4. **Ensemble Model** (96% accuracy) - Combines predictions from all three models

### Model Specifications:
- Input size: 224×224 pixels (RGB)
- Output classes: 5 common rice diseases + healthy class
- Training dataset: 17,500 labeled images
- Augmentation: Rotation, flipping, brightness adjustment

## 🚀 How to Run the Models

### Prerequisites:
- Python 3.8+
- TensorFlow 2.10+
- Keras
- OpenCV
- NumPy

### Installation:
```bash
pip install tensorflow keras opencv-python numpy

### Running the Models:

1. **Clone the repository:**
```bash
git clone https://github.com/ZouGod/Rice-Disease-DL.git
cd Rice-Disease-DL/models
```

2. **For single model prediction:**
```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load model
model = load_model('efficientnet_b0_model.h5') 

# Preprocess image
img = cv2.imread('test_leaf.jpg')
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Make prediction
prediction = model.predict(img)
class_idx = np.argmax(prediction)
confidence = np.max(prediction)

print(f"Predicted class: {class_idx} with {confidence*100:.2f}% confidence")
```

3. **For ensemble prediction:**
```python
from ensemble_model import RiceDiseaseEnsemble

ensemble = RiceDiseaseEnsemble(
    efficientnet_path='efficientnet_b0_model.h5',
    densenet_path='densenet121_model.h5',
    mobilenet_path='mobilenetv2_model.h5'
)

prediction, confidence = ensemble.predict('test_leaf.jpg')
print(f"Ensemble prediction: {prediction} ({confidence*100:.2f}%)")
```

## 🏋️ Training the Models

1. **Prepare dataset:**
- Organize images in this structure:
```
dataset/
    ├── healthy/
    ├── bacterial_blight/ 
    ├── brown_spot/
    ├── leaf_blast/
    ├── leaf_scald/
    └── tungro/
```

2. **Run training:**
```bash
python train.py --model efficientnet --epochs 50 --batch_size 32 --data_path ./dataset
```

Available training arguments:
- `--model`: [efficientnet|densenet|mobilenet]
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--learning_rate`: Initial learning rate
- `--data_aug`: Enable data augmentation

## 📊 Performance Metrics

| Model | Accuracy | Precision | Recall | Inference Time |
|-------|----------|-----------|--------|----------------|
| EfficientNet-B0 | 95% | 0.94 | 0.95 | 120ms |
| DenseNet-121 | 92% | 0.91 | 0.92 | 150ms | 
| MobileNetV2 | 87% | 0.86 | 0.87 | 80ms |
| Ensemble | 96% | 0.95 | 0.96 | 300ms |

## 📁 Model Files
- `efficientnet_b0_model.h5` - Pretrained EfficientNet
- `densenet121_model.h5` - Pretrained DenseNet
- `mobilenetv2_model.h5` - Pretrained MobileNet
- `ensemble_model.py` - Ensemble model implementation
- `train.py` - Training script
- `utils.py` - Helper functions

## 📜 License
MIT License - See LICENSE for details
```

This README focuses specifically on:
1. The deep learning model architectures
2. How to run predictions with the models
3. How to train the models
4. Performance metrics
5. Required files and dependencies
