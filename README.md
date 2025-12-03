# Handwritten Digit Classifier (MNIST)

This project builds and analyzes multiple Convolutional Neural Network (CNN) models to classify handwritten digits using the MNIST dataset. The goal was to explore deep learning architecture design, evaluate model performance, and compare the effects of regularization, model size, and training strategies.

## 📌 Project Overview

Handwritten digit recognition is easy for humans but highly nonlinear for machines. Traditional algorithms struggle due to variations in handwriting, stroke thickness, rotation, and noise. CNNs, however, excel at extracting spatial features and generalizing across handwriting styles.

This project constructs **three different CNN models**, each increasingly refined:

1. **CNN #1** — Minimal architecture (1 conv layer + dense)
2. **CNN #2** — Deep architecture (3 conv layers + multiple dense layers)
3. **CNN #3** — Balanced architecture with **regularization** (pooling + dropout)

The final model achieves **>99% accuracy on the MNIST test set**.

---

## 📊 Dataset

Source: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data

- **70,000 images** of handwritten digits
- **60,000 training** examples
- **10,000 test** examples
- Grayscale
- 28×28 pixels
- Labels: digits 0–9

---

## 🧪 Exploratory Data Analysis (EDA)

The notebook includes:
- Visualization of sample digits
- Histogram of class distribution (dataset is balanced)
- Inspection of pixel value ranges
- Normalization (0–1 scaling)

---

## 🧠 Model Architectures

### **CNN #1 — Simple Baseline**
- 1 Conv2D layer  
- Flatten → Dense(10, softmax)  
- Fastest, but limited expressive power

### **CNN #2 — Deep Network**
- 3 Conv2D layers (32, 64, 128 filters)
- 2 Dense layers (64, 32 units)
- Slowest and offered no improvement over Model 1

### **CNN #3 — Regularized CNN (Final Model)**
- Conv2D → MaxPooling  
- Conv2D → MaxPooling  
- Dense → Dropout  
- Dense(10, softmax)  
- Best-performing model  
- Achieved **0.992 test accuracy**

---

## 📈 Training & Evaluation

All models were trained using:

- **Adam optimizer**
- **Sparse categorical crossentropy**
- **Accuracy metric**
- 10 epochs
- Batch size: 128

Plotting functions visualize:
- Train vs validation loss
- Train vs validation accuracy

### **Confusion matrix**  
A confusion matrix is generated using scikit-learn to evaluate per-class performance and detect misclassifications.

---

## 🧐 Key Learnings & Takeaways

### ✔ No overfitting occurred in any model
Training and test metrics stayed closely aligned throughout training.

### ✔ Regularization still improved performance  
Even without visible overfitting, pooling and dropout helped the model focus on the most important image features and push test accuracy above 99%.

### ✔ Model size alone doesn't guarantee better accuracy  
The deeper Model 2 trained slower and didn't outperform the smaller baseline.

### ✔ Training for many epochs was unnecessary  
Performance plateaued around **epoch 6**, so longer training offered little benefit.

---

## 🚀 Future Improvements

- Add **Batch Normalization** for more stable training
- Use **data augmentation** to increase dataset variability
- Try **learning rate schedules** (cosine decay, ReduceLROnPlateau)
- Explore stronger architectures such as:
  - LeNet-5  
  - ResNet-style mini blocks  
  - MobileNet-like efficient CNNs  

---

## 📚 Repository Contents

- `handwritten_digit_classifier.ipynb` — Full EDA, modeling, evaluation
- `README.md` — Project summary and documentation

---

## 🏁 Final Result

The final model achieved:

- **Training accuracy:** 0.9928  
- **Test accuracy:** 0.9922  
- Strong generalization  
- Best performance among all tested architectures

This project demonstrates a complete deep-learning workflow, from dataset loading and EDA to model building, hyperparameter exploration, and performance evaluation.
