## Project Overview: Intel Image Classification

This project aims to classify images into different categories using the Intel Image Classification dataset. Two distinct deep learning models, a custom Convolutional Neural Network (CNN) and a transfer learning approach using MobileNetV2, were developed, trained, and evaluated for this task.

### 1. Dataset Acquisition and Exploration

*   **Data Source:** The `intel-image-classification` dataset was downloaded using KaggleHub. The dataset consists of segmented images categorized into 'buildings', 'forest', 'glacier', 'mountain', 'sea', and 'street'.
*   **Data Structure:** The dataset is organized into training (`seg_train`), testing (`seg_test`), and prediction (`seg_pred`) directories.
*   **Class Distribution:** An initial exploration revealed the distribution of images across the six classes in the training, validation, and test sets. The distribution appears relatively balanced, which is beneficial for training unbiased models.
*   **Sample Images:** Visualizations of sample training images were provided to give an insight into the data.

### 2. Data Preprocessing and Augmentation

*   **Image Sizing:** All images were resized to 150x150 pixels.
*   **Batching:** Images were processed in batches of 32.
*   **Data Augmentation (Training Data):** To enhance model generalization and prevent overfitting, the training data underwent augmentation techniques such as rescaling, zooming, width/height shifting, and fill mode nearest.
*   **Validation and Test Data:** Validation and test data were only rescaled to match the input format of the models, without augmentation.
*   **Data Generators:** `ImageDataGenerator` from Keras was used to efficiently load and preprocess images, creating `train_generator`, `val_generator`, and `test_generator`.

### 3. Model Development

#### A. Custom Convolutional Neural Network (CNN)

*   **Architecture:** A sequential CNN model was built with multiple `Conv2D` layers (128, 64, 32 filters respectively), `BatchNormalization` layers, `MaxPooling2D` for downsampling, a `Flatten` layer, and `Dense` layers (256 units) with `Dropout` for regularization. The output layer uses `softmax` activation for multi-class classification.
*   **Compilation:** The model was compiled using the `Adam` optimizer with a learning rate of 0.001 and `categorical_crossentropy` as the loss function, with `accuracy` as the metric.

#### B. MobileNetV2 (Transfer Learning)

*   **Base Model:** MobileNetV2, a pre-trained convolutional neural network, was used as a feature extractor. The `include_top=False` argument ensures that the classification head of MobileNetV2 is not included, allowing for custom classification layers to be added.
*   **Feature Extraction:** The `base_model` (MobileNetV2) layers were set to `trainable=False` to freeze their weights, utilizing its learned features without further modification during initial training.
*   **Custom Head:** A custom classification head was added, consisting of a `GlobalAveragePooling2D` layer, a `Dense` layer (256 units) with `relu` activation and `Dropout` (0.5), and a final `Dense` layer with `softmax` activation for the 6 output classes.
*   **Compilation:** Similar to the CNN, this model was also compiled with the `Adam` optimizer (learning rate 0.001), `categorical_crossentropy` loss, and `accuracy` metric.

### 4. Training and Evaluation

*   **Training Process:** Both models were trained for 50 epochs using their respective data generators.
*   **Callbacks:**
    *   `EarlyStopping`: Monitors validation loss and stops training if it doesn't improve for 5 consecutive epochs, restoring the best weights.
    *   `ReduceLROnPlateau`: Reduces the learning rate by a factor of 0.2 if the validation loss does not improve for 5 consecutive epochs.
*   **Performance Metrics:** Training and validation accuracy and loss were plotted to visualize model learning curves.
*   **Test Set Evaluation:** Both models were evaluated on the unseen `test_generator` to assess their generalization performance.
    *   `cnn_model` achieved a Test Accuracy of **0.8407** and a Test Loss of **0.4554**.
    *   `mobilenet_model` achieved a Test Accuracy of **0.9010** and a Test Loss of **0.2673**.
*   **Prediction Visualizations:** Sample predictions on the test dataset were visualized for both models.
*   **Confusion Matrix and Classification Report:** Detailed evaluation reports, including confusion matrices and classification reports (precision, recall, f1-score per class), were generated for both models to provide deeper insights into their performance.

### 5. Model Comparison and Conclusion

*   **MobileNetV2 Superiority:** The MobileNetV2 model, leveraging transfer learning, demonstrated significantly better performance with a higher test accuracy (90.10%) and lower test loss compared to the custom CNN model (84.07%). This highlights the effectiveness of pre-trained models for image classification tasks, especially when data might be limited or computational resources are constrained.
*   **Model Saving:** Both trained models (`cnn_model.keras` and `mobilenet_model.keras`) were saved to Google Drive for future use, demonstrating the process of persisting trained models.

Here all result
## CNN Model Result

| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| buildings  | 0.86 | 0.76 | 0.81 | 437 |
| forest     | 0.93 | 0.98 | 0.96 | 474 |
| glacier    | 0.82 | 0.79 | 0.80 | 553 |
| mountain   | 0.77 | 0.81 | 0.79 | 525 |
| sea        | 0.88 | 0.79 | 0.83 | 510 |
| street     | 0.81 | 0.91 | 0.86 | 501 |

### Overall Metrics

| Metric | Score |
|--------|------|
| Accuracy | 0.84 |
| Macro Avg Precision | 0.84 |
| Macro Avg Recall | 0.84 |
| Macro Avg F1-Score | 0.84 |
| Weighted Avg Precision | 0.84 |
| Weighted Avg Recall | 0.84 |
| Weighted Avg F1-Score | 0.84 |
| Total Support | 3000 |

## Mobilenet Model Result

| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| buildings  | 0.93 | 0.89 | 0.91 | 437 |
| forest     | 0.99 | 0.99 | 0.99 | 474 |
| glacier    | 0.85 | 0.83 | 0.84 | 553 |
| mountain   | 0.83 | 0.83 | 0.83 | 525 |
| sea        | 0.94 | 0.93 | 0.93 | 510 |
| street     | 0.90 | 0.96 | 0.92 | 501 |

### Overall Metrics

| Metric | Score |
|--------|------|
| Accuracy | 0.90 |
| Macro Avg Precision | 0.90 |
| Macro Avg Recall | 0.90 |
| Macro Avg F1-Score | 0.90 |
| Weighted Avg Precision | 0.90 |
| Weighted Avg Recall | 0.90 |
| Weighted Avg F1-Score | 0.90 |
| Total Support | 3000 |

In conclusion, this project successfully implemented and compared two image classification approaches on the Intel Image Classification dataset. The transfer learning approach with MobileNetV2 proved to be more effective, achieving high accuracy in categorizing environmental images.
