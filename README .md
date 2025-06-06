#  Oxford-IIIT Pet Segmentation with U-Net

This project demonstrates how to build and train a **U-Net architecture** for **semantic image segmentation** on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/). The task is to segment pet images into classes such as pet, background, and outline.

##  Overview

-  **Dataset:** Oxford-IIIT Pet Dataset (via TensorFlow Datasets)
-  **Model:** U-Net (fully convolutional network for pixel-wise segmentation)
-  **Objective:** Predict segmentation masks that classify each pixel into `pet`, `background`, or `outline`.

##  Technologies Used

- Python
- TensorFlow / Keras
- TensorFlow Datasets (`tfds`)
- NumPy
- Matplotlib

##  Dataset Details

The segmentation masks in the dataset are originally labeled as:
- `1` - foreground (pet)
- `2` - background
- `3` - not classified

These are normalized to:
- `0` - pet
- `1` - background
- `2` - outline

##  Preprocessing Steps

- Resize images to `128x128`
- Normalize pixel values to `[0, 1]`
- Random left-right flipping for augmentation
- Adjust mask labels from [1,2,3] → [0,1,2]

##  Model Architecture

The U-Net model consists of:

- **Encoder**: 4 convolutional blocks with downsampling
- **Bottleneck**: Deepest convolutional layer
- **Decoder**: 4 upsampling blocks with skip connections
- **Output Layer**: 3-channel softmax layer (for 3 mask classes)

> Loss Function: `SparseCategoricalCrossentropy`  
> Optimizer: `Adam`  
> Metrics: `Accuracy`

##  Training

```python
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(train_dataset,
          epochs=10,
          steps_per_epoch=TRAIN_LENGTH // BATCH_SIZE,
          validation_data=test_dataset,
          validation_steps=VALIDATION_STEPS)



