# 07-03

## Challenge 02 - [Optional] Dogs and Cats

![](https://images.unsplash.com/photo-1547623542-de3ff5941ddb?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1050&q=80)

Photo by [Ancaro Project](https://unsplash.com/photos/6VQlKJp2vpo)

## Guidelines

See Kaggle dataset dogs and cats [here](https://www.kaggle.com/c/dogs-vs-cats) and download the training dataset. Please do not look at the solutions, for obvious pedogogical reasons... but call the teacher instead.

### Data Preparation
We will use only a subset of the training dataset for computing resources reasons: 1000 cats + 1000 dogs images in train, 200 cats + 200 dogs in test.

First split by hand your dataset like the following:
```
- data/
--- train/
----- cats/
------- cats001.jpg
------- cats002.jpg
        ...
------ dogs/
------- dogs001.jpg
------- dogs002.jpg
        ...
--- test/
----- cats/
------- cats001.jpg
------- cats002.jpg
        ...
----- dogs/
------- dogs0001.jpg
------- dogs0002.jpg
        ...
```

### Data exploration

Feel free to open and display some images.

### Model building

Build a Convolutional Neural Network.

### Model training

Since data is huge and might not fit in your computer memory, use keras `ImageDataGenerator` and `fit_generator` to train your model on the data.

### Performances estimation

Estimate the performances of your model with the accuracy metrics. Is it good enough? 

Then try to improve by all means you can leverage: architecture, hyperparameters, data augmentation.