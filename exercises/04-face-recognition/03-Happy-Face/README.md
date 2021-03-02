# 07-05

## Challenge 03 - Happy Face

![](https://images.unsplash.com/photo-1472162072942-cd5147eb3902?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1050&q=80)

Photo by [Ben White](https://unsplash.com/photos/4K2lIP0zc_k)

## Guidelines

In this challenge, you will predict whether a person is smiling or not on an picture. The dataset is in `dataset.pkl`, and can be loaded using pickle:
```Python
import pickle
X, y = pickle.load(open("dataset.pkl", "rb"))
```

It contains images of people smiling `y=1` or not `y=0`.

You will create a classifier based on those features and labels using three methods:
- A PCA that retains 99% of the information followed by a random forest
- A CNN
- A facial landmark extraction followed by a random forest

For each case, you may have to handle the data a bit differently. In each case, compute the accuracy and compare the final results: what method is the most accurate? what method gives the best results for a limited development time?