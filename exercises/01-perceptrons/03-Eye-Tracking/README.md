# 07-02

## Challenge 02 - Eye Tracking

![](https://images.unsplash.com/photo-1531704118376-ec637130bfa0?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1050&q=80)

Photo by [Eduardo Mallmann](https://unsplash.com/photos/3LPGWASiSbM)

## Guidelines

### Introduction

Today will be a mini project based on Eye Tracking. A bit of context first. You have two folders:
* `data/open`
* `data/close`

In those two folders, are present images of eyes, either opened or closed. You will have two main tasks:
* Classification: make a classifier that predicts if the eye is opened or closed
* Regression: make a regression that predicts the center of the pupil

Before doing so, you have to handle properly the images.

### Data Exploration
First, look at the data, open an image, display it, convert it to an array.
Feel free to use keras methods `load_img` and `img_to_array`.

### Classification

Your first task will be to make a classifier that detects either an eye is closed or not based on a picture.

### Regression

Your second task will be to predict the center of an eye using regression, using the target values into the file `data/open/dataPupilCenter.csv`

The main idea is not really to find the perfect center, but to allow then to have a region of interest around the eye for further processing, like in the following image:
![](../../00-Lectures/images/eye_box.png)
