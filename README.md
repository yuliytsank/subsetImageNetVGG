# subsetImageNetVGG
This is an exploration of CNNs on the toy problem of classifying a small subset of the ImageNet data set down-sampled to a size of  56x56. 

## Classification of A Small Image Set With a VGG-Based Neural Network

#### Dataset Description Link: 

- download and unzip directory containing training and testing images from this link and place into subsetImageNetVGG directory: https://drive.google.com/open?id=1SJTzI2YIXpOEgOg4WTgxL2DmUrhd_Zwk 
- imagenet56x56_release directory containes a "test" and "train" directory. The test directory was originally used for a kaggle submission as part of coursework, so all training and testing is only done with the "train" directory. The "train" directory contains an "images" directory with 50,000 images from 1000 classes randomly sampled from ImageNet and resized to 56x56. The "train" directory also contains a "train_labels.csv" file with labels for all training images. The code splits the training images into a training and testing set. 

#### Overview of what main.py does: 

- Splits training data into “Training” and “Validation” set  
- Uses Pytorch and loads data using the torchvision DataLoader class  
- Network used is based on existing VGG-16 architecture  
- Uses Cross-Entropy Loss and SGD during training  
- Able to retrain full model with different hyperparameters (dropout, learning rate, momentum) and network depths as well as data augmentation to see effects on performance  
- Tracks performance and loss for each epoch across 90 epochs (run on a GPU)

#### Final Architecture and Settings set as default:
- VGG-16 with first 2 pooling layers taken out
- Learning rate: .01
- Momentum: .9
- Dropout: .5
- Batch Normalization applied at every layer
- Data augmentation with random horizontal flip and padding to 70x70px image, then random cropping back to 56x56 image

### Testing different network architectures by making modifications to VGG

#### Max Pooling Choice:
Here the effects on performance are tested when taking out the first two pooling layers vs. taking out the last two pooling layers from the VGG-16 architecture. Two pooling layers need to be taken out in order to adjust the accepted input to the network from 224x224 to a 56x56 size. There is a clear advantage of using a network with the first two pooling layers removed instead of the last two layers. 

![alt text](</Results Plots/Pool_layers_choice.png?raw=true> "Choice of pooling layers reduction")

#### Analysis of Overfitting:
In the previous plot, there seems to clear overfitting because of a large gap in performance between the training and testing sets  as well as a rising loss after reaching a minimum around epoch 15. Since cross-entropy loss was used, the probability scores across the labels for each test sample were extracted, then ordered them from highest to lowest, and then averaged across the test samples to get a single distribution. This was then done for different epochs (plots from epoch 1,5,20, and 45 are shown below. The trend is that as training progresses, the probability mass becomes more and more concentrated at the higher values. There seems to be a tradeoff between loss that comes from a correct label prediction vs. an incorrect one. Early on, there is high loss for correct predictions. As the loss for correct predictions gets lower, the loss for incorrect predictions grows. Confidence in the correct label is expressed as a high value for the highest probability score. If the model is not confident enough, a correct answer will still result in a high loss. If the model is too confident (too much emphasis put on certain features due to overtraining), a mistake will result in a higher loss. 

Epoch 1 (high loss):  
<img src="/Results Plots/probs_ordered1.png" height="70%" width="70%">

Epoch 5 (decreasing loss):  
<img src="/Results Plots/probs_ordered5.png" height="70%" width="70%">

Epoch 20 (minimum loss):  
<img src="/Results Plots/probs_ordered20.png" height="70%" width="70%">

Epoch 45 (increasing loss):  
<img src="/Results Plots/probs_ordered45.png" height="70%" width="70%">

