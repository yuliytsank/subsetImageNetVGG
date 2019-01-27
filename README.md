# subsetImageNetVGG
This is an exploration of CNNs on the toy problem of classifying a small subset of the ImageNet data set down-sampled to a size of  56x56. 

## Classification of A Small Image Set With a VGG-Based Neural Network

#### Dataset Description Link: 

- download and unzip directory containing training and testing images from this link and place into subsetImageNetVGG directory: https://drive.google.com/open?id=1SJTzI2YIXpOEgOg4WTgxL2DmUrhd_Zwk 
- imagenet56x56_release directory containes a "test" and "train" directory. The test directory was originally used for a kaggle submission as part of coursework, so all training and testing is only done with the "train" directory. The "train" directory contains an "images" directory with 50,000 images from 100 classes randomly sampled from ImageNet and resized to 56x56. The "train" directory also contains a "train_labels.csv" file with labels for all training images. The code splits the training images into a training and testing set. 

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

###### Individual Class Summary Statistics:
Below are plots of confusion matrices for the target class vs. the output class at different epochs. Next to the plots are summary statistics of the performance of individual classes (the number of classes that have performance above 50%, performance below 20% and the standard deviation of performance across classes. The “#Classes Lower Perf” metric refers to the number of classes that show lower performance relative to an earlier epoch that is listed. For example, for epoch 5, that metric will show the number of classes at epoch 5 that have lower performance than at epoch 1. For epoch 20 it will be the number of classes at epoch 20 that have lower performance than at epoch 5. 
When looking at the summary statistics, it seems that when loss is increasing again (at epoch 45), the statistics for both the lowest performing classes and highest performing classes keep improving, however there is an increase in the number of classes that show lower performance than at a previous epoch. This discrepancy in performance between high-performing and low-performing classes does not seem to be large (the standard deviation of performance across classes stays roughly the same), so it may be beneficial to keep training even after the loss starts rising again.  

Epoch 1 (high loss): Above  0.5: 6 | Below  0.2: 78 | Std: 0.18:  
<img src="/Results Plots/conf_mat1.png" height="70%" width="70%">

Epoch 5 (decreasing loss): Above  0.5: 16| Below  0.2: 46 | Num Classes Lower Perf: 18 | Std: 0.19
<img src="/Results Plots/conf_mat5.png" height="70%" width="70%">

Epoch 20 (minimum loss): Above  0.5: 25 | Below  0.2: 23 | Num Classes Lower Perf: 12 | Std: 0.199  
<img src="/Results Plots/conf_mat20.png" height="70%" width="70%">

Epoch 45 (increasing loss): Above  0.5: 28 | Below  0.2: 8 | Num Classes Lower Perf: 36 | Std: 0.18   
<img src="/Results Plots/conf_mat45.png" height="70%" width="70%">


#### Network Depth:
Here, the effects on performance of VGG-based networks of different depths are tested. The only difference between these networks is the number of convolutional layers that are used. It seems that having a deeper network does improve performance but there is diminishing return, where the performance difference between a 19 and 11 layer network is smaller than an 11 and 8 layer network.

<img src="/Results Plots/Num Layers.png">

#### Dropout Effects:
Here, 3 different dropout values are tested to see the effects on performance. A small dropout value of .25 shows slightly lower performance than a medium dropout value of .5 because there is likely more overfitting with a smaller dropout value. A very large dropout value of .9 results in a large drop in performance because there is not enough complexity left in the network. 

<img src="/Results Plots/Dropout_25_included.png">

#### Batch Normalization:
Using batch normalization at every layer seems to provide a large boost in performance. Here performance is compared in a VGG-16 network with the same hyperparameters with and without batch normalization. 

<img src="/Results Plots/batch_norm.png">

#### Learning Rate:
The effects of three different learning rates are tested, with an order of magnitude difference between them (.001, .01, and .1). The learning rate in the middle shows the highest performance. A learning rate that is too small, likely does contribute enough ‘energy’ to the system in order to escape a local minimum. 

<img src="/Results Plots/LR.png">

#### Momentum:
Three different momentum values are tested. A high momentum value of .9 results in better performance than lower values.

<img src="/Results Plots/Momentum.png">

#### Data Augmentation:
Augmenting the 40,000 training images (the rest of the 10,000 were used for testing) by adding a random horizontal flip as well as random cropping seems to improve performance. The augmentation is done by processing the training data slightly differently at every epoch, which makes the network more robust and helps prevent overfitting. The cropping is done by first zero-padding the 56x56 images to create 70x70 images and then randomly cropping back down to 56x56 images. Since the augmentation is done across epochs, it takes longer for the network to converge. 

<img src="/Results Plots/DataAugmentation.png">
