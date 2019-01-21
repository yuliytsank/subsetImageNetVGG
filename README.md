# subsetImageNetVGG
This is an exploration of CNNs on the toy problem of classifying a small subset of the ImageNet data set down-sampled to a size of  56x56. 

## Classification of A Small Image Set With a VGG-Based Neural Network

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
Here the effects on performance are tested when taking out the first two pooling layers vs. taking out the last two pooling layers from the VGG-16 architecture. Two pooling layers need to be taken out in order to adjust the accepted input to the network from 224x224 to a 56x56 size. There is a clear advantage of using a network with the first two pooling layers removed. 

![alt text](/Results Plots/Pool_layers_choice.png "Choice of pooling layers reduction")

