from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import random
import numpy as np
import copy
import os
import csv
import time
import custom_VGG 

#skeleton of code (setup of basic torch settings) is based on an MNIST classification script at https://github.com/pytorch/examples/blob/master/mnist/main.py

# Training settings
final_test = 0# this is left out of the optinal arguments because it was only used in a specific case for a kaggle submission
parser = argparse.ArgumentParser(description='CNN exploration with subset of ImageNet, using VGG structure as a base')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default = 90, metavar='N',
                    help='number of epochs to train (default: 90)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--drop-out', type=float, default=0.5, metavar='DO',
                    help='dropout proportion during training rate (default: 0.5)')
parser.add_argument('--train-prop', type=float, default=0.8, metavar='DO',
                    help='proportion of labels set to use for training (default: 0.8)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
fraction = args.train_prop

''' Set options for model type, optimizer type, and loss function to use'''
model = custom_VGG.vgg16_bn(num_classes = 100, dropout_p = args.drop_out)
criterion = nn.CrossEntropyLoss() #set loss function to use 
if args.cuda: #move model to GPU if using one
    model.cuda()
    model.features = torch.nn.DataParallel(model.features)
    criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum) #set optimizer to use
softmax = nn.Softmax() #instantiate softmax function to extract probability scores later

#Preaallocate dictionary to record train and test set losses and performance
save_name = 'stats_final_dataAug_crop_Vgg16Bn_lr-'+str(args.lr)+'_'+'m-'+str(args.momentum)+'_'+'drpOut-'+str(args.drop_out)
stats = {}
stats['losses'] = {}
stats['losses']['train'] = np.zeros([args.epochs])
stats['losses']['test'] = np.zeros([args.epochs])
stats['perform'] = {}
stats['perform']['train'] = np.zeros([args.epochs])
stats['perform']['test'] = np.zeros([args.epochs])
stats['time'] = np.zeros([args.epochs])

#Set paths for training data, testing data, and labels , as well as output csv for kaggle submission  
path_to_train_csv = os.path.join('.', 'imagenet56x56_release', 'train', 'train_labels.csv')
path_to_test_csv = os.path.join('.', 'imagenet56x56_release', 'test', 'test_sample_submission_kaggle.csv')
path_to_train_images = os.path.join('.', 'imagenet56x56_release', 'train', 'images')
path_to_test_images = os.path.join('.', 'imagenet56x56_release', 'test', 'images')
path_to_output = os.path.join('.', 'imagenet56x56_release', 'test', 'test_labels.csv')

''' Custom dataloader class to read image path, label pairs from a csv file and load them '''
class GetData(datasets.ImageFolder):
    from torchvision.datasets.folder import default_loader

    def __init__(self, path_to_csv, path_to_images, final_test = 0, train = True, train_fraction = .8, 
                 transform=None, target_transform=None,
                 loader=default_loader):

        classes = ['class_'+'0'*(3-len(str(cl))) +str(cl) for cl in range(0,100)] #create class label names 
        class_to_idx = {classes[i]: i for i in range(len(classes))}               #create class label indices  
        subset_train_imgs, subset_test_imgs = self.get_dataset(path_to_csv, path_to_images, train_fraction, train) #get subset of training and testing images from large dataset
        
        if train: 
            imgs = subset_train_imgs #if loading in training set
        else: 
            imgs = subset_test_imgs #if loading in testing set

        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        

    def get_dataset(self,path_to_csv, path_to_images, train_fraction, train):
        import csv
        train_imgs = []
        with open(path_to_csv, 'rb') as csvfile:
             csvreader = csv.reader(csvfile, delimiter=',')
             next(csvreader)
             for row in csvreader: #loop through image paths and labels
                 if (train != 1) & (final_test==1):
                     im_and_class = os.path.join(path_to_images,row[0]+'.JPEG'), 1 #if ifinal kaggle submission, then extract only image names
                 else:
                     im_and_class = os.path.join(path_to_images,row[0]+'.JPEG'), int(row[1]) #if not final kaggle submission, extract image names and labels
                 train_imgs.append(im_and_class)

        if (train != 1) & (final_test==1):
            subset_train_imgs = []
            subset_test_imgs = train_imgs #if final kaggle submission, indicate only a testing set without labels
        else:
            subset_train_imgs = train_imgs[0:int(len(train_imgs)*train_fraction)] # if not final kaggle submission, indicate a training set based on chosen proportion of images used for testing
            del train_imgs[0:int(len(train_imgs)*train_fraction)] # delete part of train images used for training, in order to only leave a set of test images
            subset_test_imgs = train_imgs #assign test images left from training set
        
        return subset_train_imgs, subset_test_imgs

''' Unused functions that may be useful for a different custom dataloader######'''
def addGaussNoise(train_loader, std_val):
    n_ims, h,w = train_loader.dataset.train_data.size() #get dimentions of data set
    temp = train_loader.dataset.train_data.float()+torch.normal(std = torch.ones(n_ims,h,w)*std_val) # add gaussian noise to intermediate variable 
    temp[temp<0] = 0 #make sure values stay in 0-255 range
    temp[temp>255] = 255    
    train_loader.dataset.train_data = temp.byte() #assign intermediate variable with noise back to  dataset
    return train_loader #output data set back after adding noise to it 

def randomize_labels(train_loader, proportion):
    s = train_loader.dataset.train_labels.size() #find size of dataset
    num_labels = int(round(int(s[0])*proportion)) #find number of samples in dataset used for training/testing based on specified proportion
    temp = range(0,len(train_loader.dataset.train_labels)) #create list of indices matching number of samples
    random.shuffle(temp) #randomize list of indices
    temp2 = temp[0:num_labels] #extract a subset of randomized labels based on proportion 
    random.shuffle(temp2)
    temp3 = copy.copy(temp)
    temp3[0:num_labels] = temp2
    train_loader.dataset.train_labels[torch.LongTensor(temp[0:num_labels])] = train_loader.dataset.train_labels[torch.LongTensor(temp3[0:num_labels])]
    return train_loader

  
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad() #reset gradients for optimizer
        output = model(data) #get output labels
        loss = criterion(output, target) #get loss based on output and target labels
        train_loss += loss.data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    train_loss /=len(train_loader)  
    correct_prop = np.float(correct) / np.float(len(train_loader.dataset))
    stats['perform']['train'][epoch-1] = correct_prop
    stats['losses']['train'][epoch-1] = train_loss
           
def test(epoch):
    #Preallocate space for various outputs 
    confusion_sums = np.zeros([100,100]) #confusion matrix
    class_trials = np.zeros([100,100]) #number of trials for each entry in confusion matrix
    probs_list = np.empty((len(test_ids),100)) #list of probabilities after output from softmax 
    model.eval() #turn model evaluation mode on for testing so that parameters remain constant
    test_loss = 0
    correct = 0
    test_batch = 0
    for data, target in test_loader: #loop through batches in test data set 
        test_batch+=1
        if args.cuda: #move data to cuda if using a GPU
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data) #get output predictions for current batch
        probs = softmax(output)
        probs_list[range(args.test_batch_size*(test_batch-1),args.test_batch_size*test_batch),:] = probs.data.cpu().numpy()
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        #update info for confusion matrix
        for target_class in range(0,100): # loop through target labels 
            class_trials[target_class,:] += sum(target.data==target_class)
            for output_class in range(0,100): #loop through output labels
                examples = target.data[(target.data==target_class)&(pred==output_class)]    #find number of matches between target label and output label
                confusion_sums[target_class, output_class]+= len(examples)                  #fill in confusion matrix for current element of target and output label
 

    if epoch in save_epochs: #save confusion matrices and output of softmax probability scroes for further analysis
        confusion_mat = {}
        confusion_mat['confusion_sums'] = confusion_sums
        confusion_mat['class_trials'] = class_trials            
        np.save('confusion_mat_'+str(epoch), confusion_mat)
        np.save('probs_list'+str(epoch), probs_list)               
            
    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    correct_prop = np.float(correct) / np.float(len(test_loader.dataset))
    
    #save loss and performance across epochs
    stats['perform']['test'][epoch-1] = correct_prop
    stats['losses']['test'][epoch-1] = test_loss
    stats['time'][epoch-1] = time.time()-start 
    np.save(save_name, stats) 
    
    if final_test: #If this is for a final kaggle submission, prepare an output csv file with assigned labels to test image names   
        top_row = test_loader.dataset.classes
        top_row.insert(0,'id' )   
        if epoch in save_epochs:
            ids_and_probs = np.append(np.asarray(test_ids)[:,None], probs_list, 1)
            ids_and_scores = np.append(np.asarray(test_ids)[:,None], probs_list, 1)
            probs_output = np.append(np.asarray(top_row)[:,None].T, ids_and_probs, 0) 
            scores_output = np.append(np.asarray(top_row)[:,None].T, ids_and_scores, 0) 
            np.savetxt(path_to_output[:-4]+str(epoch)+path_to_output[-4:], probs_output, fmt = '%s', delimiter = ',')
            np.savetxt(path_to_output[:-4]+'scores'+str(epoch)+path_to_output[-4:], scores_output, fmt = '%s', delimiter = ',')
    
    return correct_prop


''' Load testing and training sets and run epochs##########################'''
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}#set options for memory and multithreading when loading data
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],    #set parameters for normalizing mean and std of each color channel in images
                                     std=[0.229, 0.224, 0.225])

##########################Extract image paths files for training or testing ###
test_ids = []
if not final_test: #if not final kaggle submission
    path_to_test_images = path_to_train_images #test images are a subset of train images 
    path_to_test_csv = path_to_train_csv # the train csv file contains labels for a split training and testing set
    test_ids = [row[:-5] for row in os.listdir(path_to_test_images)]
    del test_ids[0:int(len(test_ids)*fraction)]
    test_shuffle = True
else:
    with open(path_to_test_csv, 'rb') as csvfile: #open csv file with image paths 
     csvreader = csv.reader(csvfile, delimiter=',')
     next(csvreader)
     for row in csvreader:
         test_id = os.path.join(row[0])
         test_ids.append(test_id)
    fraction = 1
    test_shuffle = False # do not shuffle image paths in order to submit ordered labels to kaggle
###############################################################################

test_loader = torch.utils.data.DataLoader(                      #load in test image set
    GetData(path_to_test_csv, path_to_test_images, final_test, train = 0, train_fraction = fraction,
                       transform=transforms.Compose([
                       transforms.ToTensor(),
                       normalize
                   ])),
    batch_size=args.batch_size, shuffle=test_shuffle, **kwargs)

start = time.time()
for epoch in range(1, args.epochs + 1): #loop through epochs and randomly transform data on each one
    train_loader = torch.utils.data.DataLoader(                                                         #need to reload data every epoch because of different random transforms
    GetData(path_to_train_csv, path_to_train_images, final_test, train = 1, train_fraction = fraction,  #use custom dataloader
                   transform=transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomCrop(56, padding=7),    #add 7px of padding to 56x56 image on all sides and then randomly crop back to 56x56                         
                           transforms.ToTensor(),
                        normalize
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
    
    save_epochs = [24,25,26,35,36,37,38] #use these epoch number to save learning curve and perormance data for early stopping to minimize testing error
    train(epoch)
    test(epoch)
