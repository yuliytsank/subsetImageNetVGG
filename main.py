from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import random
import numpy as np
import pdb
import copy
import os
import csv
import time

#parts of code are based on an MNIST classification script at https://github.com/pytorch/examples/blob/master/mnist/main.py

# Training settings
parser = argparse.ArgumentParser(description='CNN exploration with subset of ImageNet, using VGG structure as a base')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 1000)')
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
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

save_name = 'stats_final_dataAug_crop_Vgg16Bn_lr-'+str(args.lr)+'_'+'m-'+str(args.momentum)+'_'+'drpOut-'+str(args.drop_out)
stats = {}
stats['losses'] = {}
stats['losses']['train'] = np.zeros([args.epochs])
stats['losses']['test'] = np.zeros([args.epochs])
stats['perform'] = {}
stats['perform']['train'] = np.zeros([args.epochs])
stats['perform']['test'] = np.zeros([args.epochs])
stats['time'] = np.zeros([args.epochs])


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
    
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

path_to_train_csv = os.path.join('.', 'imagenet56x56_release', 'train', 'train_labels.csv')
path_to_test_csv = os.path.join('.', 'imagenet56x56_release', 'test', 'test_sample_submission_kaggle.csv')
path_to_train_images = os.path.join('.', 'imagenet56x56_release', 'train', 'images')
path_to_test_images = os.path.join('.', 'imagenet56x56_release', 'test', 'images')

path_to_output = os.path.join('.', 'imagenet56x56_release', 'test', 'test_labels.csv')

fraction = .8
final_test = 1

test_ids = []
with open(path_to_test_csv, 'rb') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=',')
     next(spamreader)
     for row in spamreader:
         test_id = os.path.join(row[0])
         test_ids.append(test_id)
         
#test_ids = [row[:-5] for row in os.listdir(path_to_test_images)]

if not final_test:
    path_to_test_images = path_to_train_images
    path_to_test_csv = path_to_train_csv
    test_ids = [row[:-5] for row in os.listdir(path_to_test_images)]
    del test_ids[0:int(len(test_ids)*fraction)]
    test_shuffle = True
else:
#    path_to_test_csv = path_to_train_csv
    fraction = 1
    test_shuffle = False

class GetData(datasets.ImageFolder):
    from torchvision.datasets.folder import default_loader

    def __init__(self, path_to_csv, path_to_images, final_test = 0, train = True, train_fraction = .8, 
                 transform=None, target_transform=None,
                 loader=default_loader):

        classes = ['class_'+'0'*(3-len(str(cl))) +str(cl) for cl in range(0,100)]
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        subset_train_imgs, subset_test_imgs = self.get_dataset(path_to_csv, path_to_images, train_fraction, train)
        
        if train:
            imgs = subset_train_imgs
        else:
            imgs = subset_test_imgs

        
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
             spamreader = csv.reader(csvfile, delimiter=',')
             next(spamreader)
             for row in spamreader:
                 if (train != 1) & (final_test==1):
                     im_and_class = os.path.join(path_to_images,row[0]+'.JPEG'), 1
                 else:
                     im_and_class = os.path.join(path_to_images,row[0]+'.JPEG'), int(row[1])
                 train_imgs.append(im_and_class)

        if (train != 1) & (final_test==1):
            subset_train_imgs = []
            subset_test_imgs = train_imgs
        else:
            subset_train_imgs = train_imgs[0:int(len(train_imgs)*train_fraction)]
            del train_imgs[0:int(len(train_imgs)*train_fraction)]
            subset_test_imgs = train_imgs
        
        return subset_train_imgs, subset_test_imgs


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
train_loader = torch.utils.data.DataLoader(
    GetData(path_to_train_csv, path_to_train_images, final_test, train = 1, train_fraction = fraction,
                   transform=transforms.Compose([
#                           transforms.Scale(224),
                           transforms.RandomHorizontalFlip(),  
                           transforms.ToTensor(),
#                       transforms.Normalize((0.1307,), (0.3081,))
                        normalize
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    GetData(path_to_test_csv, path_to_test_images, final_test, train = 0, train_fraction = fraction,
                   transform=transforms.Compose([
#                       transforms.Scale(224),
                       transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
                        normalize
                   ])),
    batch_size=args.batch_size, shuffle=test_shuffle, **kwargs)

def addGaussNoise(train_loader, std_val):
    n_ims, h,w = train_loader.dataset.train_data.size()
    temp = train_loader.dataset.train_data.float()+torch.normal(std = torch.ones(n_ims,h,w)*std_val)
    temp[temp<0] = 0
    temp[temp>255] = 255    
    train_loader.dataset.train_data = temp.byte()
    return train_loader       
#    train_loader.dataset.train_data = torch.clamp(train_loader.dataset.train_data, max=255) # cmin
#    train_loader.dataset.train_data = torch.clamp(train_loader.dataset.train_data, min=0) # cmax

def randomize_labels(train_loader, proportion):
    s = train_loader.dataset.train_labels.size()
    num_labels = int(round(int(s[0])*proportion))
    temp = range(0,len(train_loader.dataset.train_labels))
    random.shuffle(temp)
    temp2 = temp[0:num_labels]
    random.shuffle(temp2)
    temp3 = copy.copy(temp)
    temp3[0:num_labels] = temp2
    train_loader.dataset.train_labels[torch.LongTensor(temp[0:num_labels])] = train_loader.dataset.train_labels[torch.LongTensor(temp3[0:num_labels])]
    return train_loader

import VGG_code    
model = VGG_code.vgg16_bn(num_classes = 100, dropout_p = args.drop_out)

criterion = nn.CrossEntropyLoss()
#model.features = torch.nn.DataParallel(model.features)
#model.cuda()
if args.cuda:
    model.cuda()
    model.features = torch.nn.DataParallel(model.features)
    criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
#        pdb.set_trace()
        output = model(data)
#        loss = F.nll_loss(output, target)
        loss = criterion(output, target)
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
#    np.save('train_losses_vgg16_bn', train_losses)
#preallocate space for confusion matrix

           
softmax = nn.Softmax()

top_row = test_loader.dataset.classes
top_row.insert(0,'id' )
save_epochs = [24,25,26,35,36,37,38]
def test(epoch):
    confusion_sums = np.zeros([100,100])
    class_trials = np.zeros([100,100])
    probs_list = np.empty((len(test_ids),100))
    scores_list = np.empty((len(test_ids),100))
    model.eval()
    test_loss = 0
    correct = 0
    test_batch = 0
    for data, target in test_loader:
        test_batch+=1
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        scores = output
        probs = softmax(output)
        probs_list[range(args.test_batch_size*(test_batch-1),args.test_batch_size*test_batch),:] = probs.data.cpu().numpy()
        scores_list[range(args.test_batch_size*(test_batch-1),args.test_batch_size*test_batch),:] = scores.data.cpu().numpy()
#        test_loss += F.nll_loss(output, target).data[0]
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        #update info for confusion matrix
        for target_class in range(0,100):
            class_trials[target_class,:] += sum(target.data==target_class)
            for output_class in range(0,100):
                examples = target.data[(target.data==target_class)&(pred==output_class)]
                confusion_sums[target_class, output_class]+= len(examples)
    
    if epoch in save_epochs:
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
    
    stats['perform']['test'][epoch-1] = correct_prop
    stats['losses']['test'][epoch-1] = test_loss
    stats['time'][epoch-1] = time.time()-start 
    np.save(save_name, stats) 
    
    if final_test:    
        if epoch in save_epochs:
            ids_and_probs = np.append(np.asarray(test_ids)[:,None], probs_list, 1)
            ids_and_scores = np.append(np.asarray(test_ids)[:,None], probs_list, 1)
            probs_output = np.append(np.asarray(top_row)[:,None].T, ids_and_probs, 0) 
            scores_output = np.append(np.asarray(top_row)[:,None].T, ids_and_scores, 0) 
            np.savetxt(path_to_output[:-4]+str(epoch)+path_to_output[-4:], probs_output, fmt = '%s', delimiter = ',')
            np.savetxt(path_to_output[:-4]+'scores'+str(epoch)+path_to_output[-4:], scores_output, fmt = '%s', delimiter = ',')
    
    return correct_prop


#    train_loader = randomize_labels(train_loader_static, rand_mat[0,ind])
#train_loader = addGaussNoise(train_loader_static, noise_mat[0,ind])
start = time.time()
for epoch in range(1, args.epochs + 1):
#    pdb.set_trace()
    train_loader = torch.utils.data.DataLoader(
    GetData(path_to_train_csv, path_to_train_images, final_test, train = 1, train_fraction = fraction,
                   transform=transforms.Compose([
#                           transforms.Scale(224),
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomCrop(56, padding=7),                               
                           transforms.ToTensor(),
#                           transforms.Lambda(lambda x: x +torch.normal(std = torch.ones(56,56,3)*.2)),
#                       transforms.Normalize((0.1307,), (0.3081,))
                        normalize
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
    
    train(epoch)
    test(epoch)
   
#    pdb.set_trace()

#correct_prop = test(epoch)
#    rand_mat[1,ind] = correct_prop
#noise_mat[1,ind] = correct_prop
#confusion_matrix = confusion_sums/class_trials
#np.save('noise_mat', noise_mat)
#np.save('rand_mat', rand_mat)
