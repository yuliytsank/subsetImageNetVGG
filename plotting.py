from matplotlib import pyplot, rc
from matplotlib.legend_handler import HandlerLine2D
import numpy
import argparse

parser = argparse.ArgumentParser(description='Plot results of saved learning curve data')
parser.add_argument('--saved-data-path', type=str, default='stats_dataAug_crop_Vgg16Bn_lr-0.01_m-0.9_drpOut-0.5.npy', metavar='N',
                    help='path of data file to use for plotting')

args = parser.parse_args()

stats = numpy.load(args.saved_data_path).item()
stats2 = numpy.load('stats_Vgg16Bn_lr-0.01_m-0.9_drpOut-0.5.npy').item()
stats3 = numpy.load('stats_Vgg16Bn_lr-0.01_m-0.9_drpOut-0.5.npy').item()

label1 = 'Yes Data Augment'
label2 = 'No Data Augment'
label3 = 'Momentum .9'
 
lwidth = 3
axis_fontsize = 40
title_fontsize = 60

pyplot.figure(1)

rc('xtick', labelsize=25) 
rc('ytick', labelsize=25) 

pyplot.subplot(121)
line1, = pyplot.plot(range(1,91), stats['perform']['train'], 'b--', 
                     linewidth=lwidth, label = 'Train Set -' +label1)
line2, = pyplot.plot(range(1,91), stats['perform']['test'], 'b', 
                     linewidth=lwidth, label = 'Test Set -' +label1)
line1, = pyplot.plot(range(1,91), stats2['perform']['train'], 'r--', 
                     linewidth=lwidth, label = 'Train Set -'+label2 )
line2, = pyplot.plot(range(1,91), stats2['perform']['test'], 'r', 
                     linewidth=lwidth, label = 'Test Set -'+label2)
#line1, = pyplot.plot(range(1,91), stats3['perform']['train'], 'g--', 
#                     linewidth=lwidth, label = 'Train Set -'+label3 )
#line2, = pyplot.plot(range(1,91), stats3['perform']['test'], 'g', 
#                     linewidth=lwidth, label = 'Test Set -'+label3)
pyplot.xlabel('Epoch', fontsize = axis_fontsize)
pyplot.ylabel('Performance', fontsize = axis_fontsize)
pyplot.title('Performance', fontsize = title_fontsize)


pyplot.subplot(122)

line1, = pyplot.plot(range(1,91), stats['losses']['train'], 'b--', 
                     linewidth=lwidth, label = 'Train Set -' +label1)
line2, = pyplot.plot(range(1,91), stats['losses']['test'], 'b', 
                     linewidth=lwidth, label = 'Test Set -' +label1)

line1, = pyplot.plot(range(1,91), stats2['losses']['train'], 'r--', 
                     linewidth=lwidth, label = 'Train Set -'+label2 )
line2, = pyplot.plot(range(1,91), stats2['losses']['test'], 'r', 
                     linewidth=lwidth, label = 'Test Set -'+label2)

#line1, = pyplot.plot(range(1,91), stats3['losses']['train'], 'g--', 
#                     linewidth=lwidth, label = 'Train Set -'+label3 )
#line2, = pyplot.plot(range(1,91), stats3['losses']['test'], 'g', 
#                     linewidth=lwidth, label = 'Test Set -'+label3)

pyplot.xlabel('Epoch', fontsize = axis_fontsize)
pyplot.ylabel('Loss', fontsize = axis_fontsize)
pyplot.title('Loss', fontsize = title_fontsize)

pyplot.legend(fontsize = 20, handler_map={line1: HandlerLine2D()}, 
                           bbox_to_anchor=(.65, 1), bbox_transform=pyplot.gcf().transFigure)
