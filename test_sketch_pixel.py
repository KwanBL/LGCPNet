from __future__ import print_function
import argparse
import numpy as np
import torch.utils.data
from torch.autograd import Variable
from datasets import PartDataset
from MyDataSet import LabelDataset
from model.SketchNet3 import SketchSeg
import datetime


cate = 'Table'
root = './label/'
batch = 1
numpoints = 2500
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '/home/bailiang/codes/guanbiliang/3d_seg_Pistol/output/model/'+'best'''+cate+'model.pth',  help='model path')
parser.add_argument('--idx', type=int, default = 0,   help='model index')

classnumber = 3

opt = parser.parse_args()
print (opt)
#classes = ['tube','base','shade']
starttime = datetime.datetime.now()
d = LabelDataset(root = 'test3000',npoints = numpoints, class_choice = cate, train = False)
dataloader = torch.utils.data.DataLoader(d, batch_size=batch,
                                          shuffle=False, num_workers=4)
classifier = SketchSeg(k = classnumber)
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()
classifier.cuda()
pixel = 0.0
print("the number of images:%d\n" %(len(d)))

for i, data in enumerate(dataloader, 0):
    points, target = data
    points, target = Variable(points).float(), Variable(target).long()
    point = points.transpose(2, 1)
    points, target = point.cuda(), target.cuda()
    pred = classifier(points)
    #print(pred.size(), target.size())

    pred = pred.view(-1, classnumber)
    target = target.view(-1, 1)[:, 0] - 1
    #print(pred.size(), target.size())

    pred_choice = pred.data.max(1)[1]
    btiou = pred_choice.size()[0]
    bta = points.size()[0]
    count = pred_choice.eq(target.data).cpu().sum().float()
    pixel_accuracy = count / (numpoints*batch)
    #print("Pixel accuracy is:%f" %(pixel_accuracy))
    pixel += pixel_accuracy
    pred_choice = pred_choice.cpu().int().numpy().tolist()
    labels = np.asarray(pred_choice)
    np.savetxt(root+cate+str(i) + '.txt', labels, fmt="%i")

pixel /= (i+1)
print("pixel is:%f" %(pixel))


