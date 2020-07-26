from __future__ import print_function
import argparse
import os
import random
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from datasets import PartDataset
from model.SketchNet3 import SketchSeg
import torch.nn.functional as F




parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=12, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')

numpoints = 2500
opt = parser.parse_args()
print (opt)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cate = 'cosegVasesLarge'
num_classes = 4


train_dataset = PartDataset(root = 'PSB_COSEG_test2500', npoints = numpoints, class_choice = cate)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

test_dataset = PartDataset(root = 'PSB_COSEG_test2500', npoints = numpoints, class_choice = cate, train = False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                          shuffle=False, num_workers=int(opt.workers))


print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = SketchSeg(k = num_classes)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
classifier.cuda()

num_batch = len(train_dataset)/opt.batchSize
best_test_acc = 0

for epoch in range(opt.nepoch):
    #scheduler.step()
    count = 0.0
    pixel = 0.0
    train_acc = 0.0
    classifier.train()
    for i, data in enumerate(train_loader, 0):
        points, target = data
        points, target = Variable(points).float(), Variable(target).long()
        point = points.transpose(2,1)
        points, target = point.cuda(), target.cuda()
        optimizer.zero_grad()
        pred = classifier(points)
        pred = pred.view(-1, num_classes)
        target = target.view(-1,1)[:,0] - 1
        #print(pred.size(), target.size())
        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        btiou = pred_choice.size()[0]
        bta = points.size()[0]
        correct = pred_choice.eq(target.data).cpu().sum().float()
        train_accuracy = correct / (float)(bta * numpoints)
        print("Train_accuracy is:%f" % (train_accuracy))
        train_acc += train_accuracy
    print('[%d: %d/%d] train loss: %f accuracy: %f ' %
          (epoch, i, num_batch, loss.data, train_acc / (i + 1)))

    classifier.eval()
    for i, data in enumerate(test_loader, 0):
        points, target = data
        points, target = Variable(points).float(), Variable(target).long()
        point = points.transpose(2, 1)
        points, target = point.cuda(), target.cuda()
        pred = classifier(points)
        #print(pred.size(), target.size())
        pred = pred.view(-1, num_classes)
        target = target.view(-1, 1)[:, 0] - 1
        #print(pred.size(), target.size())
        pred_choice = pred.data.max(1)[1]
        btiou = pred_choice.size()[0]
        bta = points.size()[0]
        count = pred_choice.eq(target.data).cpu().sum().float()
        pixel_accuracy = count / (numpoints * bta)
        #print("Pixel accuracy is:%f" % (pixel_accuracy))
        pixel += pixel_accuracy

    test_acc = pixel / (i + 1)
    print("test_acc is:%f" % (test_acc))

    if test_acc >= best_test_acc:
        best_test_acc = test_acc
        torch.save(classifier.state_dict(), './output/model/'+'best'+cate+'model.pth')
        print("Update!")
    print("best_test_acc is:%f" % (best_test_acc))

print("best_test_acc is:%f" % (best_test_acc))

