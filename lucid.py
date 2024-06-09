from __future__ import print_function, division 
import os
import sys
from xml.etree.ElementPath import prepare_self
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
import pickle
import pandas as pd
from PIL import Image
import argparse
from apex import amp
from sklearn.metrics import roc_auc_score
from models.modeling_lucid import LUCID, CONFIGS, LUCID_drophead,LUCID_fixed
from tqdm import tqdm,trange
import argparse
import warnings
from torchsummary import summary
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
if not sys.warnoptions:
    warnings.simplefilter("ignore")

tk_lim = 40

disease_list = ['1', '2', '3', '4']

def load_weights(model, weight_path):
    pretrained_weights = torch.load(weight_path, map_location=torch.device('cpu'))
    model_weights = model.state_dict()

    load_weights = {k: v for k, v in list(pretrained_weights.items())[:-1] if k in model_weights}

    model_weights.update(load_weights)
    model.load_state_dict(model_weights)
    print("Loading LUCID...")
    return model

def computeAUROC (dataGT, dataPRED, classCount=4):
    outAUROC = []
        
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()

    for i in range(classCount):
        outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            
    return outAUROC

class Data(Dataset):
    def __init__(self, set_type, img_dir, transform=None, target_transform=None):
        dict_path = set_type+'.pkl'
        # dict_path = "subset.pkl"
        f = open(dict_path, 'rb') 
        self.mm_data = pickle.load(f)
        f.close()
        self.idx_list = list(self.mm_data.keys())
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        k = self.idx_list[idx]
        img_path = os.path.join(self.img_dir, "0"*(10-len(k))+k) + '.jpg'

        img = Image.open(img_path).convert('RGB')

        label = np.array(self.mm_data[k]['label_EGFR']).astype('float32')
        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)

        # if list(label)==[0,0,0,0] or list(label)==[1,0,0,0] or list(label)==[0,1,0,0]:
        #     label=np.array([1,0])
        # else:
        #     label=np.array([0,1])

        cc = torch.tensor(self.mm_data[k]['pdesc'],requires_grad=False).float()
        demo = torch.from_numpy(np.array(self.mm_data[k]['bics'])).float()
        lab = torch.from_numpy(self.mm_data[k]['bts']).float()
        return img, label, cc, demo, lab

def test(args):
    torch.manual_seed(0)
    num_classes = args.CLS
    config = CONFIGS["LUCID"]
    lucid = LUCID(config, 224, zero_head=True, num_classes=2,use_focal_loss=True).cuda()
    # model = LUCID_drophead(config, 224, zero_head=True, num_classes=num_classes,fixed_weight=True)
    # lucid = load_weights(model, 'model.pth')
    # lucid = LUCID_fixed(config,lucid,num_classes=2,use_focal_loss=True).cuda()
    img_dir = args.DATA_DIR

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
    }

    all_data = Data(args.SET_TYPE, img_dir, transform=data_transforms['test'])

    train_size = int(0.7 * all_data.__len__())
    dev_size=int(0.2* all_data.__len__())
    test_size = all_data.__len__() - train_size-dev_size
    train_dataset, dev_dataset,test_dataset = torch.utils.data.random_split(all_data, [train_size, dev_size,test_size])

    trainloader = DataLoader(train_dataset, batch_size=args.BSZ, shuffle=True, num_workers=16, pin_memory=True)
    devloader = DataLoader(dev_dataset, batch_size=args.BSZ, shuffle=True, num_workers=16, pin_memory=True)
    testloader = DataLoader(test_dataset, batch_size=args.BSZ, shuffle=False, num_workers=16, pin_memory=True)

    optimizer_lucid = torch.optim.AdamW(lucid.parameters(), lr=3e-4, weight_decay=0.01)
    # lucid, optimizer_lucid = amp.initialize(lucid.cuda(), optimizer_lucid, opt_level="O1")

    lucid = torch.nn.DataParallel(lucid)

    # ----- Train ------
    best_model=[]
    best_f1=0

    for epoch in trange(50):
        lucid.train()
        pred_list=[]
        label_list=[]
        total_loss=0
        outGT = torch.FloatTensor().cuda(non_blocking=True)
        outPRED = torch.FloatTensor().cuda(non_blocking=True)
        for data in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            imgs, labels, cc, demo, lab = data
            if args.USE_TEXT:
                cc = cc.view(-1, tk_lim, cc.shape[3]).cuda(non_blocking=True).float()
                demo = demo.view(-1, 1, demo.shape[1]).cuda(non_blocking=True).float()
                lab = lab.view(-1, lab.shape[1], 1).cuda(non_blocking=True).float()
                sex = demo[:, :, 1].view(-1, 1, 1).cuda(non_blocking=True).float()
                age = demo[:, :, 0].view(-1, 1, 1).cuda(non_blocking=True).float()
            else:
                cc=None
                lab=None
                sex=None
                age=None
            imgs = imgs.cuda(non_blocking=True)
            print(imgs.shape)
            labels = labels.cuda(non_blocking=True)
            # loss= lucid(imgs, cc, lab, sex, age,labels)
            loss,preds = lucid(imgs, cc, lab, sex, age,labels)
            loss=loss.sum()
            loss.backward()
            optimizer_lucid.step()
            optimizer_lucid.zero_grad()
            total_loss+=loss
            probs = torch.sigmoid(preds)
            outGT = torch.cat((outGT, labels), 0)
            outPRED = torch.cat((outPRED, probs.data), 0)

            pred_list.extend(list(torch.argmax(preds,1).cpu()))
            label_list.extend(list(torch.argmax(labels,1).cpu()))

        aurocIndividual = computeAUROC(outGT, outPRED, classCount=num_classes)
        aurocMean = np.array(aurocIndividual).mean()
        
        print('mean AUROC:' + str(aurocMean))
         
        for i in range (0, len(aurocIndividual)):
            print(disease_list[i] + ': '+str(aurocIndividual[i]))
        
        print(total_loss)
        print(preds[:5])
        print(labels[:5])
        print(confusion_matrix(pred_list,label_list))
        print(epoch,accuracy_score(pred_list,label_list))
        print(epoch,f1_score(pred_list,label_list,average="macro"))
    
        #----- Dev ------

        lucid.eval()
        with torch.no_grad():
            pred_list=[]
            label_list=[]
            outGT = torch.FloatTensor().cuda(non_blocking=True)
            outPRED = torch.FloatTensor().cuda(non_blocking=True)
            for data in devloader:
                # get the inputs; data is a list of [inputs, labels]
                imgs, labels, cc, demo, lab = data
                if args.USE_TEXT:
                    cc = cc.view(-1, tk_lim, cc.shape[3]).cuda(non_blocking=True).float()
                    demo = demo.view(-1, 1, demo.shape[1]).cuda(non_blocking=True).float()
                    lab = lab.view(-1, lab.shape[1], 1).cuda(non_blocking=True).float()
                    sex = demo[:, :, 1].view(-1, 1, 1).cuda(non_blocking=True).float()
                    age = demo[:, :, 0].view(-1, 1, 1).cuda(non_blocking=True).float()
                else:
                    cc=None
                    lab=None
                    sex=None
                    age=None
                imgs = imgs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                loss,preds = lucid(imgs, cc, lab, sex, age,labels)

                probs = torch.sigmoid(preds)
                outGT = torch.cat((outGT, labels), 0)
                outPRED = torch.cat((outPRED, probs.data), 0)

                pred_list.extend(list(torch.argmax(preds,1).cpu()))
                label_list.extend(list(torch.argmax(labels,1).cpu()))

            aurocIndividual = computeAUROC(outGT, outPRED, classCount=num_classes)
            aurocMean = np.array(aurocIndividual).mean()
            
            print('mean AUROC:' + str(aurocMean))
            
            for i in range (0, len(aurocIndividual)):
                print(disease_list[i] + ': '+str(aurocIndividual[i]))

            print(preds[:5])
            print(labels[:5])
            print(confusion_matrix(pred_list,label_list))
            print(epoch,accuracy_score(pred_list,label_list))
            print(epoch,f1_score(pred_list,label_list,average="macro"))
            if f1_score(pred_list,label_list,average="macro")>best_f1:
                best_f1=f1_score(pred_list,label_list,average="macro")
                print("Best Model!")
                best_model=lucid.state_dict()

    lucid = LUCID(config, 224, zero_head=True, num_classes=2,use_focal_loss=True).cuda()
    lucid = torch.nn.DataParallel(lucid)
    lucid.load_state_dict(best_model)
    #----- Test ------
    print('--------Start testing-------')
    lucid.eval()
    with torch.no_grad():
        pred_list=[]
        label_list=[]
        outGT = torch.FloatTensor().cuda(non_blocking=True)
        outPRED = torch.FloatTensor().cuda(non_blocking=True)
        for data in testloader:
            # get the inputs; data is a list of [inputs, labels]
            imgs, labels, cc, demo, lab = data
            if args.USE_TEXT:
                cc = cc.view(-1, tk_lim, cc.shape[3]).cuda(non_blocking=True).float()
                demo = demo.view(-1, 1, demo.shape[1]).cuda(non_blocking=True).float()
                lab = lab.view(-1, lab.shape[1], 1).cuda(non_blocking=True).float()
                sex = demo[:, :, 1].view(-1, 1, 1).cuda(non_blocking=True).float()
                age = demo[:, :, 0].view(-1, 1, 1).cuda(non_blocking=True).float()
            else:
                cc=None
                lab=None
                sex=None
                age=None
            imgs = imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            loss,preds = lucid(imgs, cc, lab, sex, age,labels)

            probs = torch.sigmoid(preds)
            outGT = torch.cat((outGT, labels), 0)
            outPRED = torch.cat((outPRED, probs.data), 0)

            pred_list.extend(list(torch.argmax(preds,1).cpu()))
            label_list.extend(list(torch.argmax(labels,1).cpu()))

        aurocIndividual = computeAUROC(outGT, outPRED, classCount=num_classes)
        aurocMean = np.array(aurocIndividual).mean()
        
        print('mean AUROC:' + str(aurocMean))
        
        for i in range (0, len(aurocIndividual)):
            print(disease_list[i] + ': '+str(aurocIndividual[i]))

        print(args.USE_TEXT)
        print(preds[:5])
        print(labels[:5])
        print(confusion_matrix(pred_list,label_list))
        print(accuracy_score(pred_list,label_list))
        print(f1_score(pred_list,label_list,average="macro"))
    
    torch.save(best_model,f"best_model_from_scratch_{args.USE_TEXT}.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--CLS', action='store', dest='CLS', required=True, type=int)
    parser.add_argument('--BSZ', action='store', dest='BSZ', required=True, type=int)
    parser.add_argument('--DATA_DIR', action='store', dest='DATA_DIR', required=True, type=str)
    parser.add_argument('--SET_TYPE', action='store', dest='SET_TYPE', required=True, type=str)
    parser.add_argument('--USE_TEXT', action='store', dest='USE_TEXT', required=True, type=str)
    args = parser.parse_args()
    test(args)
