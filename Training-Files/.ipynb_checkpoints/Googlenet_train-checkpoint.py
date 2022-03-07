import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import random
import PIL 
import time
from PIL import Image
from sklearn.metrics import classification_report,auc,roc_curve,precision_recall_fscore_support

#----------------------------------------paths and hyper parameters-----------------------------
config = dict(
    train_path = '../../../dataset/RetinalDiseaseData/Retinaldata/Training_Set/Training_Set/Training',
    val_path = '../../../dataset/RetinalDiseaseData/Retinaldata/Evaluation_Set/Evaluation_Set/Validation',
    test_path = '../../../dataset/RetinalDiseaseData/Retinaldata/Test_Set/Test_Set/Test',
    train_csv = '../../../dataset/RetinalDiseaseData/Retinaldata/Training_Set/Training_Set/RFMiD_Training_Labels.csv',
    val_csv = '../../../dataset/RetinalDiseaseData/Retinaldata/Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv',
    test_csv= '../../../dataset/RetinalDiseaseData/Retinaldata/Test_Set/Test_Set/RFMiD_Testing_Labels.csv',
    BATCH_SIZE=8,
    IMAGE_HEIGHT= 512,
    IMAGE_WIDTH = 770,
    lr=0.001,
    EPOCHS=30,
    pin_memory=True,
    num_workers=4,
    gpu_id="5",
    SEED=42,
    return_logs=False,
    saved_path = '../saved-models/googlenet_v1.pth'
)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]=config['gpu_id']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('gpu_id: ',config['gpu_id'])
print(device)

random.seed(config['SEED'])
np.random.seed(config['SEED'])
torch.manual_seed(config['SEED'])
torch.cuda.manual_seed(config['SEED'])
torch.backends.cudnn.benchmarks = True
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#-------------------------------------------dataset and dataloader-----------------------------------
class CustomTraindata():
    def __init__(self):
        self.train_df = pd.read_csv(config['train_csv'])
        self.images = config['train_path']
        self.transformation2 = transforms.ToPILImage()
        self.transformation = transforms.Compose([
            transforms.Resize((config['IMAGE_HEIGHT'],config['IMAGE_WIDTH'])),
            transforms.ToTensor()])
    def __len__(self):
        return len(self.train_df)
    
    def __getitem__(self,idx):
        df_row = self.train_df.iloc[idx,:]
        img_id, disease = df_row['ID'],df_row['Disease_Risk']
        # diesease == 1 (infection) else no infection
        img_name = f'{img_id}.png'
        
        img = Image.open(os.path.join(self.images,img_name)).convert("RGB")
        width,height = img.size
        scaling = width / height
        img = self.transformation(img)
        return (img,disease)
    
class CustomValdata():
    def __init__(self):
        self.val_df = pd.read_csv(config['val_csv'])
        self.images = config['val_path']
        self.transformation2 = transforms.ToPILImage()
        self.transformation = transforms.Compose([
            transforms.Resize((config['IMAGE_HEIGHT'],config['IMAGE_WIDTH'])),
            transforms.ToTensor()])
    def __len__(self):
        return len(self.val_df)
    
    def __getitem__(self,idx):
        df_row = self.val_df.iloc[idx,:]
        img_id, disease = df_row['ID'],df_row['Disease_Risk']
        # diesease == 1 (infection) else no infection
        img_name = f'{img_id}.png'
        
        img = Image.open(os.path.join(self.images,img_name)).convert("RGB")
        width,height = img.size
        scaling = width / height
        img = self.transformation(img)
        return (img,disease)
    
class CustomTestdata():
    def __init__(self):
        self.test_df = pd.read_csv(config['test_csv'])
        self.images = config['test_path']
        self.transformation2 = transforms.ToPILImage()
        self.transformation = transforms.Compose([
            transforms.Resize((config['IMAGE_HEIGHT'],config['IMAGE_WIDTH'])),
            transforms.ToTensor()])
    def __len__(self):
        return len(self.test_df)
    
    def __getitem__(self,idx):
        df_row = self.test_df.iloc[idx,:]
        img_id, disease = df_row['ID'],df_row['Disease_Risk']
        # diesease == 1 (infection) else no infection
        img_name = f'{img_id}.png'
        
        img = Image.open(os.path.join(self.images,img_name)).convert("RGB")
        width,height = img.size
        scaling = width / height
        img = self.transformation(img)
        return (img,disease)
    

train_data = CustomTraindata()
val_data = CustomValdata()
test_data = CustomTestdata()

train_loader = DataLoader(train_data,batch_size=config['BATCH_SIZE'],shuffle=True,pin_memory=config['pin_memory'],num_workers=config['num_workers'])
val_loader = DataLoader(val_data,batch_size=config['BATCH_SIZE'],shuffle=True,pin_memory=config['pin_memory'],num_workers=config['num_workers'])
test_loader = DataLoader(test_data,batch_size=config['BATCH_SIZE'],shuffle=True,pin_memory=config['pin_memory'],num_workers=config['num_workers'])

print(int(len(train_loader)*config['BATCH_SIZE']))
print(int(len(test_loader)*config['BATCH_SIZE']))
print(int(len(val_loader)*config['BATCH_SIZE']))
    
#----------------------------------------------------google net-----------------------------------------------------
class BasicConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, **kwargs):
    super(BasicConv2d,self).__init__()
    self.conv = nn.Conv2d(in_channels,out_channels,bias=False,**kwargs)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()

  def forward(self,x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.relu(x)
    return x

class Inception(nn.Module):
  def __init__(self,in_channels,b1,b2_int,b2,b3_int,b3,b4):
    super(Inception,self).__init__()
    self.branch1 = BasicConv2d(in_channels,b1,kernel_size=1,stride=1)
    self.branch2 = nn.Sequential(
                    BasicConv2d(in_channels,b2_int,kernel_size=1,stride=1),
                    BasicConv2d(b2_int,b2,kernel_size=3,stride=1,padding=1),
    )
    self.branch3 = nn.Sequential(
                    BasicConv2d(in_channels,b3_int,kernel_size=1,stride=1),
                    BasicConv2d(b3_int,b3,kernel_size=3,stride=1,padding=1)
    )
    self.branch4 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1,ceil_mode=True),
                    BasicConv2d(in_channels,b4,kernel_size=1,stride=1)
    )
  def forward(self,x):
    o1 = self.branch1(x)
    o2 = self.branch2(x)
    o3 = self.branch3(x)
    o4 = self.branch4(x)
    # print(o1.shape,o2.shape,o3.shape,o4.shape)
    o = torch.cat((o1,o2,o3,o4),dim=1)
    return o;


class GoogleNet(nn.Module):
  def __init__(self,in_channels=3,n_classes=2):
    super(GoogleNet,self).__init__()
    self.in_channels = in_channels
    self.nc = n_classes
    self.conv1 = BasicConv2d(self.in_channels,64,kernel_size=7,stride=2,padding=3)
    self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=0,dilation=1,ceil_mode=True)
    self.conv2 = BasicConv2d(64,64,kernel_size=1,stride=1)
    self.conv3 = BasicConv2d(64,192,kernel_size=3,stride=1,padding=1)
    self.maxpool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=0,dilation=1,ceil_mode=True)
    self.inception3a = Inception(192,64,96,128,16,32,32)
    self.inception3b = Inception(256,128,128,192,32,96,64)
    self.maxpool3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=0,dilation=1,ceil_mode=True)
    self.inception4a = Inception(480,192,96,208,16,48,64)
    self.inception4b = Inception(512,160,112,224,24,64,64)
    self.inception4c = Inception(512,128,128,256,24,64,64)
    self.inception4d = Inception(512,112,144,288,32,64,64)
    self.inception4e = Inception(528,256,160,320,32,128,128)
    self.maxpool4 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,ceil_mode=True)
    self.inception5a = Inception(832,256,160,320,32,128,128)
    self.inception5b = Inception(832,384,192,384,48,128,128)
    self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    self.dropout = nn.Dropout(p = 0.2,inplace=False)
    self.fc = nn.Linear(1024,self.nc)

  def forward(self,x):
    x = self.conv1(x)
    x = self.maxpool1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.maxpool2(x)
    x = self.inception3a(x)
    x = self.inception3b(x)
    x = self.maxpool3(x)
    x = self.inception4a(x)
    x = self.inception4b(x)    
    x = self.inception4c(x)    
    x = self.inception4d(x)    
    x = self.inception4e(x)
    x = self.maxpool4(x)
    x = self.inception5a(x)
    x = self.inception5b(x)
    x = self.avgpool(x)    
    x = x.flatten(start_dim = 1)
    x = self.dropout(x)
    x = self.fc(x)
    return x;


#-----------------------------------------------necessary functions for inferencing and training---------------------------------
transformations = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def train(model,lossfunction,optimizer,n_epochs=200,return_logs=False):
  tval = {'valloss':[],'valacc':[],'trainacc':[],"trainloss":[]}
  starttime = time.time()
  for epochs in range(n_epochs):
      model.train()
      cur_loss = 0
      curacc = 0
      len_train = len(train_loader)
      for idx , (data,target) in enumerate(train_loader):
          data = transformations(data)    
          data = data.to(device)
          target = target.to(device)

          scores = model(data)    
          loss = lossfunction(scores,target)
          cur_loss += loss.item() / (len_train)
          scores = F.softmax(scores,dim = 1)
          _,predicted = torch.max(scores,dim = 1)
          correct = (predicted == target).sum()
          samples = scores.shape[0]
          curacc += correct / (samples * len_train)
          
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
        
          if return_logs:
              print('TrainBatchDone:{:d}'.format(idx),end='\r') 
  
      model.eval()

      valacc = 0;valloss = 0
      vl_len = len(val_loader)
      for idx ,(data,target) in enumerate(val_loader):
          
          data = transformations(data)
          data = data.to(device)
          target = target.to(device)
        
          correct = 0;samples=0
          with torch.no_grad():
              scores = model(data)
              loss = lossfunction(scores,target)
              scores =F.softmax(scores,dim=1)
              _,predicted = torch.max(scores,dim = 1)
              correct += (predicted == target).sum()
              samples += scores.shape[0]
              valloss += loss.item() / vl_len
              valacc += correct / (samples * vl_len)
                
          if return_logs:
              print('ValidnBatchDone:{:d}'.format(idx),end='\r') 


      model.train()
      
      # print(correct.get_device(),samples.get_device(),len(validate_loader).get_device())
      tval['valloss'].append(float(valloss))
      tval['valacc'].append(float(valacc))
      tval['trainacc'].append(float(curacc))
      tval['trainloss'].append(float(cur_loss))
      
      print('epoch:[{:d}/{:d}], TrainAcc:{:.3f}, TrainLoss:{:.3f}, ValAcc:{:.3f}, ValLoss:{:.3f}'.format(epochs+1,n_epochs,curacc,cur_loss,valacc,valloss)) 

  torch.save(model.state_dict(),config['saved_path'])
  time2 = time.time() - starttime
  print('done time {:.3f} hours'.format(time2/3600))
  return tval

def getparams(model):
    total_parameters = 0
    for name,parameter in model.named_parameters():
        if parameter.requires_grad:
            total_parameters += parameter.numel()
    print(f"total_trainable_parameters are : {round(total_parameters/1e6,2)}M")


def evaluate(model,loader,return_logs=False):
  model.eval()
  correct = 0;samples =0
  fpr_tpr_auc = {}
  pre_prob = []
  lab = []
  predicted_labels = []

  with torch.no_grad():
      for idx,(x,y) in enumerate(loader):
          x = transformations(x)
          x = x.to(device)
          y = y.to(device)
          # model = model.to(device)

          scores = model(x)
          predict_prob = F.softmax(scores)
          _,predictions = predict_prob.max(1)

          predictions = predictions.to('cpu')
          y = y.to('cpu')
          predict_prob = predict_prob.to('cpu')

          predicted_labels.extend(list(predictions.numpy()))
          pre_prob.extend(list(predict_prob.numpy()))
          lab.extend(list(y.numpy()))

          correct += (predictions == y).sum()
          samples += predictions.size(0)
        
          if return_logs:
              print('batches done : ',idx,end='\r')
      
      print('correct are {:.3f}'.format(correct/samples))
      

  lab = np.array(lab)
  predicted_labels = np.array(predicted_labels)
  pre_prob = np.array(pre_prob)
  fpr,tpr,_ = roc_curve(lab,pre_prob[:,1])
  aucc = auc(fpr,tpr)
  fpr_tpr_auc[1] = [fpr,tpr,aucc]
  model.train()
  with open('../pickle-files-roc/fta_google.pkl','wb') as f:
        pickle.dump(fpr_tpr_auc,f)
  return fpr_tpr_auc,lab,predicted_labels,pre_prob

def loss_acc_curve(tval):
    plt.figure(figsize=(5,4))
    plt.plot(list(range(1,config['EPOCHS']+1)),tval['trainloss'],label='train-loss')
    plt.plot(list(range(1,config['EPOCHS']+1)),tval['trainacc'],label='train-accuracy')
    plt.plot(list(range(1,config['EPOCHS']+1)),tval['valloss'],label='val-loss')
    plt.plot(list(range(1,config['EPOCHS']+1)),tval['valacc'],label='val-accuracy')
    plt.xlabel('epochs')
    plt.ylabel('loss/accuracy')
    plt.title('loss_accuracy')
    plt.legend()
    plt.savefig('../roc_loss_plots/loss-acc-googlenet.svg',format='svg')

def roc_plot(fta):
    fpr,tpr,aucc = fta[1]
    plt.figure(figsize=(5,4))
    plt.plot(fpr,tpr,label=f'auc: {aucc:.3f}')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('roc_googlenet')
    plt.legend()
    plt.savefig('../roc_loss_plots/roc-googlenet.svg',format='svg')
    

#---------------------------------------------train and test---------------------------------------------------------

CNN_arch = GoogleNet()

CNN_arch = CNN_arch.to(device)

lossfunction = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=CNN_arch.parameters(),lr=config['lr'])

getparams(CNN_arch)
history = train(CNN_arch,lossfunction,optimizer,n_epochs=config['EPOCHS'],return_logs=config['return_logs'])
loss_acc_curve(history)

test_fta,y_true,y_pred,prob = evaluate(CNN_arch,test_loader,return_logs=config['return_logs'])
roc_plot(test_fta)

print(classification_report(y_true,y_pred))
test_pre,test_rec,test_f1,_ = precision_recall_fscore_support(y_true,y_pred)

print('class-wise')
print(test_pre)
print(test_rec)
print(test_f1)

print('avg-out')
print(test_pre.mean())
print(test_rec.mean())
print(test_f1.mean())
