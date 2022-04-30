import matplotlib.pyplot as plt
import os
import pickle as pkl

DIR = 'pickle-files-roc'
pkl_files = os.listdir(DIR)
print(pkl_files)
# exit(0)
# pkl_files = ['fta_resnet.pkl', 'fta_google.pkl', 'fta_inception.pkl', 'fta_inception_aux.pkl', 'fta_squeezenet.pkl', 'fta_shuffle.pkl']

plt.figure(figsize=(6,5))
for idx, file_name in enumerate(pkl_files):
    with open(os.path.join(DIR,pkl_files[idx]),'rb') as f:
        data = pkl.load(f)
        fpr,tpr,aucc = data[1]
    plt.plot([0,1],[0,1],linestyle='dashed')
    if 'google' in file_name:
        plt.plot(fpr,tpr,label=f'GoogleNet: {aucc:.2f}')
    elif 'aux' in file_name and 'inception' in file_name:
        plt.plot(fpr,tpr,label=f'Inception_aux: {aucc:.2f}')
    elif 'resnet' in file_name:
        plt.plot(fpr,tpr,label=f'ResNet: {aucc:.2f}')
    elif 'shuffle' in file_name:
        plt.plot(fpr,tpr,label=f'ShuffleNet: {aucc:.2f}')
    elif 'squeezenet' in file_name:
        plt.plot(fpr,tpr,label=f'SqueezeNet: {aucc:.2f}')
    elif 'inception' in file_name:
        plt.plot(fpr,tpr,label=f'Inception: {aucc:.2f}')
    elif 'b0' in file_name:
        plt.plot(fpr,tpr,label=f'Efficientb0: {aucc:.2f}')
    elif 'b2' in file_name:
        plt.plot(fpr,tpr,label=f'Efficientb2: {aucc:.2f}')    
    elif 'mob' in file_name:
        plt.plot(fpr,tpr,label=f'MobileNetv2: {aucc:.2f}')



plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('roc-plot')
plt.legend()
plt.savefig('roc_all_models.png')
plt.savefig('roc__models.svg',format='svg')
plt.show()