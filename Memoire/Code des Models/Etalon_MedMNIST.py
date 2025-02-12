#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Chaque bloc possède un lien vers la source originale, formaté comme ceci : #Source: []

# Rappels avant de lancer le programme :
#   - bien vérifier le dataset et le dossier utilisé.
#   - bien vérifier les paramètres et les dossiers de sauvegarde.
#   - faire attention de ne pas lancer le modèle alors qu'il a déjà été entraîné.
#   - Si le premier epoch est très lent, c’est normal, il suffit d'attendre un peu que les données soient chargées.

# Cette cellule contient les imports nécessaires au bon fonctionnement du code.

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms.v2 as transformsV2
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import datasets, models

import os
import numpy as np
import matplotlib.pyplot as plt
import random

from PIL import Image


import seaborn as sns
import pandas as pd

from sklearn.metrics import confusion_matrix

torch.__version__


# In[ ]:


# Fonction utilisée pendant le programme pour donner une seed aux différentes fonctions aléatoires. 
# A mettre au début de chaque cellule.
# Source : [https://pytorch.org/docs/stable/notes/randomness.html]
def stop_random():
    seed=621
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    


# In[ ]:


# Configuration générale. Attention à bien choisir les bons paramètres et le bon emplacement de sauvegarde.
stop_random()
nb_Epoch = 100
batch_Size = 128
taille_Image = 28
augmentation=False
dossier = "Full"
dataset = "Blood"#Derma
BNF = False
#Nomenclature:Nom_Model#Dossier#Epoch#BNF
model_Path = './PytorchModelV3/'+"Res18#"+dossier +"#"+dataset+str(nb_Epoch) +"#" +str(BNF)+".pth"
path_Train = dataset+"_"+dossier+"/Dataset/train"
path_Val =  dataset+"_"+dossier+"/Dataset/val"
nb_Classe = len(next(os.walk(dataset+"_"+dossier+"/test"))[1])
print("Model choisi:RES18, avec le dossier: " +dossier+ ", avec "+ str(nb_Epoch) + " epoch")
print("Emplacement: " +model_Path)


# In[ ]:


stop_random()
transform_Train = transformsV2.Compose([
    transformsV2.ToTensor(),

])

# Si l'on a besoin de faire de l'augmentation de données.
if(augmentation):
    print("augmentation:")
    transform_Train = transformsV2.Compose([
    transformsV2.RandomResizedCrop(size=28, scale=(0.3, 1.0), ratio=(1, 1)),   
    transformsV2.RandomHorizontalFlip(),
    transformsV2.ToTensor(),
    transformsV2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform = transformsV2.Compose([
    transformsV2.ToTensor(),  
])

trainset= datasets.ImageFolder(root=path_Train, transform=transform_Train)
valset = datasets.ImageFolder(root=path_Val, transform=transform)

print(transform_Train)


# In[ ]:


#Source:[https://pytorch.org/hub/pytorch_vision_resnet/]
# On récupère un modèle ResNet18 non entraîné.
stop_random()
import torch
import torch.nn as nn
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, nb_Classe)
device = torch.device('cuda:0')
net = model
net.to(device)



# In[ ]:


#Source:[https://saturncloud.io/blog/how-to-use-class-weights-with-focal-loss-in-pytorch-for-imbalanced-multiclass-classification/]
stop_random()
# Chargement des données et gestion des poids des classes.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_Size,shuffle=True, num_workers=4)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_Size,shuffle=False, num_workers=4)
classes = trainset.classes
num_classes = len(trainset.classes)

nom_Classe= trainset.targets
training_Nb_Exemple = torch.bincount(torch.tensor(nom_Classe))
total = len(trainset)


class_weights = []
for exemple in training_Nb_Exemple:
    poid = (total-exemple) / (total)
    class_weights.append(poid)
    print(poid)
    
#Source:[https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html]
class_weights = torch.tensor(class_weights,dtype=torch.float32).cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)


# In[ ]:


#Source:[https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html]
#Source:[https://saturncloud.io/blog/calculating-the-accuracy-of-pytorch-models-every-epoch/]

stop_random()
import numpy as np
#training
t_accuracy_tab = []
t_loss_tab = []

#validation
v_accuracy_tab = []
v_loss_tab = []

epochs = []

nb_correct= 0
nb_exemple = 0
lr = 0.1
optimizer = optim.SGD(net.parameters(), lr, momentum=0.9)

for epoch in range(nb_Epoch): 
    t_accuracy = 0
    t_loss = 0 
    
    v_accuracy = 0
    v_loss = 0
  
    nb_exemple = 0
    nb_correct = 0
    net.train()

    if(epoch%20==0):
        lr = lr/2
        print(lr)
        if(lr < 0.0005): 
            lr = 0.0005
        optimizer = optim.SGD(net.parameters(), lr, momentum=0.9)
    
    for i, data in enumerate(trainloader, 0):
       
        inputs, labels = data[0].to(device), data[1].to(device)# GPU
        optimizer.zero_grad()
        outputs = net(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        t_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        
        nb_exemple += labels.size(0)   
        nb_correct += (predicted == labels).sum().item()       
    
    t_loss = t_loss/len(trainloader)#Moyenne de la loss sur tout l'epoch
    t_accuracy = 100*(nb_correct /nb_exemple)#Accuracy moyenne sur un epoch
    t_accuracy_tab.append(t_accuracy)
    t_loss_tab.append(t_loss)
    
    nb_exemple = 0
    nb_correct = 0
    
    net.eval()

    if(epoch%5 == 0):   #Collecte de données      
        with torch.inference_mode():
            for data in valloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                v_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                
                nb_exemple += labels.size(0)   
                nb_correct += (predicted == labels).sum().item()  
        v_loss  =  v_loss/len(valloader)
        v_accuracy = 100*(nb_correct /nb_exemple)
        v_accuracy_tab.append(v_accuracy)
        v_loss_tab.append(v_loss)
    else:
        v_accuracy_tab.append(v_accuracy)
        v_loss_tab.append(v_loss)
    
    epochs.append(epoch)
    
    print(f'Epoch {epoch+1}:T_Accuracy = {t_accuracy:.2f}% T_Loss:= {(t_loss):.2f}')
    print(f'Epoch {epoch+1}:V_Accuracy = {v_accuracy:.2f}% V_Loss:= {(v_loss):.2f}')
   

#loss_Brut = t_loss_tab
t_loss_temp= np.array(t_loss_tab)
t_accuracy_temp= np.array(t_accuracy_tab)

v_loss_temp= np.array(v_loss_tab)
v_accuracy_temp= np.array(v_accuracy_tab)



# In[ ]:


# Graphique des données et enregistrement des données au format Excel.
stop_random()
print('Entrainement fini')
plt.plot(epochs, t_loss_temp, label='Training_Loss')
plt.plot(epochs, v_loss_temp, label='Validation_Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.title('Accuracy')
plt.plot(epochs, t_accuracy_temp, label='Training_Accuracy')
plt.plot(epochs, v_accuracy_temp, label='Validation_Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Enregistrement 
import csv
titre = ['training_loss','validation_loss','training_accuracy','validation_accuracy']

tableaux_complet = list(zip(t_loss_temp,v_loss_temp,t_accuracy_temp,v_accuracy_temp))
path_save_data = './Entrainement/'+ dataset + "/Res18#"+dossier +"#"+str(nb_Epoch) +"#" +str(BNF)+".csv"
print(path_save_data)
with open(path_save_data, mode='w', newline='', encoding='utf-8') as fichier:  
    writer = csv.writer(fichier)
    writer.writerow(titre)
    for i in range(len(t_loss_temp)):
        temp1 =f"{t_loss_temp[i]:.5f}".replace('.', ',')
        temp2 =f"{v_loss_temp[i]:.5f}".replace('.', ',')
        temp3 =f"{t_accuracy_temp[i]:.5f}".replace('.', ',')
        temp4 =f"{v_accuracy_temp[i]:.5f}".replace('.', ',')   
        writer.writerow([temp1, temp2, temp3, temp4])
print("Fichier sauvegarder: "+ path_save_data)




# In[ ]:


#Enregistrement du model
torch.save(net.state_dict(), model_Path)


# In[ ]:


#Chargement du model
net.load_state_dict(torch.load(model_Path))


# In[ ]:


# Partie test : va tester le modèle sur l'ensemble du dossier de test, et donner les prédictions par classe ainsi qu'une matrice de confusion.
stop_random()
print("NB de classe: " + str(nb_Classe))
image_transform = transform 
total = 0
nb_true_pred = 0
total_AC = 0
taille_Dossier= [0 for x in range(nb_Classe)]
prediction= [[0 for x in range(nb_Classe)] for y in range(nb_Classe)] 
model = net
model.eval()
image_folder = dataset+'_Full/test/'
items = os.listdir(image_folder)
print(len(items))

for j in range(0,len(items)):
    image_folder = dataset+'_Full/test/'
    image_folder = image_folder + "[" + str(j) + "]"
    nb_true_pred = 0
    taille_Dossier[j] = len(os.listdir(image_folder))
    for filename in os.listdir(image_folder):  
        if filename.endswith(".png"):#Avant Jpeg
            image_path = os.path.join(image_folder, filename) 
            image = Image.open(image_path).convert('RGB')
            image_tensor = image_transform(image) 
            image_tensor = image_tensor.to(device)
            image_tensor = image_tensor.unsqueeze(0)
            with torch.inference_mode():
                output = model(image_tensor)             
                probabilities = F.softmax(output, dim=1)
            class_names = classes
            _, predicted_class = torch.max(output, 1)
            predicted_class_name = class_names[predicted_class.item()]          
            confidence = torch.max(probabilities).item() * 100    
            true_class=os.path.basename(os.path.dirname(image_path))
            temp = predicted_class_name
            temp = temp.replace('[','')
            temp = temp.replace(']','')
           
            prediction[j][int(temp)] =prediction[j][int(temp)] +1      
            if(true_class == predicted_class_name):
                nb_true_pred = nb_true_pred +1
    print("Classe:"+ str(j) + f" % de bonne prédiction:" + str((nb_true_pred/taille_Dossier[j])* 100) +"%" )
    total = total + taille_Dossier[j]
    total_AC = total_AC + nb_true_pred
    
print("Accuray général:" +str(( total_AC / total)*100)+"%")
for j in range(0,len(items)):
    for i in range (0,len(items)):
        prediction[j][i] = (prediction[j][i]/taille_Dossier[j])*100
        
matrice_C = np.array(prediction)
plt.figure(figsize=(6, 6))
sns.heatmap(matrice_C, annot=True, fmt='.2f', cmap='Blues')
plt.xlabel('Prédiction')
plt.ylabel('Vrai classe')
plt.title('Matrice')
image_Path = 'M_image_'+dataset+'/'+str(nb_Epoch) +"_" + "EPOCH"+"_" + dossier + "RES18.png"
plt.savefig(image_Path)
plt.show()



# In[ ]:


# Utilisé pour copier plus simplement les résultats dans un fichier Excel.
total2=0
for j in range(0,nb_Classe):
    total2 = total2 + prediction[j][j]
    temp =f"{prediction[j][j]:.2f}".replace('.', ',')
    print(temp)
temp =f"{(total_AC / total)*100:.2f}".replace('.', ',')
print(temp)
temp =f"{total2/nb_Classe:.2f}".replace('.', ',')
print(temp)


# In[ ]:




