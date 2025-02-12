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
import torchvision.transforms as transforms
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import datasets, models

import os
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


import seaborn as sns
import pandas as pd

from sklearn.metrics import confusion_matrix
import pytorch_lightning as pl

from lightly.data import LightlyDataset
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import MoCoProjectionHead
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    update_momentum,
)
#Import spécifique à chaque model
from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.moco_transform import MoCoV2Transform
from lightly.utils.scheduler import cosine_schedule


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


# Configuration générale. Attention à bien choisir les bons paramètres et les bon emplacement de sauvegarde.
stop_random()
nb_Epoch = 100
batch_Size = 128
taille_Image = 28
augmentation=False
num_workers = 4 
dossier="Full"
dataset="Blood"#Derma
BNF = False
#Nomenclature:Nom_Model#Dossier#Epoch#BNF
Nom_Model = "MoCo"
model_Path = './PytorchModelV3/'+"MoCo_Model#"+"Full" +"#"+dataset+str(nb_Epoch) +"#" +"FALSE"+".pth"
path_Class = './PytorchModelV3/'+"MoCo_Class#"+dossier +"#"+dataset+str(nb_Epoch) +"#" +str(BNF)+".pth"
path_Train_SSL_M = dataset+"_SSL"
path_Train = dataset+"_"+dossier+"/Dataset/train"
path_val =  dataset+"_"+dossier+"/Dataset/val"
nb_Classe = len(next(os.walk(dataset+"_"+dossier+"/test"))[1])
print("Model choisi MoCo, avec le dossier: " +dossier+ ", avec "+ str(nb_Epoch) + " epoch")
print("Emplacement: " +model_Path)
print("Emplacement: " +path_Class)







# In[ ]:


#Source:[https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_moco_memory_bank.html]
#Source:[https://docs.lightly.ai/self-supervised-learning/examples/moco.html]

transform = MoCoV2Transform(input_size=28,gaussian_blur=0.0,)

train_classifier_transforms = torchvision.transforms.Compose(
    [      
        torchvision.transforms.ToTensor(),
    ])

if(augmentation):# Si l'on a besoin de faire de l'augmentation de données.
    print("augmentation:")
    train_classifier_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=28, scale=(0.3, 1.0), ratio=(1, 1)),   
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


val_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
    ])



dataset_train_model = LightlyDataset(input_dir=path_Train_SSL_M, transform=transform)
dataset_train_classifier = LightlyDataset(input_dir=path_Train, transform=train_classifier_transforms)
dataset_val = LightlyDataset(input_dir=path_val, transform=val_transforms)

print(train_classifier_transforms)



# In[ ]:


#Source:[https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_moco_memory_bank.html]

# Explication sur les paramètres des dataloaders :
#      shuffle --> mélange les images, à utiliser surtout pour les données d'entraînement du modèle SSL.
#      drop_last --> sinon, parfois, l'entraînement peut planter à un epoch.

stop_random()
dataloader_train_model = torch.utils.data.DataLoader(
    dataset_train_model,
    batch_size=batch_Size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
    persistent_workers=True,
)

dataloader_train_classifier = torch.utils.data.DataLoader(
    dataset_train_classifier,
    batch_size=batch_Size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
    persistent_workers=True,
)

dataloader_val = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=batch_Size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
    persistent_workers=True,
)


# In[ ]:


#Source:[https://docs.lightly.ai/self-supervised-learning/examples/moco.html]
stop_random()
class Moco(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = MoCoProjectionHead(512, 1024, 1024)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NTXentLoss(memory_bank_size=(4096, 1024))
        self.train_losses = []

    def forward(self, x):
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query
        
    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key
    
    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, nb_Epoch, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        x_query, x_key = batch[0]
        query = self.forward(x_query)
        key = self.forward_momentum(x_key)
        loss = self.criterion(query, key)
        self.log("train_loss_ssl", loss)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=False)
        return loss


    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim
        
    def on_train_epoch_end(self):
        epoch_loss = self.trainer.callback_metrics['train_loss']
        self.train_losses.append(epoch_loss.item())
        print(f'Epoch {self.current_epoch}: Train Loss: {epoch_loss.item()}')   

    def on_train_end(self):# Affiche la loss à la fin de l'entraînement.
        print("LOSS:")
        for i in self.train_losses:
            temp =f"{i:.5f}".replace('.', ',')
            print(temp )



# In[ ]:


#Source:[https://saturncloud.io/blog/how-to-use-class-weights-with-focal-loss-in-pytorch-for-imbalanced-multiclass-classification/]
stop_random()
import torch.optim as optim
trainset = dataset_train_classifier
# Parce que les dataloaders de Lightly n'ont pas l'attribut "targets", donc on est obligé d'utiliser cette méthode.


nb_Element = [0] * nb_Classe
i = 0
total = 0
for sous_d in os.listdir(path_Train):
    sous_d = os.path.join(path_Train, sous_d)
    nb_Element[i] = len(os.listdir(sous_d))
    i = i +1
for k in range(0, nb_Classe):
    total = total + nb_Element[k]

class_weights = []
for exemple in nb_Element:
    poid = (total-exemple) / (total)
    class_weights.append(poid)
    print(poid)


#Source:[https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html]

class_weights = torch.tensor(class_weights,dtype=torch.float32).cuda()
criterion_ = nn.CrossEntropyLoss(weight=class_weights)
print(class_weights)


# In[ ]:


#Source:[https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_moco_memory_bank.html]
class Classifier(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
   
        self.train_losses_tab = []
        self.train_accuracy_tab = []   
        self.val_losses_tab = []
        self.val_accuracy_tab = []

        self.train_losses = 0
        self.train_accuracy = 0
        self.val_losses = 0
        self.val_accuracy = 0
        
        self.t_iteration = 0
        self.v_iteration = 0
        
        # freeze the backbone
        if(not(BNF)):
            deactivate_requires_grad(backbone)              
        self.fc1 = nn.Linear(512, 2048)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2048, nb_Classe)
        
        self.validation_step_outputs = []
        self.criterion = criterion_
        
    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        y_hat = x
        return y_hat

    def training_step(self, batch, batch_idx):  
        x, y, _ = batch
        y_hat = self.forward(x)
        loss =  self.criterion(y_hat, y)
        # Calcul et enregistrement des données utiles.
        
        t_acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.train_losses = self.train_losses + loss.item()
        self.train_accuracy = self.train_accuracy + t_acc.item()
        self.t_iteration =  self.t_iteration +1
        return loss
       
    def on_train_epoch_end(self):

        self.train_losses_tab.append(self.train_losses/self.t_iteration)
        self.train_accuracy_tab.append(self.train_accuracy/self.t_iteration)
        self.train_losses = 0
        self.train_accuracy = 0
        self.t_iteration = 0

        
    def on_train_end(self):    
        
        self.train_losses= np.array(self.train_losses_tab)
        self.train_accuracy= np.array(self.train_accuracy_tab)

        self.val_losses= np.array(self.val_losses_tab)
        self.val_accuracy= np.array(self.val_accuracy_tab)
        
        #Enregistrement 
        import csv
        titre = ['training_loss','validation_loss','training_accuracy','validation_accuracy']

        tableaux_complet = list(zip(self.train_losses,self.train_accuracy,self.val_losses,self.val_accuracy))
        path_save_data = './Entrainement/'+ dataset + "/Moco_Class#"+dossier +"#"+str(nb_Epoch) +"#" +str(BNF)+".csv"
        print(path_save_data)
        with open(path_save_data, mode='w', newline='', encoding='utf-8') as fichier:  
            writer = csv.writer(fichier)
            writer.writerow(titre)
            for i in range(len(self.train_losses)):
                temp1 =f"{self.train_losses[i]:.5f}".replace('.', ',')
                temp2 =f"{self.val_losses[i]:.5f}".replace('.', ',')
                temp3 =f"{self.train_accuracy[i]:.5f}".replace('.', ',')
                temp4 =f"{self.val_accuracy[i]:.5f}".replace('.', ',')   
                writer.writerow([temp1, temp2, temp3, temp4])
            
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss =  self.criterion(y_hat, y)
        v_acc = (y_hat.argmax(dim=1) == y).float().mean()
        
        self.val_losses =self.val_losses + loss.item()
        self.val_accuracy = self.val_accuracy + v_acc.item()
        self.v_iteration = self.v_iteration +1

    def on_validation_epoch_end(self):
        
        self.val_losses_tab.append(self.val_losses/self.v_iteration)
        self.val_accuracy_tab.append(self.val_accuracy/self.v_iteration)
        
        self.val_losses = 0
        self.val_accuracy = 0
        self.v_iteration = 0

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.trainer.max_epochs)
        return [optim], [scheduler]



# In[ ]:


torch.set_float32_matmul_precision('medium')# Réduit légèrement la précision, mais améliore la rapidité de l'entraînement.


# In[ ]:


#Entrainement du model
stop_random()
model = Moco()
print("Entrainement du model: " + Nom_Model)
trainer = pl.Trainer(max_epochs=nb_Epoch, devices=1, accelerator="gpu")
trainer.fit(model, dataloader_train_model)

torch.save(model.state_dict(), model_Path)



# In[ ]:


#Chargement du model 
model = Moco()
model.load_state_dict(torch.load(model_Path))
print(model_Path)


# In[ ]:


#Source:[https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_moco_memory_bank.html]
stop_random()
import contextlib
import os
import sys

print("Entrainement du classifieur pour le model: " + Nom_Model)

if(not(BNF)):
    model.eval()
    print("Eval")
else:
    model.train()
    print("Train")

log_file_path = 'training_output.log'# PyTorch n'aime pas afficher beaucoup de texte, donc on le stocke dans un fichier à la place.
with open(log_file_path, 'w') as log_file_path:
    with contextlib.redirect_stdout(log_file_path), contextlib.redirect_stderr(log_file_path): 
        classifier = Classifier(model.backbone)
        trainer = pl.Trainer(max_epochs=nb_Epoch, devices=1, accelerator="gpu")
        trainer.fit(classifier, dataloader_train_classifier, dataloader_val) 

print("FINITO")


#Sauvegarde le classificateur
torch.save(classifier.state_dict(), path_Class)



# In[ ]:


#Charge le classificateur 
classi =  Classifier(model.backbone)
classi.load_state_dict(torch.load(path_Class))
classifier= classi


# In[ ]:


stop_random()
# Partie test : va tester le modèle sur l'ensemble du dossier de test, et donner les prédictions par classe ainsi qu'une matrice de confusion.
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix


print("Model choisi:" + Nom_Model +",Avec "+ str(nb_Epoch)+" Epoch,et le dossier " +dossier)
print(model_Path)
print("NB de classe: " + str(nb_Classe))


class_names=['[0]', '[1]', '[2]', '[3]', '[4]', '[5]', '[6]' ,'[7]']
if(nb_Classe == 8):
    class_names=['[0]', '[1]', '[2]', '[3]', '[4]', '[5]', '[6]' ,'[7]']
else:
    class_names=['[0]', '[1]', '[2]', '[3]', '[4]', '[5]', '[6]']

total_Local = 0
total = 0
nb_true_pred = 0
total_AC = 0
taille_dossier= [0 for x in range(nb_Classe)]

predictions= [[0 for x in range(nb_Classe)] for y in range(nb_Classe)] 


for j in range(0,nb_Classe):
    image_folder = dataset +"_Full/test/"
   
    image_folder = image_folder + "[" + str(j) + "]"
    nb_true_pred = 0
    taille_dossier[j] = len(os.listdir(image_folder))
    classifier.eval()
  
    for filename in os.listdir(image_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
        
            image = Image.open(image_path).convert('RGB')
            image_tensor = val_transforms(image)  
            image_tensor = image_tensor.unsqueeze(0) 
            
            with torch.no_grad():
 
                output = classifier(image_tensor)
                probabilities = F.softmax(output, dim=1)
                
                _, predicted_class = torch.max(output, 1)
                predicted_class_name = class_names[predicted_class.item()]
                
                confidence = torch.max(probabilities).item() * 100
                true_class=os.path.basename(os.path.dirname(image_path))
                temp = predicted_class_name
                temp = temp.replace('[','')
                temp = temp.replace(']','')
                predictions[j][int(temp)] = predictions[j][int(temp)] +1    
                if(true_class == predicted_class_name):
                    nb_true_pred = nb_true_pred +1
    print("Classe:"+ str(j) + f" % de bonne prédiction:" + str((nb_true_pred/taille_dossier[j])* 100) +"%" )
    total = total + taille_dossier[j]
    total_AC = total_AC + nb_true_pred
    
print("Accuray général:" +str(( total_AC / total)*100)+"%")

for j in range(0,nb_Classe):
    for i in range (0,nb_Classe):
        predictions[j][i] = (predictions[j][i]/taille_dossier[j])*100
matrice_C = np.array(predictions)
plt.figure(figsize=(6, 6))
sns.heatmap(matrice_C, annot=True, fmt='.2f', cmap='Blues')
plt.xlabel('Prédiction')
plt.ylabel('Vrai classe')
plt.title('Matrice')
image_Path = 'M_image_'+dataset+'/'+str(nb_Epoch) +"_" + "EPOCH"+"_" + dossier +"_"+ Nom_Model+"_"+str(BNF)+".png"
print(image_Path)
plt.savefig(image_Path)
plt.show()



# In[ ]:


# Utilisé pour copier plus simplement les résultats dans un fichier Excel.
total2=0
for j in range(0,nb_Classe):
    total2 = total2 + predictions[j][j]
    temp =f"{predictions[j][j]:.2f}".replace('.', ',')
    print(temp)
temp =f"{(total_AC / total)*100:.2f}".replace('.', ',')

print(temp)
temp =f"{total2/nb_Classe:.2f}".replace('.', ',')
print(temp)



