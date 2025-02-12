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
#from lightly.models import ResclasGenerator
from lightly.models.modules.heads import MoCoProjectionHead
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    update_momentum,
)
#Import spécifique à chaque model
#Moco:
from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.moco_transform import MoCoV2Transform
from lightly.utils.scheduler import cosine_schedule

#SimCLR:
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.transforms import SimCLRTransform, utils

#AIM
from lightly.models import utils
from lightly.models.modules import AIMPredictionHead, MaskedCausalVisionTransformer
from lightly.transforms import AIMTransform
#MAE
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from lightly.transforms import MAETransform
import csv
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
BNF = True
dossier="Full"
dataset="Derma"
Moco_path_Class = './PytorchModelV3/'+"MoCo_Class#"+dossier +"#"+dataset+str(nb_Epoch) +"#" +str(BNF)+".pth"
SimCLR_path_Class = './PytorchModelV3/'+"SimCLR_Class#"+dossier +"#"+dataset+str(nb_Epoch) +"#" +str(BNF)+".pth"
AIM_path_Class = './PytorchModelV3/'+"AIM_Class#"+dossier +"#"+dataset+str(nb_Epoch) +"#" +str(BNF)+".pth"
MAE_path_Class = './PytorchModelV3/'+"MAE_Class#"+dossier +"#"+dataset+str(50) +"#" +str(BNF)+".pth"

print(Moco_path_Class)
path_val =  dataset+"_"+dossier+"/Dataset/val"
path_Train = dataset+"_"+dossier+"/Dataset/train"
nb_Classe = len(next(os.walk(dataset+"_"+dossier+"/test"))[1])


Model_Actif = "FSAAM" 
#FSM Sim et MoCo
#FAA Aim et MAE
#FSAAM Sim,AIM,MAE,MoCo



# In[ ]:


#Source:[https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_moco_memory_bank.html]

# Explication sur les paramètres des dataloaders :
#      shuffle --> mélange les images, à utiliser surtout pour les données d'entraînement du modèle SSL.
#      drop_last --> sinon, parfois, l'entraînement peut planter à un epoch.

stop_random()

val_transforms_MAE = torchvision.transforms.Compose(
    [      
        transforms.Resize(224),
        torchvision.transforms.ToTensor(),
    ])

val_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
    ])


import torch.optim as optim

#Car les dataloaders de lightly n’ont pas l’attribut targets, donc on est obligé d’utiliser cette méthode 
#Ici ne sert a pas a grand chose, mais garder pour la continuiter des scripts
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


#MoCo Model
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
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=False)#Log
        return loss


    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim
        
    def on_train_epoch_end(self):
        epoch_loss = self.trainer.callback_metrics['train_loss']
        self.train_losses.append(epoch_loss.item())
        print(f'Epoch {self.current_epoch}: Train Loss: {epoch_loss.item()}')   

    def on_train_end(self):
        print("LOSS:")
        for i in self.train_losses:
            temp =f"{i:.5f}".replace('.', ',')
            print(temp )



# In[ ]:


#Source:[https://docs.lightly.ai/self-supervised-learning/examples/simclr.html]
#Model SimCLR
stop_random()
class SimCLR(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SimCLRProjectionHead(512, 1024, 1024)
        self.criterion = NTXentLoss()
        self.train_losses = []

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1) = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=False)#Log
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)#0.06
        return optim
        
    def on_train_epoch_end(self):
        epoch_loss = self.trainer.callback_metrics['train_loss']
        self.train_losses.append(epoch_loss.item())
        print(f'Epoch {self.current_epoch}: Train Loss: {epoch_loss.item()}')
        
    def on_train_end(self):
        print("LOSS:")
        for i in self.train_losses:
            temp =f"{i:.5f}".replace('.', ',')
            print(temp )


# In[ ]:


#Source:[https://docs.lightly.ai/self-supervised-learning/examples/aim.html]

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.models import utils
from lightly.models.modules import AIMPredictionHead, MaskedCausalVisionTransformer
from lightly.transforms import AIMTransform

class AIM(pl.LightningModule):#
    def __init__(self) -> None:
        super().__init__()
        self.train_losses = []
        vit = MaskedCausalVisionTransformer(
            img_size=28,
            patch_size=4,
            embed_dim=280,
            depth=12,
            num_heads=1,
            qk_norm=False,
            class_token=False,
            no_embed_class=True,
        )
        utils.initialize_2d_sine_cosine_positional_embedding(
            pos_embedding=vit.pos_embed, has_class_token=vit.has_class_token
        )
        self.patch_size = vit.patch_embed.patch_size[0]
        self.num_patches = vit.patch_embed.num_patches

        self.backbone = vit
        self.projection_head = AIMPredictionHead(
           
            input_dim=vit.embed_dim, output_dim=3 * self.patch_size**2, num_blocks=1
        )

        self.criterion = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        views, targets = batch[0], batch[1]
        images = views[0] 
        batch_size = images.shape[0]

        mask = utils.random_prefix_mask(
            size=(batch_size, self.num_patches),
            max_prefix_length=self.num_patches - 1,
            device=images.device,
        )
        features = self.backbone.forward_features(images, mask=mask)

        features = self.backbone._pos_embed(features)
        predictions = self.projection_head(features)


        patches = utils.patchify(images, self.patch_size)
        patches = utils.normalize_mean_var(patches, dim=-1)

        loss = self.criterion(predictions, patches)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=False)#Log
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=1.5e-4)
        return optim
        
    def on_train_epoch_end(self):
        epoch_loss = self.trainer.callback_metrics['train_loss']
        self.train_losses.append(epoch_loss.item())
        print(f'Epoch {self.current_epoch}: Train Loss: {epoch_loss.item()}')
        
    def on_train_end(self):
        print("LOSS:")
        for i in self.train_losses:
            temp =f"{i:.5f}".replace('.', ',')
            print(temp )    


# In[ ]:


#Source:[https://docs.lightly.ai/self-supervised-learning/examples/mae.html]
#Model MAE


from vit_pytorch import ViT
import pytorch_lightning as pl
import torch
import torchvision
from timm.models.vision_transformer import vit_base_patch32_224
from torch import nn

from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from lightly.transforms import MAETransform

from timm import create_model


class MAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        num_classes = 8
        vit = vit_base_patch32_224()
        decoder_dim = 512
        
        self.mask_ratio = 0.75
        self.patch_size = vit.patch_embed.patch_size[0]
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length
        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=decoder_dim,
            in_chans=3,#NEW
            decoder_depth=1,
            decoder_num_heads=16,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        )
        self.criterion = nn.MSELoss()

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.decoder.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def training_step(self, batch, batch_idx):
        views = batch[0]
        images = views[0]  # views contains only a single view
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
        )

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        loss = self.criterion(x_pred, target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=False)#Log
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=1.5e-4)
        return optim

    def on_train_epoch_end(self):
        epoch_loss = self.trainer.callback_metrics['train_loss']
        self.train_losses.append(epoch_loss.item())
        print(f'Epoch {self.current_epoch}: Train Loss: {epoch_loss.item()}')
        
    def on_train_end(self):
        print("LOSS:")
        for i in self.train_losses:
            temp =f"{i:.5f}".replace('.', ',')
            print(temp )    


# In[ ]:


#Source:[https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_moco_memory_bank.html]
#Classificateur  Moco
class Classifier_MoCo(pl.LightningModule):
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


#Source:[https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_moco_memory_bank.html]
#Classificateur  SimCLR
class Classifier_SimCLR(pl.LightningModule):#Pour les variables ne pas oubliez les SELF
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
        
        titre = ['training_loss','validation_loss','training_accuracy','validation_accuracy']

        tableaux_complet = list(zip(self.train_losses,self.train_accuracy,self.val_losses,self.val_accuracy))
        path_save_data = './Entrainement/'+ dataset + "/SimCLR_Class#"+dossier +"#"+str(nb_Epoch) +"#" +str(BNF)+".csv"
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


#Source:[https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_moco_memory_bank.html]
#Classificateur AIM
class Classifier_AIM(pl.LightningModule):#Pour les variables ne pas oubliez les SELF
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
        
        self.fc0 = nn.Linear(49000, 512)
        deactivate_requires_grad(self.fc0)
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
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        y_hat = x
        return y_hat

    def training_step(self, batch, batch_idx):  
        x, y, _ = batch
        y_hat = self.forward(x)
        loss =  self.criterion(y_hat, y)

        
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
        titre = ['training_loss','validation_loss','training_accuracy','validation_accuracy']

        tableaux_complet = list(zip(self.train_losses,self.train_accuracy,self.val_losses,self.val_accuracy))
        path_save_data = './Entrainement/'+ dataset + "/AIM_Class#"+dossier +"#"+str(nb_Epoch) +"#" +str(BNF)+".csv"
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


#Source:[https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_moco_memory_bank.html]
#Classificateur MAE
class Classifier_MAE(pl.LightningModule):#Pour les variables ne pas oubliez les SELF
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        #Marche très bien
   
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
        self.fc1 = nn.Linear(768, 2048)
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
        titre = ['training_loss','validation_loss','training_accuracy','validation_accuracy']

        tableaux_complet = list(zip(self.train_losses,self.train_accuracy,self.val_losses,self.val_accuracy))
        path_save_data = './Entrainement/'+ dataset + "/MAE_Class#"+dossier +"#"+str(nb_Epoch) +"#" +str(BNF)+".csv"
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


torch.set_float32_matmul_precision('medium')
model_Moco =Moco()
model_SimClR =SimCLR()
model_AIM =AIM()
model_MAE =MAE()
#Charge les différents classificateurs 

class_Moco =  Classifier_MoCo(model_Moco.backbone)
class_SimCLR =  Classifier_SimCLR(model_SimClR.backbone)
class_AIM =  Classifier_AIM(model_AIM.backbone)
class_MAE =  Classifier_MAE(model_MAE.backbone)

class_Moco.load_state_dict(torch.load(Moco_path_Class))
class_SimCLR.load_state_dict(torch.load(SimCLR_path_Class))
class_AIM.load_state_dict(torch.load(AIM_path_Class))
class_MAE.load_state_dict(torch.load(MAE_path_Class))


# In[ ]:


stop_random()
# Partie test : va tester le modèle sur l'ensemble du dossier de test, et donner les prédictions par classe ainsi qu'une matrice de confusion.
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix


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
i = 0

for j in range(0,nb_Classe):
    image_folder = dataset +"_Full/test/"
   
    image_folder = image_folder + "[" + str(j) + "]"
    nb_true_pred = 0
    taille_dossier[j] = len(os.listdir(image_folder))
    
    class_Moco.eval()
    class_SimCLR.eval()
    class_AIM.eval()
    class_MAE.eval()

    
    output_SimCLR=None
    output_MoCo=None
    output_AIM=None
    output_MAE=None

    probabilities_MoCo=None
    probabilities_SimCLR=None
    probabilities_AIM=None
    probabilities_MAE=None
    probabilities=None
    

    for filename in os.listdir(image_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path).convert('RGB')
            image_tensor = val_transforms(image)  
            image_tensor = image_tensor.unsqueeze(0) 
            if(Model_Actif == "FAA" or Model_Actif == "FSAAM"):
                #Pour MAE car image en 224 px
                image_tensor_MAE = val_transforms_MAE(image)     
                image_tensor_MAE = image_tensor_MAE.unsqueeze(0) 
            
            with torch.no_grad():

                if(Model_Actif == "FSM" or Model_Actif == "FSAAM"):
                    output_MoCo  = class_Moco(image_tensor)
                    output_SimCLR= class_SimCLR(image_tensor)
                    
                    probabilities_MoCo= F.softmax(output_MoCo, dim=1)
                    probabilities_SimCLR= F.softmax(output_SimCLR, dim=1)
                    probabilities= probabilities_SimCLR*probabilities_MoCo
                    
                if(Model_Actif == "FAA" or Model_Actif == "FSAAM"):
                    output_AIM= class_AIM(image_tensor)
                    output_MAE= class_MAE(image_tensor_MAE)
                    
                    probabilities_AIM = F.softmax( output_AIM, dim=1)
                    probabilities_MAE = F.softmax( output_MAE, dim=1)
                    probabilities= probabilities_MAE*probabilities_AIM
                    
                if(Model_Actif == "FSAAM"):
                    probabilities= probabilities_SimCLR*probabilities_MoCo*probabilities_MAE*probabilities_AIM
                _, predicted_class = torch.max(probabilities, dim=1)
                predicted_class_name = class_names[predicted_class.item()]
                
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
image_Path = 'M_Image_Fusion/'+str(nb_Epoch) +"_" + "EPOCH"+"_" + dossier +"_"+ dataset+"_"+Model_Actif+"_"+str(BNF)
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


