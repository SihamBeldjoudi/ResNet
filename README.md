# RegNet  (Regulated ResNet)
## Contributeurs
- Siham Beldjoudi
- Kevin Lieske
## Partie 1 : ResNet20 pour la Classification 

Ce dépôt contient Notre implémentation de **ResNet** pour la classification d'images sur le dataset **CIFAR-10**. Notre version **ResNet20** est une version simplifiée de l'architecture **ResNet** (Residual Network), utilisée pour résoudre les problèmes de dégradation des performances dans les réseaux de neurones profonds. 
L'implémentation est faite en utilisant **PyTorch**.

## Architecture

L'architecture **ResNet20** utilise des **blocs résiduels** pour faciliter l'entraînement et améliorer la performance des réseaux profonds.

### Vue d'ensemble de l'architecture :

1. **Première couche de Convolution**  
   - Convolution 3x3 avec 16 filtres, suivi d'une **Batch Normalization** et d'une activation **ReLU**.  
   - Dimensions de sortie : **(16, 32, 32)**

2. **Trois couches résiduelles (Residual Blocks)**  
   Chaque couche résiduelle est composée de 2 convolutions 3x3 et utilise des **shortcuts** (ou raccourcis) pour éviter la dégradation des performances. Ces couches sont organisées en trois groupes :
   
   - **Layer 1** : 3 blocs résiduels avec **stride = 1**  
   - **Layer 2** : 3 blocs résiduels avec **stride = 2** (réduit la taille des images)  
   - **Layer 3** : 3 blocs résiduels avec **stride = 2** (réduction supplémentaire de la taille des images)

3. **Global Average Pooling**  
   - Cette étape réduit chaque carte de caractéristiques à une seule valeur, passant de dimensions **(64, 8, 8)** à **(64, 1, 1)**.

4. **Couche entièrement connectée (Fully Connected Layer)**  
   - Cette couche prend le vecteur aplati de **64** valeurs et le transforme en un vecteur de **10** valeurs correspondant aux 10 classes du dataset CIFAR-10.

### Détail des Blocs Résiduels

Chaque **bloc résiduel** est constitué de deux **convolutions 3x3** successives, suivies de **Batch Normalization** et d'activations **ReLU**. Les **shortcuts** (ou connexions résiduelles) permettent à l'entrée du bloc d'être ajoutée directement à la sortie, facilitant ainsi l'apprentissage.

- Si les dimensions d'entrée et de sortie sont identiques, l'entrée est ajoutée à la sortie du bloc.
- Si les dimensions diffèrent (par exemple, en cas de réduction de taille), un **raccourci** avec une **convolution 1x1** est utilisé pour ajuster les dimensions.

## Code du Modèle

Voici l'implémentation du modèle **ResNet20** en PyTorch. Il utilise des blocs résiduels (appelés **BasicBlock**) pour permettre l'apprentissage des réseaux profonds.

```python
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
```

---

## Paramètres d'Entraînement

Voici les paramètres utilisés pour entraîner le modèle :

- **Dataset** : CIFAR-10 (10 classes)
- **Époques** : 150
- **Taille du batch** : 64
- **Optimiseur** : **SGD** avec `momentum=0.9` et `weight_decay=1e-4`
- **Taux d'apprentissage** : 0.1 (réduit à 0.01 après 80 époques)
- **Fonction de perte** : **CrossEntropyLoss**

### Code pour l'ajustement du taux d'apprentissage :

```python
def adjust_learning_rate(optimizer, epoch):
    if epoch >= 80:
        lr = 0.01
    else:
        lr = 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

---

## Entraînement et Évaluation

Le modèle est entraîné sur le dataset **CIFAR-10** pendant 150 époques. Le taux d'apprentissage est ajusté après la 80ème époque pour améliorer la convergence. Une fois l'entraînement terminé, les poids du modèle sont sauvegardés dans un fichier `.pth`.

Le script évalue également le modèle sur le jeu de validation après chaque époque, en affichant la perte et la précision.

---

## Visualisation des Résultats

Après l'entraînement, des images sont sélectionnées aléatoirement à partir du jeu de test pour visualiser les prédictions du modèle. La couleur du titre de chaque image montre si la prédiction est correcte (vert) ou incorrecte (rouge).

---

## Partie 2 : Implémentation du Modèle Basé sur RegNet

### Contexte
L'approche s'inspire de l'article *"RegNet: Self-Regulated Network for Image Classification"* de Jing Xu et al., qui propose l'utilisation d'un module récurrent auto-régulé pour affiner l'extraction des caractéristiques. Nous avons incorporé cette idée en intégrant des mécanismes récurrents aux blocs d'un ResNet.

### Méthodologie
Une couche récurrente convolutive `ConvRNN` a été définie et intégrée dans un bloc de ResNet (`BasicRNNBlock`). Une variante plus avancée, `ConvGRU`, a également été testée.

**Structures mises en place :**

- **ConvRNN** : Applique une récurrence sur des cartes de caractéristiques convolutives.
- **ConvGRU** : Variante avancée du ConvRNN, intégrant des portes pour mieux gérer l'information.
- **BasicRNNBlock** : Bloc de ResNet modifié avec une couche récurrente (`ConvRNN` ou `ConvGRU`).
- **ResNetRNN** : Architecture ResNet intégrant ces blocs récurrents.

### Variantes explorées

#### ResNet + ConvRNN

Le modèle **ResNet + ConvRNN** intègre une couche récurrente convolutive simple, permettant de capturer les dépendances spatiales dans l'image. 

Voici l'implémentation de `ConvRNN` en PyTorch :

```python
class ConvRNN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(ConvRNN, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv_x = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size=3, padding=1)

    def forward(self, x, h):
        if h is None or h.shape[1] != self.hidden_channels:
            h = torch.zeros(x.shape[0], self.hidden_channels, x.shape[2], x.shape[3], device=x.device)

        x = torch.cat([x, h], dim=1)
        x = self.conv_x(x)
        return torch.tanh(x)
```

#### ResNet + ConvGRU

Le modèle **ResNet + ConvGRU** améliore le ConvRNN en ajoutant des **portes de mise à jour et de réinitialisation**, permettant une meilleure gestion des informations passées.  
Le ConvGRU est implémenté comme décrit dans Ballas et al. 2015 : *Delving Deeper into Convolutional Networks for Learning Video Representations*.

Voici l'implémentation de `ConvGRU` en PyTorch :

```python
class ConvGRU(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(ConvGRU, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv_x = nn.Conv2d(in_channels + hidden_channels, 2 * hidden_channels, kernel_size=3, padding=1)
        self.conv_y = nn.Conv2d(in_channels + 2*hidden_channels, hidden_channels, kernel_size=3, padding=1)

    def forward(self, x, h):
        if h is None or h.shape[1] != self.hidden_channels:
            h = torch.zeros(x.shape[0], self.hidden_channels, x.shape[2], x.shape[3], device=x.device)

        x = torch.cat([x, h], dim=1)
        gates = self.conv_x(x)
        a, b = torch.split(gates, self.hidden_channels, dim=1)
        a, b = torch.sigmoid(a), torch.sigmoid(b)

        y = torch.cat([x, b * h], dim=1)
        y = torch.tanh(self.conv_y(y))
        h = a * h + (1 - a) * y

        return h
```

### Entraînement et évaluation
L'entraînement a été réalisé via la fonction `train_model`, en testant :

1. **ConvRNN** comme module récurrent de base.
2. **ConvGRU**, une alternative plus avancée et performante.

Les modèles entraînés ont ensuite été évalués sur CIFAR-10 et nos donnees personnelles pour la classification de viande (avec 4 classes), et leurs prédictions ont été visualisées.

## Partie 3:  RegNet-400MF sur CIFAR-10 et Données Personnelles

Cette troisieme partie met en œuvre le réseau neuronal **RegNet-400MF** pour la classification d'images sur le dataset **CIFAR-10** et sur une base de données personnelle contenant **4 classes de pièces de viande**. L'objectif est de comparer les performances de RegNet-400MF avec nos propres modèles **ResNet-20 + ConvRNN et ConvGRU**, en nous inspirant de l'architecture de l'article *RegNet: Self-Regulated Network for Image Classification*.

## Modèle Utilisé

- **RegNet-Y 400MF**, avec une dernière couche ajustée pour **10 classes** (CIFAR-10) et **4 classes** (notre dataset)

## Prétraitement des Données

Le prétraitement des données est essentiel pour garantir de bonnes performances. Voici les principales étapes :

```python
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)), # Taille adaptée à RegNet
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

Nous chargeons soit les jeux de données CIFAR-10 soit notre propre base de données de 4 classes :

```python
import torchvision

batch_size = 64
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
```

## Entraînement du Modèle

Nous utilisons **SGD** comme optimiseur et **CrossEntropyLoss** comme critère de perte.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import regnet_y_400mf

model = regnet_y_400mf(pretrained=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modification de la dernière couche pour 10 classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10) # 10 classes pour CIFAR-10
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
```

Ajustement du taux d'apprentissage :

```python
def adjust_learning_rate(optimizer, epoch):
    lr = 0.01 if epoch >= 8 else 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

Boucle d'entraînement :

```python
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    adjust_learning_rate(optimizer, epoch)
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Accuracy: {accuracy:.2f}%, Loss: {running_loss/len(trainloader):.4f}")
```

## Évaluation et Comparaison

Nous testons nos modèles sur **CIFAR-10** et notre dataset personnel de **4 classes** pour comparer leurs performances.

```python
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')
```

## Visualisation des Prédictions

Affichage de quelques images de test avec leurs prédictions :

```python
import matplotlib.pyplot as plt
import numpy as np
import random

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

random_indices = random.sample(range(len(testset)), 16)
random_images = torch.stack([testset[i][0] for i in random_indices]).to(device)
random_labels = torch.tensor([testset[i][1] for i in random_indices]).to(device)

outputs = model(random_images)
_, predicted = torch.max(outputs, 1)

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    title = f"Pred: {classes[predicted[i]]}\nVrai: {classes[random_labels[i]]}"
    ax.imshow(np.transpose((random_images[i].cpu() / 2 + 0.5).numpy(), (1, 2, 0)))
    ax.set_title(title, fontsize=10, color='green' if predicted[i] == random_labels[i] else 'red')
    ax.axis('off')
plt.show()
```

## Améliorations Possibles

- Expérimenter avec **d'autres datasets comme ImageNet qui a ete utilisee dans l'article**. Cela necessite le materiel adecquat pour ce genre d'application.
- Tester des **augmentations de données** pour améliorer la robustesse

---

Projet développé dans le cadre d'une expérimentation sur **RegNet et ResNet**, avec l'intégration de **ConvGRU et ConvRNN** pour la classification d'images.




### Installation

Clonez le dépôt :

```bash
git clone https://github.com/SihamBeldjoudi/ResNet-RNN_RegNet.git
```

Installez les dépendances :

```bash
pip install -r requirements.txt
```

---

## License

Ce projet est sous la licence Apache. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

