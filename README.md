
## Partie 1 : ResNet20 pour la Classification 

Ce dépôt contient une implémentation de **ResNet** pour la classification d'images sur le dataset **CIFAR-10**. **ResNet20** est une version simplifiée de l'architecture **ResNet** (Residual Network), utilisée pour résoudre les problèmes de dégradation des performances dans les réseaux de neurones profonds. L'implémentation est faite en utilisant **PyTorch**.

## Architecture

L'architecture **ResNet20** est composée de 20 couches et utilise des **blocs résiduels** pour faciliter l'entraînement et améliorer la performance des réseaux profonds.

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

---

### Architecture détaillée

| Étape                  | Type de couche            | Sortie (dimension)            |
|------------------------|---------------------------|-------------------------------|
| **Entrée**             | Image 32x32 (RGB)         | (3, 32, 32)                   |
| **Première couche**    | Convolution 3x3, 16 filtres| (16, 32, 32)                  |
| **Layer 1**            | 3 blocs résiduels (stride 1)| (16, 32, 32)                  |
| **Layer 2**            | 3 blocs résiduels (stride 2)| (32, 16, 16)                  |
| **Layer 3**            | 3 blocs résiduels (stride 2)| (64, 8, 8)                    |
| **Global Average Pooling** | Moyenne de chaque carte de caractéristiques | (64, 1, 1) |
| **Fully Connected**    | Couche linéaire (10 classes) | (10)                           |

---

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




Les modèles entraînés ont ensuite été évalués sur CIFAR-10, et leurs prédictions ont été visualisées.


### Installation

Clonez le dépôt :

```bash
git clone https://github.com/SihamBeldjoudi/ResNet.git
cd ResNet20-CIFAR10
```

Installez les dépendances :

```bash
pip install -r requirements.txt
```

---

## License

Ce projet est sous la licence Apache. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

