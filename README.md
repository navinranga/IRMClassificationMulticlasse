# Réseau de Neurones pour la Classification d'Images IRM (Projet CLANU)

## Objectif

Ce projet a été réalisé dans le cadre du module d’analyse numérique CLANU. Il a pour objectif de mettre en œuvre un **réseau de neurones généralisé pour la classification multiclasse** appliqué à des **images d’IRM cérébrales**.

Le but est de reconnaître automatiquement, pour chaque image :

- **L’orientation anatomique** : axial, coronal, sagittal
- **Le type de séquence IRM** : T1, T2, ou PD (Proton Density)

Ce qui représente **9 classes distinctes** (3 orientations × 3 types de pondération).

---

## 📥 Cloner le dépôt

Pour récupérer le projet en local, utilisez la commande suivante :


    git clone https://github.com/navinranga/IRMClassificationMulticlasse.git
    cd IRMClassificationMulticlasse

---

## Contenu du projet

- `data/` et `+database/`: Données  d'IRM cérébrales et constitution des bases d'entraintement et de test
- `+L_layers_nn/` : Contient les fichiers constituant le modèle et la prédiction : `model.m` et `predict.m`
- `+visu/` : Fonctions utilitaires pours la visualisation
- `clanu22_23.pdf` : Enoncé du projet
- `scrip_1_classification_R2.m` et `script_2_nn_classification.m` : scripts de classification
- `README.md` : Ce fichier

---

## Technologies utilisées

- **MATLAB** (version recommandée : R2022a ou supérieure)

---

## Démarche

1. **Prétraitement** des images IRM :
   - Conversion en niveaux de gris
   - Redimensionnement à une taille fixe
   - Étiquetage selon les métadonnées (orientation + type)

2. **Conception d’un réseau de neurones** :
   - Architecture feedforward
   - Fonction de coût : entropie croisée
   - Stratégies contre le surapprentissage : early stopping, validation croisée, régularisation

3. **Évaluation des performances** :
   - Précision globale

---

## Résultats obtenus

- Bonne séparation des 9 classes
- Résistance au surapprentissage

---

## Contexte mathématique

La classification est basée sur une généralisation des réseaux de neurones vus en MA1 :
- Classification multiclasse à l’aide de **sorties codées one-hot**
- Optimisation par descente de gradient avec backpropagation
- Utilisation de **fonctions d’activation non-linéaires** (sigmoïde, softmax)

---

## Auteurs

- Navin RANGA (Projet réalisé dans le cadre du module CLANU (Analyse Numérique) à l'**INSA Lyon** – Département de Genie Electrique)
- Encadré par Pr Elie Bretin

