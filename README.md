# ScenarIA : Forcings-conditioned climate emulator for exploring novl emission pathways

## Context
Greenhouse gas emission scenarios uncertainty is a significant barrier to climate risk assessment, requiring expensive simulations to explore potential forced responses and the associated range of plausible outcomes. Statistical emulators offer a faster alternative to estimate climate responses for new emission pathways but often struggle to generalize or capture changes in high-variance variables like precipitation. We present ScenarIA, a generalizable, deep learning-based emulator of spatial forced responses and associated variability. By training on multi-scenario large ensemble climate simulations, ScenarIA aims to capture the complete distribution of future climate trajectories for a wide range of emission pathways.

---

## Code structure
```
.
├── configs
│   ├── config.yaml
│   ├── plots.yaml
│   ├── runs.yaml
│   └── sensitivity.yaml
├── README.md
├── src
│   ├── data
│   │   ├── dataloader.py
│   │   ├── download_data.py
│   │   ├── lightning_module.py
│   │   ├── preprocessing.ipynb
│   │   ├── preprocessing.py
│   ├── evaluation
│   │   ├── compare.py
│   ├── main.py
│   ├── models
│   │   ├── alternative_convlstm.py
│   │   ├── CNNLSTM.py
│   │   ├── miniunet.py
│   │   ├── time_unet.py
│   │   └── unet.py
│   ├── plots
│   │   ├── explore_test_dataset.ipynb
│   │   ├── plots_predictions.ipynb
│   │   └── plots_rawdata.ipynb
│   ├── predict.py
│   ├── train.py
│   └── utils
│       ├── datautils.py
│       ├── evalutils.py
│       ├── losses.py
│       ├── metrics.py
│       ├── plotutils.py
│       ├── settings.py
│       ├── transforms.py
│       └── utils.py
```

---

## Usefull commands

### Dataset
Les données d'entrée sont stockées sous forme de fichiers NETCDF4 dans le répertoire `dataset`. Il y a un fichier par scénario ou expériences et chaque simulation doit provenir du même modèle physique.
Les entrées `inputs_{simulation}.nc` représentent les forçages CO2, CH4, SO2 et BC. Les variables ont pour dimensions (time, lat, lon) (à l'exception des gaz à effet de serre CO2 et CH4 qui n'ont pas de variation spatiale).
Les sorties `outputs_{simulation}.nc` représentent les variables physiques tas et pr. Les variables ont pour dimensions (time, lat, lon, member).

Un fichier `coords.npz` doit également se trouver dans le dossier `dataset`. Il donne l'information de la grille de sortie cible.

Les classes `Inputs` et `Outputs` permet de traiter les NETCDF4 pour les enregistrer au bon format.

```
python src/data/preprocessing.py
```

### Training
Le fichier config.yaml permet de configurer les paramètres de l'expérience et les hyperparamètres d'entraînement.
Une nouvelle expérience est à créer à chaque nouveau trio de données train/val/test. Le fichier `experiment_infos.yaml` répertorie l'ensemble des expériences.

Dans le cas d'un entraînement unique:
```
python src/train.py
```

Dans le cas d'une série d'entraînements:
```
python src/main.py
```

### Prediction
Le scipt `predict.py` récupère le chemin du modèle entrainé choisi dans le fichier `runs.yaml` pour prédire une nouvelle trajectoire qui sera sauvegardé dans un fichier NETCDF.

Afin d'étudier la robustesse de notre méthode, on entraîne 10 réseaux différents avec différents mélanges de données d'entrainement. Le script `predict.py` permet d'ajouter une dimension 'runs' au fichier NETCDF qui correspond aux différentes seeds.

```
python src/predict.py
```

### Evaluation
```
python src/evaluation/compare.py
```
