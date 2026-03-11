# ScenarIA : Forcings-conditioned climate emulator for exploring novl emission pathways

## Context
Greenhouse gas emission scenarios uncertainty is a significant barrier to climate risk assessment, requiring expensive simulations to explore potential forced responses and the associated range of plausible outcomes. Statistical emulators offer a faster alternative to estimate climate responses for new emission pathways but often struggle to generalize or capture changes in high-variance variables like precipitation. We present ScenarIA, a generalizable, deep learning-based emulator of spatial forced responses and associated variability. By training on multi-scenario large ensemble climate simulations, ScenarIA aims to capture the complete distribution of future climate trajectories for a wide range of emission pathways.

---

## Code structure
```
├── configs
│   ├── config.yaml
│   ├── plots.yaml
│   └── runs.yaml
├── README.md
├── src
│   ├── data
│   │   ├── dataloader.py
│   │   ├── download_data.py
│   │   ├── lightning_module.py
│   │   ├── preprocessing.py
│   ├── evaluation
│   │   └── compare.py
│   ├── models
│   │   ├── alternative_convlstm.py
│   │   ├── CNNLSTM.py
│   ├── plots
│   │   ├── explore_test_dataset.ipynb
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
└── tests
    ├── test_data.py
    ├── test.ipynb
    └── test.py
```

---

## Usefull commands

### Dataset
Les données d'entrée sont stockées sous forme de fichiers NETCDF4 dans le répertoire `dataset`. Il y a un fichier par scénario ou expériences et chaque simulation doit provenir du même modèle physique.
Les entrées `inputs_{simulation}.nc` représentent les forçages CO2, CH4, SO2 et BC. Les variables ont pour dimensions (time, lat, lon) (à l'exception des gaz à effet de serre CO2 et CH4 qui n'ont pas de variation spatiale).
Les sorties `outputs_{simulation}.nc` représentent les variables physiques taset pr. Les variables ont pour dimensions (time, lat, lon, member).

Un fichier `coords.npz` doit également se trouver dans le dossier `dataset`. Il donne l'information de la grille de sortie cible.

La class `Outputs` permet de traiter les NETCDF4 pour les enregistrer au bon format.

```
python src/data/preprocessing.py
```

### Training
```
python src/train.py
```

### Evaluation
```
python src/evaluation/compare.py
```

### Prediction
```
python src/predict.py
```