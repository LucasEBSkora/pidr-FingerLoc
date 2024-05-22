# 🇫🇷 Méthode de Localisation Indoor Avancée par Fingerprinting BLE pour Architecture IoT (FingerLoc)

Cette repositoire contient le résultat d'un Projet d'Introduction à la Recherche concernant l'implementation d'un algorithme de localisation des objets connectés en interieur par fingerprinting.

Pour faire marcher le programme principal qui est dans le fichier `src/locationFinder.py`, il faut avoir un fichier dans le dossier `src` du repositoire appelé `mqtt_credentials.py` avec le format suivant:

```python
server = "server.com"
port = 1883
subscribe_topic = "topic/name"
username = "beep_boop"
password = "12345"
```
Qui n'est pas public par des raison évidents. Le vraie algorithme de Fingerprinting est dans le fichier `src/location_algorithm.py`, ou il peut être changé si necessaire.

pour génerer plus de fingerprints, le programme `src/acquire_fingerprints.py` peut être utilisé. Il prend comme parametre le nom du fichier où les nouvelles fingerprints doivent être stockés. Ces nouvelles donnés doivent être traités avec le programme `formatData/format_data.py` pour pouvoir être utilisé par les autres programmes dans ce repositoire. Si une taille de grille plus grand est desirée, le programme `formatData/change_grid_sizes.py` peut être utilisé. Les données utilisés pour ces experiments sont stockés dans le dossier `data`.

Les autres dossiers dans ce repositoire contiennent les programmes utilisés pour les testes et experiments et le résultats de ces experiments.

# 🇺🇸 Advanced Indoor Localisation Method by BLE Fingerprinting for IoT Architectures (FingerLoc)

This repository contains the result of a Introduction to Research Project concerning the implementation of a connected object indoor localisation algorithm using fingerprinting.

In order to make the main program, which is located in the `src/locationFinder.py` file work, you must have a file in the `src` folder called `mqtt_credentials.py` with the following format:

```python
server = "server.com"
port = 1883
subscribe_topic = "topic/name"
username = "beep_boop"
password = "12345"
```
Which isn't public for self-evident reasons. The actual Fingerprinting algorithm is in the `src/location_algorithm.py` script, where it may be changed if necessary.

In order to generate more fingerprints, the `src/acquire_fingerprints.py` program can be utilised. It takes one parameter, which is the name of the file in which to stock the new fingerprints. This new data must be treated by the `formatData/format_data.py` program so that the other programs in this repository are able to use them. If a larger grid size is desired, the `formatData/change_grid_sizes.py` program may be utilised. The data that was used in these experiments is stocked in the `data` folder.

The other folders in this repository contain the scripts used in the tests and experiments and the results of those tests and experiments.