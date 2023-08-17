# Bonding-aware Materials Representation for Deep Learning Atomistic Models
This repository contains code related to the scientific article titled "Bonding-aware Materials Representation for Deep Learning Atomistic Models" authored by Or Shafir and Ilya Grinberg from the Department of Chemistry at Bar-Ilan University, Ramat-Gan, Israel.

## Abstract
Deep potentials for molecular dynamics (MD) achieve first-principles accuracy at significantly lower computational cost. However, their application in large length- and time-scale simulations is limited by their reduced speed compared to analytical atomistic potentials. This limitation arises due to the complexity of the neural network architecture and the long time required for embedding calculations. In this study, we propose a chemical-bonding-aware embedding for neural network potentials that achieve state-of-the-art accuracy in predicting forces and local electronic density of states (LDOS). Our method utilizes a compact 16x32 neural network, significantly reducing computational costs while maintaining high accuracy.

## Repository Structure
README.md: This file, providing an overview of the repository and its contents.
Files which are not mentioned in the follwing "embedding" and "example" catagories are obselete.

### Embedding

Run the generate_embedding.py script to generate the chemical-bonding-aware embedding for neural network potentials.
Use the generated data for your own deep learning atomistic models and simulations.
Please refer to the original article for more details on the methods and algorithms used in this work.

1. generate_embedding.py: Python script to generate the chemical-bonding-aware embedding for neural network potentials.<br>
   - Input: data-name.parquet: The dataset extracted from calculations for the material ("name").<br>
   - Output: X-name-paths.parquet: Preprocessed and transformed data for the material ("name") containing paths and neighbors information.<br>

### Forces Example MoO3 / BaTiO3
#### Training-related Files
1. train_loop-0.py: training loop which loads the features, target, filter the embedding based on wanted levels (number of steps and neighbors)
   - Input: X-name-paths.parquet: Preprocessed and transformed data for the material ("name") containing paths and neighbors information.<br>
   - Output: model which is saved in the results subdirectory.
2. run-0.sh: A bash script which reuns train_loop-0.py
3. learning_curve_analysis2.ipynb:  Jupyter notebook that allows examining the learning curve for the different models based on different embedding level of details (number of steps and neighbors).<br>

#### Analysis-related Files
1. benchmark_analysis.ipynb: Jupyter notebook that compares the valdiation error for different embedding level of details (number of steps and neighbors). <br>
2. prediction_error_analysis.ipynb:  Jupyter notebook that calculate the prediction of all of the models based on different embedding level of details (number of steps and neighbors) and compares the errors <br>


Usage
To use the provided code, follow these steps:



Dependencies
The code in this repository requires the following dependencies:

Python (version X.X.X)
NumPy (version X.X.X)
Pandas (version X.X.X)
SciPy (version X.X.X)
Citation
If you use this code or data in your work, please consider citing the original article:

Or Shafir and Ilya Grinberg. "Bonding-aware Materials Representation for Deep Learning Atomistic Models." Journal Name, Volume(Issue), Year, Pages. DOI: XXXX/XXXXX
Contact
For any questions or inquiries, please contact the authors of the original article:

Or Shafir: or.shafir@gmail.com
Ilya Grinberg: ilya.grinberg@biu.ac.il
