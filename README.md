# Bonding-aware Materials Representation for Deep Learning Atomistic Models
This repository contains code related to the scientific article titled "Bonding-aware Materials Representation for Deep Learning Atomistic Models" authored by Or Shafir and Ilya Grinberg from the Department of Chemistry at Bar-Ilan University, Ramat-Gan, Israel.

## Abstract
Deep potentials for molecular dynamics (MD) achieve first-principles accuracy at significantly lower computational cost. However, their application in large length- and time-scale simulations is limited by their reduced speed compared to analytical atomistic potentials. This limitation arises due to the complexity of the neural network architecture and the long time required for embedding calculations. In this study, we propose a chemical-bonding-aware embedding for neural network potentials that achieve state-of-the-art accuracy in predicting forces and local electronic density of states (LDOS). Our method utilizes a compact 16x32 neural network, significantly reducing computational costs while maintaining high accuracy.

## Conclusion
In conclusion, we have developed a novel chemical-bonding-aware embedding based on the moments theorem for use in atomistic neural network models for molecular dynamics simulations. We have demonstrated the effectiveness of this embedding in accurately predicting forces and LDOS for MoO3 and BaTiO3 using an ultrasmall 16x32 neural network architecture. The key advantage of our approach is the highly descriptive and comprehensive bonding representation provided by the embedding, which is directly relevant to the prediction targets. This eliminates the need for the neural network to redundantly learn information already well-known to physicists and chemists, resulting in a smoother learning trajectory and overall learning process. Additionally, the low propagation time step time and simplicity of the embedding enable short time steps, making it highly practical for MD simulations.

## Repository Structure
data-MoO3.parquet: The dataset extracted from calculations for MoO3.
X-BaTiO3-paths.parquet: Preprocessed and transformed data for BaTiO3 containing paths and neighbors information.
generate_embedding.py: Python script to generate the chemical-bonding-aware embedding for neural network potentials.
README.md: This file, providing an overview of the repository and its contents.
Usage
To use the provided code, follow these steps:

Clone or download the repository.
Make sure you have the required dependencies installed.
Run the generate_embedding.py script to generate the chemical-bonding-aware embedding for neural network potentials.
Use the generated data for your own deep learning atomistic models and simulations.
Please refer to the original article for more details on the methods and algorithms used in this work.

Dependencies
The code in this repository requires the following dependencies:

Python (version X.X.X)
NumPy (version X.X.X)
Pandas (version X.X.X)
SciPy (version X.X.X)
Citation
If you use this code or data in your work, please consider citing the original article:

mathematica
Copy code
Or Shafir and Ilya Grinberg. "Bonding-aware Materials Representation for Deep Learning Atomistic Models." Journal Name, Volume(Issue), Year, Pages. DOI: XXXX/XXXXX
Contact
For any questions or inquiries, please contact the authors of the original article:

Or Shafir: email@example.com
Ilya Grinberg: email@example.com
