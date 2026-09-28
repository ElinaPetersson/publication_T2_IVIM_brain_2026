# Code for: T2 relaxation effects on intravoxel incoherent motion parameter estimates in brain tumors and healthy brain

Elina Petersson(1), Maria Ljungberg(1,2), Maja Sohlin(1,2), Mats Laesser(3,4), Isabella M Björkman-Burtscher(3,4), Oscar Jalnefjord(1,2) 
1 Department of Medical Radiation Sciences, Institute of Clinical Sciences, Sahlgrenska Academy, University of Gothenburg, Gothenburg, Sweden 
2 Department of Biomedical Engineering and Medical Physics, Sahlgrenska University Hospital, Region Västra Götaland, Gothenburg, Sweden 
3 Department of Radiology, Institute of Clinical Sciences, Sahlgrenska Academy, University of Gothenburg, Gothenburg, Sweden 
4 Department of Radiology, Sahlgrenska University Hospital, Region Västra Götaland, Gothenburg, Sweden 



## Description
Code required to reproduce the optimization of b-values and echo times in the paper. Also code for bayesian fitting of the parameters.

## Requirements
numpy
scipy
ivim (https://github.com/oscarjalnefjord/ivim)
...

## Reproducing the analysis
1. Run the file optimization.py
2. Acquire diffusion MRI data with the optimized protocol
3. Run fitting with bayes in fit.py