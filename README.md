# Marmalade

Current version: 1.0  
Author: Boda Liu  
Contact: boda_liu@hotmail.com  
Starting date: 2021/1/17  

Marmalade (Magma Remake with Machine Learning and Data from Experiments)
A machine learning model for modeling magma differentiation

This software makes predictions about equilibrium phases for a given bulk composition at given conditions. Unlike other thermadynamic softwares which are based on Gibbs energy minimization, Marmalade uses artificial neural network to learn from petrology experiments to automatically predict the composition and proportion of equilibrium phases. Currently, the data from petrology experiments include compilations of LEPR (http://lepr.ofm-research.org/YUI/access_user/login.php filtered range 650-1350C, 0-3GPa) and several new studies. The complete list of references is in a separate file references.txt.

1. About input parameters:
The input bulk composition takes a number of components: 'SiO2','TiO2','Al2O3','FeO','MgO','CaO','Na2O','K2O', 'H2O'. Anhysrous components will be normalized to a sum of 100. FeO is total iron.
Input intensive variables include temperature (celsius), pressure (GPa), and oxygen fugacity (relative to QFM). Please refer to B. R. Frost (1991) in Mineralogical Society of America "Reviews in Mineralogy" Volume 25 to convert fO2 relative to other oxygen buffers to delta QFM.

2. About potential applications:
Currently, the software is suitable for modeling fractionation of mafic magma in the T, P range, 650-1350C and 0-3GPa respectively. The oxygen fugacity would better be within QFM+-2.

3. About output parameters:
Currently, the software can predict the composition of the melt and proportions of several phases including 'Liquid', 'Clinopyroxene', 'Olivine', 'Garnet', 'Plagioclase', 'Amphibole', 'Orthopyroxene'.

An example of modeling fractionation of hydrous magmas. This work is presented as an abstract at Goldschmidt 2021 (Abstract #6433).

We use the machine learning model to predict the composition of the melt and proportion of phases. Predictions of the curret model are compared to actual data and predictions made by MELTS. The machine learning model did a reasonably good job at reproducing Al2O3 of the melt, proportions of melt and amphibole. For applications to the fractionation of hydrous arc magma, amphibole is a common phase. Natural observations also see amphibole as a major phase in crustal cumulates. Therefore, the machine learning model could be suitable for modeling the fractionation of arc magma.

![gold_fig](https://user-images.githubusercontent.com/30700234/113654301-2150cb00-965d-11eb-9e2c-98c5cdcb20bf.png)
