# Marmalade

<br />Current version: 1.0
<br />Author: Boda Liu
<br />Contact: boda_liu@hotmail.com
<br />Starting date: 2021/1/17

Marmalade (Magma Remake with Machine Learning and Data from Experiments) 岩石学机器学习软件

This software makes predictions about equilibrium phases for a given bulk composition at given conditions. Unlike other thermadynamic softwares which are based on Gibbs energy minimization, Marmalade uses artificial neural network to learn from petrology experiments to automatically predict the composition and proportion of equilibrium phases. Currently, the data from petrology experiments include compilations of LEPR (http://lepr.ofm-research.org/YUI/access_user/login.php filtered range 650-1350C, 0-3GPa) and several new studies. The complete list of references is in a separate file references.txt.

1. About input parameters:
The input bulk composition takes a number of components: 'SiO2','TiO2','Al2O3','FeO','MgO','CaO','Na2O','K2O', 'H2O'. Anhysrous components will be normalized to a sum of 100. FeO is total iron.
Input intensive variables include temperature (celsius), pressure (GPa), and oxygen fugacity (relative to QFM). Please refer to B. R. Frost (1991) in Mineralogical Society of America "Reviews in Mineralogy" Volume 25 to convert fO2 relative to other oxygen buffers to delta QFM.

2. About potential applications:
Currently, the software is suitable for modeling fractionation of mafic magma in the T, P range, 650-1350C and 0-3GPa respectively. The oxygen fugacity would better be within QFM+-2.

3. About output parameters:
Currently, the software can predict the composition of the melt and proportions of several phases including 'Liquid', 'Clinopyroxene', 'Olivine', 'Garnet', 'Plagioclase', 'Amphibole', 'Orthopyroxene'.

An example of modeling fractionation of hydrous magmas.

![gold_fig](https://user-images.githubusercontent.com/30700234/113654301-2150cb00-965d-11eb-9e2c-98c5cdcb20bf.png)
