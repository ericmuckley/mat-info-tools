# Materials informatics tools

General tools for materials informatics workflows.



## Featurization of chemical formulas

Citrine's [chemical featurization documentation](https://help.citrination.com/knowledgebase/articles/1863853-citrine-public-feature-descriptions), describes 134 chemical features extracted from [magpie](https://bitbucket.org/wolverton/magpie/src/master/) and the [chemistry development kit](https://cdk.github.io/). Features are divided into two sets:
1. The Standard Set - 63 features (default features generated by the Citrine platform)
    * Elemental Properties - 24 features
    * Molecule Features - 7 features
    * Analytic Features - 32 features
1. The Extended Set - 71 features (features which are generally more expensive to compute and of less value to ML models)
    * Elemental Properties – 30 features
    * Molecule Features – 41 features

These features, and many more, can be created using [matminer](https://hackingmaterials.lbl.gov/matminer/index.html), which has integrated featurization methods from [pymatgen](https://pymatgen.org/) and [magpie](https://bitbucket.org/wolverton/magpie/src/master/). To install matminer, use `pip install matminer`.

Useful links
* [Matminer summary table of features](https://hackingmaterials.lbl.gov/matminer/featurizer_summary.html#)
* [Matminer Github repo](https://github.com/hackingmaterials/matminer)
* [Matminer notebook examples](https://github.com/hackingmaterials/matminer_examples)
* [Matminer examples repo](https://nbviewer.jupyter.org/github/hackingmaterials/matminer_examples/blob/master/matminer_examples/index.ipynb)
