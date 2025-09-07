# Identifying pan-cancer and cancer subtype miRNAs using interpretable convolutional neural network (ICNNM)
## Abstract
Background: MiRNAs are short-length (∼22nt) non-coding RNAs and are considered to be important biomarkers
in pan-cancer analysis. Pan-cancer analysis is the study of finding the commonalities and differences in
genetic and cellular alterations in various types of cancers. A common computational challenge in handling
miRNA expression data is that it is high dimensional and complex (HDC) in nature. In this regard, convolutional
neural networks are proven to be good performers due to their nature of finding patterns in complex data.
Methodology: An interpretable convolutional neural network model (ICNNM) is developed for classifying
miRNA expression based pan-cancer data. The ICNNM is a one dimensional model. The layers and other
hyperparameters are optimized using Bayesian optimization with multivariate tree parzen estimator (BoMTPE).
An interpretable approach is developed using SHapley Additive exPlanations (SHAP) values for explaining the
behavior of ICNNM. This approach helps in introducing an attribution score for identifying relevant miRNAs
using SHAP values. The attribution scores are assigned higher values for those miRNAs which help in the
accurate prediction of tumor class of patients by utilizing the game theory concept in computing the SHAP
values. The model is evaluated on 9 datasets among which 6 datasets (4 general pan cancer and two subtypes)
are derived from a single TCGA pan-cancer dataset, one dataset is downloaded as Breast sub-type from TCGA,
and two datasets, nasopharyngeal carcinoma and bone and soft tissue sarcoma, are downloaded from GEO as
rare cancer ones.
Results: The ICNNM is seen to perform better as compared to related techniques such as three variations of
the CNN model, random forest RF, SVM, Gboost, XGboost, and Catboost. The performance is evaluated in
terms of F1-score, discriminability power of expressions between normal and tumor classes, and biological
significance of the selected miRNAs. The biological significance is established through existing literatures and
online databases such as gene ontology and KEGG pathways after obtaining the target genes using miRDB
database. While the performance of ICNNM in terms of F1-score varies from 0.95 to 0.99 for 4 general pancancer
datasets, it varies from 0.91 to 0.99 for 3 subtype datasets and from 0.76 to 0.90 for rare cancer datasets.
Many of the selected miRNAs are found to be the key biomarkers in various tumor classes according to existing
investigations. Three miRNAs miR-503, miR-202, and miR-135a can be considered as novel predictions for
cancer classes prostate and rectum, mesothelioma, and testicular germ cells, respectively, as their target genes
are involved in related cancer pathways, obtained using miRDB database.


