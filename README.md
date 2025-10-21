# Identifying pan-cancer and cancer subtype miRNAs using interpretable convolutional neural network (ICNNM)
                                   Authors: Joginder Singh, Shubhra Sankar Ray, and Sukriti Roy
                                 E-Mail: joginder265@gmail.com, shubhra@isical.ac.in, and research.sr22@gmail.com

## Datasets
The following datasets are derived from the original pan-cancer data. They are as follows:  
- <a href = "https://drive.google.com/file/d/1JEuFi3w0DlIRmUft3fjdHh8asn8hIu8J/view">Breast Dataset </a> : This data consists of 4 different subtypes of breast cancer (BASAL, HER2, LUM-A, LUM-B). It contains 792 cancer samples and 25 normal samples.  
- <a href = "https://drive.google.com/file/d/1DMWWqt4dqb4Ixq57QAyMiVdhdI1zJ1Wx/view">Lung Dataset </a> : This data consists of 2 different subtypes of lung cancer (LUAD, LUSC). It consists of 996 cancer samples and 91 normal samples.  
- <a href = "https://drive.google.com/file/d/1CR6wCbfdfqR3Dg7oZDtAy8kJPxD09RG4/view">Kidney Dataset </a> : This data consists of 3 different subtypes of kidney cancer (KICH, KIRC, KIRP). It contains 879 cancer samples and 130 normal samples.  
- <a href = "https://drive.google.com/file/d/1JaNfq2m87z1KtuFZrNWRvS-AbkmXK_49/view">Classified Pan-cancer (CP) Dataset </a> : This classified pan-cancer data consists of 10349 samples derived from 33 different types of cancer.

## Steps to Run ICNNM
1. Open Python and install the packages **numpy**, **math**, **csv**, **pandas**, **sklearn**, **matplotlib**, **time**, **scipy**, **tensorflow**, .  
(Use command `pip install package_name` e.g., `pip install pandas`.  
In higher versions of Python, use `pip3` in place of `pip`.)  
* In Windows environment, if **Spyder** is used for Python, then one has to install the **pip** package first using the command  
  `"python get-pip.py"`  

2. Download the code for **BoMTPE** from: <a href = "https://drive.google.com/file/d/1d7UlNHs6cq7xx-Gf5wCY70f3xR6PwYhv/view?usp=drive_link">`BoMPTE.py` </a>

3. Download the code for **ICNNM** from: <a href = "https://drive.google.com/file/d/1Lj_acADTTKgjee9F45XMHz1I3UeNQruF/view?usp=drive_link">`ICNNM.py` </a>    

4. Keep the code and the datasets in the same folder, otherwise change the folder path along with the name of the dataset in the code (**Line number 25**).  

5. Run `ICNNM.py` to produce `ICNNM_Result.csv` and `ICNNM_performance.csv`.  
   - `ICNNM_Result.csv` contains the **miRNA names**.  

## Cite the Article
Singh, J., Ray, S. S., & Roy, S. (2025). Identifying pan-cancer and cancer subtype miRNAs using interpretable convolutional neural network. Journal of Computational Science, 85, 102510.

@article{singh2025identifying,
  title={Identifying pan-cancer and cancer subtype miRNAs using interpretable convolutional neural network},
  author={Singh, Joginder and Ray, Shubhra Sankar and Roy, Sukriti},
  journal={Journal of Computational Science},
  volume={85},
  pages={102510},
  year={2025},
  publisher={Elsevier}
}



