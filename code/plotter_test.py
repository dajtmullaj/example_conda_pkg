"""
Test file for the Plotter class and its methods.
"""

from chemplot import Plotter
import pandas as pd 
import sys
import io
from rdkit import Chem
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.manifold import TSNE
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Retrieve data
#molecules_data_solubility_dataset_E = pd.read_csv(".\Data\\solubility-dataset-E.csv") 
#molecules_data_aqsoldb = pd.read_csv(".\Data\\aqsoldb.csv") 
#data_LOGS = pd.read_csv(".\Data\\test_data\\R_1291_LOGS.csv") 
#data_LOGP = pd.read_csv(".\Data\\test_data\\R_4200_LOGP.csv")
#data_BACE = pd.read_csv(".\Data\\test_data\\R_1513_BACE.csv")
#data_SAMPL = pd.read_csv(".\Data\\test_data\\R_642_SAMPL.csv") 
#data_AQSOLDB = pd.read_csv(".\Data\\test_data\\R_9982_AQSOLDB.csv") 
#data_BBBP = pd.read_csv(".\Data\\test_data\\C_2039_BBBP_2.csv") 
#data_HIV_3 = pd.read_csv(".\Data\\test_data\\C_41127_HIV_3.csv") 
#data_HIV_2 = pd.read_csv(".\Data\\test_data\\C_41127_HIV_2.csv")
#data_BACE_2 = pd.read_csv(".\Data\\test_data\\C_1513_BACE_2.csv")
#data_CLINTOX_2 = pd.read_csv(".\Data\\test_data\\C_1478_CLINTOX_2.csv")
#data_BBBP_erroneous_smiles = pd.read_csv(".\Data\\test_data\\C_2039_BBBP_2_erroneous_smiles).csv") 
#data_CLINTOX_2_erroneous_smiles = pd.read_csv(".\Data\\test_data\\C_1484_CLINTOX_2_erroneous_smiles).csv")


# Construct plotter object
#cp = Plotter.from_smiles(molecules_data_aqsoldb["SMILES"])
#cp = Plotter.from_smiles(molecules_data_solubility_dataset_E["SMILES"], sim_type="structural")
#cp = Plotter.from_smiles(data_LOGS["smiles"], target=data_LOGS["target"], target_type="R", sim_type="structural")
#cp = Plotter.from_smiles(data_SAMPL["smiles"], target=data_SAMPL["target"], target_type="R")
#cp = Plotter.from_smiles(data_AQSOLDB["smiles"], target=data_AQSOLDB["target"], target_type="R")
#cp = Plotter.from_smiles(data_LOGP["smiles"], target=data_LOGP["target"], target_type="R", sim_type="structural")
#cp = Plotter.from_smiles(data_BACE["smiles"], target=data_BACE["target"], target_type="R", sim_type="structural")
#cp = Plotter.from_smiles(data_BBBP["smiles"], target=data_BBBP["target"], target_type="C", sim_type="structural")
#cp = Plotter.from_smiles(data_HIV_3["smiles"], target=data_HIV_3["target"], target_type="C", sim_type="structural")
#cp = Plotter.from_smiles(data_HIV_2["smiles"], target=data_HIV_2["target"], target_type="C", sim_type="structural")
#cp = Plotter.from_smiles(data_BACE_2["smiles"], target=data_BACE_2["target"], target_type="C", sim_type="structural")
#cp = Plotter.from_smiles(data_CLINTOX_2["smiles"], target=data_CLINTOX_2["target"], target_type="C")
#cp = Plotter.from_smiles(data_BBBP_erroneous_smiles["smiles"], target=data_BBBP_erroneous_smiles["target"], target_type="C", sim_type="tailored")
#cp = Plotter.from_smiles(data_CLINTOX_2_erroneous_smiles["smiles"], target=data_CLINTOX_2_erroneous_smiles["target"], target_type="C", sim_type="tailored")

# Plot the data
#p = cp.pca(kind="hex", remove_outliers=False, is_colored=True, colorbar=True)
#cp.tsne(kind = "scatter", pca=True, is_colored=True, random_state=2)

#cp.umap(kind="scatter", colorbar=True)
capturedOutput = io.StringIO()
sys.stdout = capturedOutput
data_LOGS = pd.read_csv(".\Data\\test_data\\R_1291_LOGS.csv") 
cp = Plotter.from_smiles(data_LOGS["smiles"], target=data_LOGS["target"], target_type="R")

sys.stdout = sys.__stdout__
print(capturedOutput.getvalue())
#p2= cp.umap(kind="kde",random_state=1)
