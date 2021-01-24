"""
Created on Sat Oct  3 21:21:19 2020

Project: chemplot (Chemical Space Visualization)
Content: Test file for descriptor

@author: murat cihan sorkun
"""

import pandas as pd 
import descriptors as desc


data_LOGS = pd.read_csv(".\Data\\test_data\\R_1291_LOGS.csv") 
df_descriptors_LOGS=desc.get_mordred_descriptors(data_LOGS["smiles"])
selected_descriptors_LOGS=desc.select_descriptors_lasso(df_descriptors_LOGS,data_LOGS["target"],kind="R",R_select=0.05)


# data_AQSOLDB = pd.read_csv("..\Data\\test_data\\R_9982_AQSOLDB.csv") 
# df_descriptors_AQSOLDB=desc.get_mordred_descriptors(data_AQSOLDB["smiles"])
# selected_descriptors_AQSOLDB=desc.select_descriptors_lasso(df_descriptors_AQSOLDB,data_AQSOLDB["target"],kind="R",R_select=0.05)


data_SAMPL = pd.read_csv(".\Data\\test_data\\R_642_SAMPL.csv") 
df_descriptors_SAMPL=desc.get_mordred_descriptors(data_SAMPL["smiles"])
selected_descriptors_SAMPL=desc.select_descriptors_lasso(df_descriptors_SAMPL,data_SAMPL["target"],kind="R",R_select=0.05)


data_BBBP = pd.read_csv(".\Data\\test_data\\C_2039_BBBP_2.csv") 
df_descriptors_BBBP=desc.get_mordred_descriptors(data_BBBP["smiles"])
selected_descriptors_BBBP=desc.select_descriptors_lasso(df_descriptors_BBBP,data_BBBP["target"],kind="C",C_select=0.05)


data_HIV_3 = pd.read_csv(".\Data\\test_data\\C_41127_HIV_3.csv") 
df_descriptors_HIV_3=desc.get_mordred_descriptors(data_HIV_3["smiles"])
selected_descriptors_HIV_3=desc.select_descriptors_lasso(df_descriptors_HIV_3,data_HIV_3["target"],kind="C",C_select=0.05)



    
    

 

            
        
    




