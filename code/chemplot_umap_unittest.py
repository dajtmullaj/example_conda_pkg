import unittest
from unittest.mock import patch

from chemplot import Plotter
import pandas as pd 
import numpy as np
from scipy import stats
from matplotlib import pyplot
from io import StringIO

class TestUMAP(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.data_LOGS = pd.read_csv(".\Data\\test_data\\R_1291_LOGS.csv") 
        cls.data_BBBP = pd.read_csv(".\Data\\test_data\\C_2039_BBBP_2.csv")  
        cls.plotter_tailored_LOGS = Plotter.from_smiles(cls.data_LOGS["smiles"], target=cls.data_LOGS["target"], target_type="R", sim_type="tailored")
        cls.plotter_tailored_BBBP = Plotter.from_smiles(cls.data_BBBP["smiles"], target=cls.data_BBBP["target"], target_type="C", sim_type="tailored")
        #cls.plotter_structural_LOGS = Plotter.from_smiles(cls.data_LOGS["smiles"], target=cls.data_LOGS["target"], target_type="R", sim_type="structural")
        #cls.plotter_structural_BBBP = Plotter.from_smiles(cls.data_BBBP["smiles"], target=cls.data_BBBP["target"], target_type="C", sim_type="structural")
      
    def test_default_n_neighbors(self):
        """
        1. Test checks if default n_neighbors is assigned
        """
        self.plotter_tailored_LOGS.umap(random_state=None, min_dist=0.9, kind='scatter', size=20, remove_outliers=False, is_colored=True, colorbar=False)
        self.assertEqual(self.plotter_tailored_LOGS.umap_fit.n_neighbors, max(2, min(15, len(self.plotter_tailored_LOGS.data)//4)))
        pyplot.close()
    
    def test_n_neighbors_below_2(self):
        """
        2. Test checks if input n_neighbors is valid
        """
        self.plotter_tailored_LOGS.umap(n_neighbors=1, random_state=None, min_dist=0.9, kind='scatter', size=20, remove_outliers=False, is_colored=True, colorbar=False)
        self.assertEqual(self.plotter_tailored_LOGS.umap_fit.n_neighbors, 2)
        pyplot.close()
        
    def test_n_neighbors(self):
        """
        3. Test checks if n_neighbors is assigned
        """
        self.plotter_tailored_LOGS.umap(n_neighbors=30, random_state=None, min_dist=0.9, kind='scatter', size=20, remove_outliers=False, is_colored=True, colorbar=False)
        self.assertEqual(self.plotter_tailored_LOGS.umap_fit.n_neighbors, max(2, min(30, len(self.plotter_tailored_LOGS.data)//4)))
        pyplot.close()
        
    def test_n_neighbors_above_quarter(self):
        """
        4. Test checks if input n_neighbors is valid
        """
        self.plotter_tailored_LOGS.umap(n_neighbors=1e+308, random_state=None, min_dist=0.9, kind='scatter', size=20, remove_outliers=False, is_colored=True, colorbar=False)
        self.assertEqual(self.plotter_tailored_LOGS.umap_fit.n_neighbors, len(self.plotter_tailored_LOGS.data)//4)
        pyplot.close()
     
    @patch('sys.stdout', new_callable=StringIO)      
    def test_INFO_n_neighbors(self, mock_stdout):
        """
        5. Test checks if user is informed about n_neighbors
        """
        self.plotter_tailored_LOGS.umap(n_neighbors=1e+308, random_state=None, min_dist=0.9, kind='scatter', size=20, remove_outliers=False, is_colored=True, colorbar=False)
        assert str('n_neighbors represents the size of the local neighborhood UMAP will look at when attempting to learn the manifold structure of the data./n'+
                   'Robust results are obtained for values of n_neighbors between 2 up to a quarter of the data. The inputed value is outside that range.\n'+
                   'Therefore the closest value, between 2 and a quarter of the data, to the parameter inputed has been used in the method.') in mock_stdout.getvalue()
        pyplot.close()
        
    def test_default_random_state(self):
        """
        6. Test checks if default random_state is assigned
        """
        self.plotter_tailored_LOGS.umap(n_neighbors=15, min_dist=0.9, kind='scatter', size=20, remove_outliers=False, is_colored=True, colorbar=False)
        self.assertIsInstance(self.plotter_tailored_LOGS.umap_fit.random_state, type(None))
        pyplot.close()
        
    def test_random_state(self):
        """
        7. Test checks if random_state is assigned
        """
        self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=1, min_dist=0.9, kind='scatter', size=20, remove_outliers=False, is_colored=True, colorbar=False)
        self.assertEqual(self.plotter_tailored_LOGS.umap_fit.random_state, 1)
        pyplot.close()
        
    def test_default_min_dist(self):
        """
        8. Test checks if default min_dist is assigned
        """
        self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=1, kind='scatter', size=20, remove_outliers=False, is_colored=True, colorbar=False)
        self.assertEqual(self.plotter_tailored_LOGS.umap_fit.min_dist, 0.9)
        pyplot.close()
        
    def test_min_dist(self):
        """
        9. Test checks if min_dist is assigned
        """
        self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=1, min_dist=0.5, kind='scatter', size=20, remove_outliers=False, is_colored=True, colorbar=False)
        self.assertEqual(self.plotter_tailored_LOGS.umap_fit.min_dist, 0.5)
        pyplot.close()
        
    def test_min_dist_below_0(self):
        """
        10. Test checks if min_dist is valid
        """
        self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=1, min_dist=-1, kind='scatter', size=20, remove_outliers=False, is_colored=True, colorbar=False)
        self.assertEqual(self.plotter_tailored_LOGS.umap_fit.min_dist, 0.0)
        pyplot.close()
        
    def test_min_dist_above_1_included(self):
        """
        11. Test checks if min_dist is valid
        """
        self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=1, min_dist=1, kind='scatter', size=20, remove_outliers=False, is_colored=True, colorbar=False)
        self.assertEqual(self.plotter_tailored_LOGS.umap_fit.min_dist, 0.99)
        pyplot.close()
        
    @patch('sys.stdout', new_callable=StringIO) 
    def test_INFO_min_dist(self, mock_stdout):
        """
        12. Test checks if user is informed about perplexity
        """
        self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=None, min_dist=1, kind='scatter', size=20, remove_outliers=False, is_colored=True, colorbar=False)
        assert str('min_dist controls how tightly UMAP is allowed to pack points together../n'+
                   'The value of min_dist can range from 0.0 up to 0.99. The inputed value is outside that range.\n'+
                   'Therefore the closest value between 0.0 and 0.99 to the parameter inputed has been used in the method.') in mock_stdout.getvalue()
        pyplot.close()
        
    def test_default_kind_none(self):
        """
        13. Test checks if default kind is assigned
        """
        result = self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=None, min_dist=0.9, size=20, remove_outliers=False, is_colored=True, colorbar=False)
        self.assertEqual(result.get_label(), "scatter")
        pyplot.close()
    
    def test_default_kind(self):
        """
        14. Test checks if default kind is assigned with anytext
        """
        result = self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=None, min_dist=0.9, kind='anytext', size=20, remove_outliers=False, is_colored=True, colorbar=False)
        self.assertEqual(result.get_label(), "scatter")
        pyplot.close()
        
    @patch('sys.stdout', new_callable=StringIO) 
    def test_INFO_kind_with_anytext(self, mock_stdout):
        """
        15. Test checks if user is informed about kind
        """
        self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=None, min_dist=0.9,kind='anytext', size=20, remove_outliers=False, is_colored=True, colorbar=False)
        assert str('kind indicates which type of plot must be visualized. Currently supported visualization are:\n'+
                   '-scatter plot (scatter)\n'+
                   '-hexagon plot (hex)\n'+
                   '-kernel density estimation plot (kde)\n'+
                   'Please input one between scatter, hex or kde for parameter kind.\n'+
                   'As default scatter has been taken.') in mock_stdout.getvalue()
        pyplot.close()
        
    def test_default_is_colored(self):
        """
        16. Test checks if default is_colored is assigned
        """
        result = self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=None, min_dist=0.9, kind='scatter', size=20, remove_outliers=False, colorbar=False)
        self.assertTrue(len(result.collections)>1)
        pyplot.close()
        
    def test_default_remove_outliers(self):
        """
        17. Test checks if default remove_outliers is assigned
        """
        self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=None, min_dist=0.9, kind='scatter', size=20, is_colored=True, colorbar=False)
        self.assertTrue(self.plotter_tailored_LOGS.df_plot_xy.equals(self.plotter_tailored_LOGS.df_2_components))
        pyplot.close()
        
    def test_default_size(self):
        """
        18. Test checks if default size is assigned
        """
        result = self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=None, min_dist=0.9, kind='scatter', remove_outliers=False, is_colored=True, colorbar=False)
        self.assertEqual(result.figure.get_size_inches()[0], 20)
        self.assertEqual(result.figure.get_size_inches()[1], 20)
        pyplot.close()
        
    def test_kind_scatter(self):
        """
        19. Test checks if kind is assigned
        """
        result = self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=None, min_dist=0.9, kind='scatter', size=20, remove_outliers=False, is_colored=True, colorbar=False)
        self.assertEqual(result.get_label(), "scatter")
        pyplot.close()
        
    def test_is_colored_true_scatter(self):
        """
        20. Test checks if is_colored is assigned
        """
        result = self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=None, min_dist=0.9, kind='scatter', size=20, remove_outliers=False, is_colored=True, colorbar=False)
        self.assertTrue(len(result.collections)>1)
        pyplot.close()
        
    def test_is_colored_false_scatter(self):
        """
        21. Test checks if is_colored is assigned
        """
        result = self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=None, min_dist=0.9, kind='scatter', size=20, remove_outliers=False, is_colored=False, colorbar=False)
        self.assertTrue(len(result.collections) == 1)
        pyplot.close()
        
    def test_remove_outliers_false_scatter(self):
        """
        22. Test checks if remove_outliers is assigned
        """
        self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=None, min_dist=0.9, kind='scatter', size=20, remove_outliers=False, is_colored=True, colorbar=False)
        self.assertTrue(self.plotter_tailored_LOGS.df_plot_xy.equals(self.plotter_tailored_LOGS.df_2_components))
        pyplot.close()
        
    def test_remove_outliers_true_scatter(self):
        """
        23. Test checks if remove_outliers is assigned
        """
        self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=None, min_dist=0.9, kind='scatter', size=20, remove_outliers=True, is_colored=True, colorbar=False)
        df_no_outliers = self.plotter_tailored_LOGS.df_2_components[(np.abs(stats.zscore(self.plotter_tailored_LOGS.df_2_components))<3).all(axis=1)]
        self.assertTrue(self.plotter_tailored_LOGS.df_plot_xy.equals(df_no_outliers))
        pyplot.close()
        
    def test_size_scatter(self):
        """
        24. Test checks if size is assigned
        """
        result = self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=None, min_dist=0.9, kind='scatter', size=20, remove_outliers=False, is_colored=True, colorbar=False)
        self.assertEqual(result.figure.get_size_inches()[0], 20)
        self.assertEqual(result.figure.get_size_inches()[1], 20)
        pyplot.close()
    
    def test_kind_hex(self):
        """
        25. Test checks if kind is assigned
        """
        result = self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=None, min_dist=0.9, kind='hex', size=20, remove_outliers=False, is_colored=True, colorbar=False)
        self.assertEqual(result.get_label(), "hex")
        pyplot.close()
        
    def test_remove_outliers_false_hex(self):
        """
        26. Test checks if remove_outliers is assigned
        """
        self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=None, min_dist=0.9, kind='hex', size=20, remove_outliers=False, is_colored=True, colorbar=False)
        self.assertTrue(self.plotter_tailored_LOGS.df_plot_xy.equals(self.plotter_tailored_LOGS.df_2_components))
        pyplot.close()
        
    def test_remove_outliers_true_hex(self):
        """
        27. Test checks if remove_outliers is assigned
        """
        self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=None, min_dist=0.9, kind='hex', size=20, remove_outliers=True, is_colored=True, colorbar=False)
        df_no_outliers = self.plotter_tailored_LOGS.df_2_components[(np.abs(stats.zscore(self.plotter_tailored_LOGS.df_2_components))<3).all(axis=1)]
        self.assertTrue(self.plotter_tailored_LOGS.df_plot_xy.equals(df_no_outliers))
        pyplot.close()
        
    def test_size_hex(self):
        """
        28. Test checks if size is assigned
        """
        result = self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=None, min_dist=0.9, kind='hex', size=20, remove_outliers=False, is_colored=True, colorbar=False)
        self.assertEqual(result.figure.get_size_inches()[0], 20)
        self.assertEqual(result.figure.get_size_inches()[1], 20)
        pyplot.close()
        
    def test_kind_kde(self):
        """
        29. Test checks if kind is assigned
        """
        result = self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=None, min_dist=0.9, kind='kde', size=20, remove_outliers=False, is_colored=True, colorbar=False)
        self.assertEqual(result.get_label(), "kde")
        pyplot.close()
        
    def test_remove_outliers_false_kde(self):
        """
        30. Test checks if remove_outliers is assigned
        """
        self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=None, min_dist=0.9, kind='kde', size=20, remove_outliers=False, is_colored=True, colorbar=False)
        self.assertTrue(self.plotter_tailored_LOGS.df_plot_xy.equals(self.plotter_tailored_LOGS.df_2_components))
        pyplot.close()
        
    def test_remove_outliers_true_kde(self):
        """
        31. Test checks if remove_outliers is assigned
        """
        self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=None, min_dist=0.9, kind='kde', size=20, remove_outliers=True, is_colored=True, colorbar=False)
        df_no_outliers = self.plotter_tailored_LOGS.df_2_components[(np.abs(stats.zscore(self.plotter_tailored_LOGS.df_2_components))<3).all(axis=1)]
        self.assertTrue(self.plotter_tailored_LOGS.df_plot_xy.equals(df_no_outliers))
        pyplot.close()
        
    def test_size_kde(self):
        """
        32. Test checks if size is assigned
        """
        result = self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=None, min_dist=0.9, kind='kde', size=20, remove_outliers=False, is_colored=True, colorbar=False)
        self.assertEqual(result.figure.get_size_inches()[0], 20)
        self.assertEqual(result.figure.get_size_inches()[1], 20)
        pyplot.close()
        
    def test_default_colorbar(self):
        """
        33. Test checks if default value of colorbar is assigned
        """
        result = self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=None, min_dist=0.9, kind='scatter', size=20, remove_outliers=False, is_colored=True)
        self.assertNotIsInstance(result.get_legend(), type(None))
        self.assertEqual(len(result.figure.axes), 1)
        pyplot.close()
        
    def test_colorbar_R_remove_legend(self):
        """
        34. Test checks if colorbar is assigned when target type is R and therefore legend removed
        """
        result = self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=None, min_dist=0.9, kind='scatter', size=20, remove_outliers=False, is_colored=True, colorbar=True)
        self.assertIsInstance(result.get_legend(), type(None))
        pyplot.close()
        
    def test_colorbar_C_keep_legend(self):
        """
        35. Test checks if colorbar is ignored when target type is C and therefore legend kept
        """
        result = self.plotter_tailored_BBBP.umap(n_neighbors=15, random_state=None, min_dist=0.9, kind='scatter', size=20, remove_outliers=False, is_colored=True, colorbar=True)
        self.assertNotIsInstance(result.get_legend(), type(None))
        pyplot.close()
        
    def test_colorbar_R_add_colorbar(self):
        """
        36. Test checks if colorbar is assigned when target type is R
        """
        result = self.plotter_tailored_LOGS.umap(n_neighbors=15, random_state=None, min_dist=0.9, kind='scatter', size=20, remove_outliers=False, is_colored=True, colorbar=True)
        self.assertTrue(len(result.figure.axes)>=1)
        pyplot.close()
        
    def test_colorbar_C_ignore_colorbar(self):
        """
        37. Test checks if colorbar is ignored when target type is C 
        """
        result = self.plotter_tailored_BBBP.umap(n_neighbors=15, random_state=None, min_dist=0.9, kind='scatter', size=20, remove_outliers=False, is_colored=True, colorbar=True)
        self.assertTrue(len(result.figure.axes)==1)
        pyplot.close()
        
    
if __name__ == '__main__':
    unittest.main()
    
