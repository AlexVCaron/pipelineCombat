"""
Statistical modeling module using statsmodels for neuroimaging data analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

logger = logging.getLogger(__name__)

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.contingency_tables import mcnemar
    STATSMODELS_AVAILABLE = True
except ImportError:
    logger.warning("statsmodels not available. Install with: uv add statsmodels")
    STATSMODELS_AVAILABLE = False


class NeuroStatAnalyzer:
    """Statistical analysis class for neuroimaging data using statsmodels."""
    
    def __init__(self):
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for statistical analysis")
        
        self.results = {}
    
    def linear_regression(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        covariates: pd.DataFrame,
        formula: Optional[str] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform linear regression analysis on neuroimaging data.
        
        Args:
            data: Neuroimaging data (subjects x features)
            covariates: Covariate dataframe
            formula: Optional formula string for statsmodels
            feature_names: Names for features (brain regions, etc.)
        
        Returns:
            Dictionary containing regression results
        """
        logger.info("Performing linear regression analysis...")
        
        if isinstance(data, pd.DataFrame):
            data_array = data.values
            if feature_names is None:
                feature_names = data.columns.tolist()
        else:
            data_array = data
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(data_array.shape[1])]
        
        n_subjects, n_features = data_array.shape
        
        if len(covariates) != n_subjects:
            raise ValueError("Number of subjects in data and covariates must match")
        
        results = {
            'coefficients': [],
            'pvalues': [],
            'tvalues': [],
            'rsquared': [],
            'feature_names': feature_names
        }
        
        # Perform regression for each feature
        for i, feature_name in enumerate(feature_names):
            y = data_array[:, i]
            
            # Create dataframe for this analysis
            df = covariates.copy()
            df['y'] = y
            
            try:
                if formula:
                    model = smf.ols(formula, data=df)
                else:
                    # Default formula with all covariates
                    predictors = ' + '.join(covariates.columns)
                    formula_str = f"y ~ {predictors}"
                    model = smf.ols(formula_str, data=df)
                
                fitted_model = model.fit()
                
                results['coefficients'].append(fitted_model.params.values)
                results['pvalues'].append(fitted_model.pvalues.values)
                results['tvalues'].append(fitted_model.tvalues.values)
                results['rsquared'].append(fitted_model.rsquared)
                
            except Exception as e:
                logger.warning(f"Regression failed for feature {feature_name}: {str(e)}")
                # Fill with NaN values
                n_params = len(covariates.columns) + 1  # +1 for intercept
                results['coefficients'].append([np.nan] * n_params)
                results['pvalues'].append([np.nan] * n_params)
                results['tvalues'].append([np.nan] * n_params)
                results['rsquared'].append(np.nan)
        
        # Convert to arrays
        results['coefficients'] = np.array(results['coefficients'])
        results['pvalues'] = np.array(results['pvalues'])
        results['tvalues'] = np.array(results['tvalues'])
        results['rsquared'] = np.array(results['rsquared'])
        
        logger.info(f"Linear regression completed for {n_features} features")
        
        self.results['linear_regression'] = results
        return results
    
    def multiple_comparisons_correction(
        self,
        pvalues: np.ndarray,
        method: str = 'fdr_bh',
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Apply multiple comparisons correction to p-values.
        
        Args:
            pvalues: Array of p-values
            method: Correction method ('bonferroni', 'fdr_bh', etc.)
            alpha: Significance level
        
        Returns:
            Tuple of (reject, pvals_corrected, alphacSidak, alphacBonf)
        """
        logger.info(f"Applying multiple comparisons correction: {method}")
        
        # Flatten p-values if multidimensional
        original_shape = pvalues.shape
        pvals_flat = pvalues.flatten()
        
        # Remove NaN values for correction
        valid_mask = ~np.isnan(pvals_flat)
        valid_pvals = pvals_flat[valid_mask]
        
        if len(valid_pvals) == 0:
            logger.warning("No valid p-values for correction")
            return np.zeros_like(pvalues, dtype=bool), pvalues, alpha, alpha
        
        # Apply correction
        reject_valid, pvals_corrected_valid, alphacSidak, alphacBonf = multipletests(
            valid_pvals, alpha=alpha, method=method
        )
        
        # Reconstruct full arrays
        reject_flat = np.zeros_like(pvals_flat, dtype=bool)
        pvals_corrected_flat = np.ones_like(pvals_flat)
        
        reject_flat[valid_mask] = reject_valid
        pvals_corrected_flat[valid_mask] = pvals_corrected_valid
        
        # Reshape back to original shape
        reject = reject_flat.reshape(original_shape)
        pvals_corrected = pvals_corrected_flat.reshape(original_shape)
        
        n_significant = np.sum(reject)
        logger.info(f"Found {n_significant} significant results after correction")
        
        return reject, pvals_corrected, alphacSidak, alphacBonf
    
    def anova_analysis(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        groups: Union[List, np.ndarray, pd.Series],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform one-way ANOVA analysis across groups.
        
        Args:
            data: Neuroimaging data (subjects x features)
            groups: Group labels for each subject
            feature_names: Names for features
        
        Returns:
            Dictionary containing ANOVA results
        """
        logger.info("Performing ANOVA analysis...")
        
        from scipy.stats import f_oneway
        
        if isinstance(data, pd.DataFrame):
            data_array = data.values
            if feature_names is None:
                feature_names = data.columns.tolist()
        else:
            data_array = data
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(data_array.shape[1])]
        
        unique_groups = np.unique(groups)
        n_features = data_array.shape[1]
        
        results = {
            'f_statistics': [],
            'pvalues': [],
            'feature_names': feature_names,
            'groups': unique_groups
        }
        
        # Perform ANOVA for each feature
        for i, feature_name in enumerate(feature_names):
            feature_data = data_array[:, i]
            
            # Split data by groups
            group_data = [feature_data[groups == group] for group in unique_groups]
            
            try:
                f_stat, p_val = f_oneway(*group_data)
                results['f_statistics'].append(f_stat)
                results['pvalues'].append(p_val)
            except Exception as e:
                logger.warning(f"ANOVA failed for feature {feature_name}: {str(e)}")
                results['f_statistics'].append(np.nan)
                results['pvalues'].append(np.nan)
        
        results['f_statistics'] = np.array(results['f_statistics'])
        results['pvalues'] = np.array(results['pvalues'])
        
        logger.info(f"ANOVA completed for {n_features} features")
        
        self.results['anova'] = results
        return results
    
    def correlation_analysis(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        behavioral_scores: Union[np.ndarray, pd.Series],
        method: str = 'pearson',
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform correlation analysis between brain data and behavioral scores.
        
        Args:
            data: Neuroimaging data (subjects x features)
            behavioral_scores: Behavioral scores for each subject
            method: Correlation method ('pearson', 'spearman')
            feature_names: Names for features
        
        Returns:
            Dictionary containing correlation results
        """
        logger.info(f"Performing {method} correlation analysis...")
        
        from scipy.stats import pearsonr, spearmanr
        
        if isinstance(data, pd.DataFrame):
            data_array = data.values
            if feature_names is None:
                feature_names = data.columns.tolist()
        else:
            data_array = data
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(data_array.shape[1])]
        
        n_features = data_array.shape[1]
        
        results = {
            'correlations': [],
            'pvalues': [],
            'feature_names': feature_names
        }
        
        # Choose correlation function
        corr_func = pearsonr if method == 'pearson' else spearmanr
        
        # Perform correlation for each feature
        for i, feature_name in enumerate(feature_names):
            feature_data = data_array[:, i]
            
            # Remove NaN values
            valid_mask = ~(np.isnan(feature_data) | np.isnan(behavioral_scores))
            
            if np.sum(valid_mask) < 3:  # Need at least 3 points
                results['correlations'].append(np.nan)
                results['pvalues'].append(np.nan)
                continue
            
            try:
                corr, p_val = corr_func(
                    feature_data[valid_mask],
                    behavioral_scores[valid_mask]
                )
                results['correlations'].append(corr)
                results['pvalues'].append(p_val)
            except Exception as e:
                logger.warning(f"Correlation failed for feature {feature_name}: {str(e)}")
                results['correlations'].append(np.nan)
                results['pvalues'].append(np.nan)
        
        results['correlations'] = np.array(results['correlations'])
        results['pvalues'] = np.array(results['pvalues'])
        
        logger.info(f"Correlation analysis completed for {n_features} features")
        
        self.results['correlation'] = results
        return results


def create_example_behavioral_data(n_subjects: int = 100) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Create example neuroimaging and behavioral data for statistical testing.
    
    Args:
        n_subjects: Number of subjects
    
    Returns:
        Tuple of (brain_data, behavioral_scores, covariates)
    """
    np.random.seed(42)
    
    # Create synthetic brain data
    n_regions = 68  # Common atlas size
    brain_data = np.random.randn(n_subjects, n_regions) * 0.5 + 1.0
    
    # Create behavioral scores correlated with some brain regions
    behavioral_scores = np.random.randn(n_subjects)
    # Add correlation with first few regions
    for i in range(5):
        behavioral_scores += brain_data[:, i] * 0.3
    
    # Add noise
    behavioral_scores += np.random.randn(n_subjects) * 0.5
    
    # Create covariates
    ages = np.random.uniform(18, 80, n_subjects)
    sex = np.random.choice(['M', 'F'], n_subjects)
    groups = np.random.choice(['Control', 'Patient'], n_subjects)
    
    covariates = pd.DataFrame({
        'age': ages,
        'sex': sex,
        'group': groups
    })
    
    logger.info(f"Created example data: {n_subjects} subjects, {n_regions} brain regions")
    
    return brain_data, behavioral_scores, covariates
