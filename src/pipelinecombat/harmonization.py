"""
Harmonization module using neuroCombat for statistical harmonization
of neuroimaging data across different scanners and sites.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

try:
    from neuroHarmonize import harmonizationLearn, harmonizationApply
    NEUROCOMBAT_AVAILABLE = True
except ImportError:
    logger.warning("neuroCombat (neuroHarmonize) not available. Install with: uv add neuroHarmonize")
    NEUROCOMBAT_AVAILABLE = False


def harmonize_data(
    data: Union[np.ndarray, pd.DataFrame],
    covars: pd.DataFrame,
    batch_col: str = 'SITE',
    smooth_terms: Optional[List[str]] = None,
    smooth_term_bounds: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Harmonize neuroimaging data across different sites/scanners using neuroCombat.
    
    Args:
        data: Data matrix with features as columns and subjects as rows
        covars: Covariate dataframe with same number of rows as data
        batch_col: Column name in covars that contains batch/site information
        smooth_terms: List of covariate names to include as smooth terms
        smooth_term_bounds: Bounds for smooth terms (min, max)
    
    Returns:
        Tuple of (harmonized_data, model_info)
    """
    if not NEUROCOMBAT_AVAILABLE:
        raise ImportError("neuroCombat is required for harmonization")
    
    logger.info(f"Starting harmonization for {data.shape[0]} subjects, {data.shape[1]} features")
    
    # Convert to numpy array if pandas DataFrame
    if isinstance(data, pd.DataFrame):
        data_array = data.values
    else:
        data_array = data.copy()
    
    # Ensure covariates include batch information
    if batch_col not in covars.columns:
        # If specified batch_col doesn't exist, try to rename 'site' to the expected column
        if 'site' in covars.columns:
            covars_copy = covars.copy()
            covars_copy[batch_col] = covars_copy['site']
        else:
            raise ValueError(f"Batch column '{batch_col}' not found in covariates")
    else:
        covars_copy = covars.copy()
    
    # Perform harmonization
    try:
        # Encode categorical variables
        covars_encoded = covars_copy.copy()
        
        # Convert categorical columns to numeric
        for col in covars_encoded.columns:
            if col != batch_col and covars_encoded[col].dtype == 'object':
                # Convert categorical to numeric (e.g., M/F -> 0/1)
                if col == 'sex':
                    covars_encoded[col] = (covars_encoded[col] == 'M').astype(int)
                else:
                    # For other categorical variables, use label encoding
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    covars_encoded[col] = le.fit_transform(covars_encoded[col])
        
        model, data_harmonized = harmonizationLearn(
            data_array.T,  # neuroCombat expects features x subjects
            covars_encoded
        )
        
        logger.info("Harmonization completed successfully")
        
        return data_harmonized.T, model  # Return subjects x features
        
    except Exception as e:
        logger.error(f"Harmonization failed: {str(e)}")
        raise


def apply_harmonization(
    data: Union[np.ndarray, pd.DataFrame],
    covars: pd.DataFrame,
    model: Dict
) -> np.ndarray:
    """
    Apply pre-trained harmonization model to new data.
    
    Args:
        data: New data to harmonize
        covars: Covariates for the new data
        model: Pre-trained harmonization model
    
    Returns:
        Harmonized data array
    """
    if not NEUROCOMBAT_AVAILABLE:
        raise ImportError("neuroCombat is required for harmonization")
    
    logger.info(f"Applying harmonization to {data.shape[0]} subjects")
    
    # Convert to numpy array if pandas DataFrame
    if isinstance(data, pd.DataFrame):
        data_array = data.values
    else:
        data_array = data.copy()
    
    try:
        data_harmonized = harmonizationApply(
            data_array.T,  # neuroCombat expects features x subjects
            covars,
            model
        )
        
        logger.info("Harmonization application completed")
        return data_harmonized.T  # Return subjects x features
        
    except Exception as e:
        logger.error(f"Harmonization application failed: {str(e)}")
        raise


def create_example_data(
    n_subjects: int = 100,
    n_features: int = 50,
    n_sites: int = 3
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Create example neuroimaging data for testing harmonization.
    
    Args:
        n_subjects: Number of subjects
        n_features: Number of features (e.g., brain regions)
        n_sites: Number of scanning sites
    
    Returns:
        Tuple of (data, covariates)
    """
    np.random.seed(42)
    
    # Create synthetic neuroimaging data with site effects
    data = np.random.randn(n_subjects, n_features)
    
    # Add site effects
    sites = np.random.choice([f'site_{i}' for i in range(n_sites)], n_subjects)
    ages = np.random.uniform(18, 80, n_subjects)
    sex = np.random.choice(['M', 'F'], n_subjects)
    
    # Add site-specific bias
    for i, site in enumerate(np.unique(sites)):
        site_mask = sites == site
        site_bias = np.random.normal(i * 0.5, 0.2, n_features)
        data[site_mask] += site_bias
    
    # Create covariates dataframe
    covars = pd.DataFrame({
        'SITE': sites,
        'age': ages,
        'sex': sex
    })
    
    logger.info(f"Created example data: {n_subjects} subjects, {n_features} features, {n_sites} sites")
    
    return data, covars
