"""
Principal Component Analysis design matrix implementation.

This module provides PCA-based design matrix functionality for dimensionality
reduction and statistical modeling in neuroimaging data harmonization.
"""
import numpy as np
from sklearn.decomposition import PCA as SKPCA

from .design import DesignMatrix


class PCA(DesignMatrix):
    """
    Principal Component Analysis design matrix.

    A design matrix implementation that uses PCA to reduce dimensionality
    and create a basis for statistical modeling. Extends DesignMatrix
    to handle numerical data transformation through PCA.
    """

    class Model(DesignMatrix.Model):
        """
        PCA model with variance-aware standardization.

        Extends the base DesignMatrix Model to handle PCA-specific
        variance calculations for standardization operations.
        """

        def __init__(self, x, variances=None):
            super().__init__(x)
            self.variances = variances

        def standard(self, y, beta=None, sigma=None, y_hat=None):
            """
            Apply the PCA standardization model to the input data.

            The method uses the design matrix for PCA-based standardization
            of neuroimaging data with variance considerations.

            Parameters
            ----------
            y : array-like, shape (n_samples, n_features)
                The input data.
            beta : array-like, shape (n_coeff, n_features), optional
                The model coefficients.
            sigma : float, optional
                The standard deviation to use for scaling.
            y_hat : array-like, shape (n_samples, n_features), optional
                Precomputed predicted values.

            Returns
            -------
            y_std : array-like, shape (n_samples, n_features)
                The standardized input data.
            """
            if sigma is None:
                sigma = np.sqrt(np.sum(self.variances))
            return super().standard(y, beta=beta, sigma=sigma, y_hat=y_hat)

    def __init__(self, numerical_data, n_components=None):
        """
        Initialize the PCA design matrix.

        Parameters
        ----------
        numerical_data : array-like, shape (n_samples, n_features)
            The numerical data to perform PCA on.
        n_components : int, optional
            The number of principal components to retain. If None,
            all components are retained.
        """
        # Don't call super().__init__ as we're not using categorical data
        pca = SKPCA(n_components=n_components)
        model = pca.fit(numerical_data)

        # Get original components and data mean from sklearn PCA
        original_components = model.components_  # (n_components, n_features)
        data_mean = model.mean_  # (n_features,)

        # Create augmented components with mean as component 0
        # Shape becomes (n_components + 1, n_features)
        augmented_components = np.vstack(
            [
                data_mean[np.newaxis, :],  # Add mean as first row
                original_components,
            ]
        )

        # Store as transposed for DesignMatrix convention
        # Shape: (n_features, n_components + 1)
        self._n_block = [augmented_components.T]

        # Augmented variances: use total variance for mean component
        total_variance = np.sum(model.explained_variance_)
        augmented_variances = np.concatenate(
            [
                [total_variance],  # Variance for mean component
                model.explained_variance_,
            ]
        )
        self._n_variance = augmented_variances

    def generate(self):
        """
        Generate the PCA model with variance information.

        Returns
        -------
        _dm : Model
            A PCA model instance with variance attributes set.
        """
        _dm = super().generate()
        _dm.variances = self._n_variance
        return _dm
