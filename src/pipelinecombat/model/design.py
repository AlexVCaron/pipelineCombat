import numpy as np
import pandas as pd


class DesignMatrix:
    def __init__(
        self, categorical_data, numerical_data=None, batch_col_index=0
    ):
        """
        Parameters
        ----------
        categorical_data : pd.DataFrame, required
            The input categorical data. At least one column must describe
            the batch of the samples. Shape: (n_samples, n_variable)
        numerical_data : pd.DataFrame, np.array, optional
            The input numerical data. Shape: (n_samples, n_features).
        batch_col_index : any, optional
            The index of the batch column in the categorical data.
        """
        if isinstance(batch_col_index, int):
            batch_col_index = categorical_data.columns[batch_col_index]

        non_batch_indexes = categorical_data.columns[
            categorical_data.columns != batch_col_index
        ]

        # Always create batch block (intercepts) - this is required
        print("batch block encoding")
        self._b_block = self._batch_categorical(
            categorical_data, indexes=[batch_col_index]
        )
        self.n_batches = self._b_block.shape[1]
        # self._b_block *= np.sum(self._b_block, axis=0) / self.n_batches

        print("categorical block encoding")
        # Create categorical block only if there are non-batch columns
        if len(non_batch_indexes) > 0:
            self._c_block = [
                self._batch_categorical(
                    categorical_data, indexes=non_batch_indexes
                )
            ]

        print("numerical block encoding")
        # Handle numerical data
        if numerical_data is not None:
            _data = numerical_data.copy()
            if isinstance(numerical_data, pd.DataFrame):
                _data = _data.values
            self._n_block = [_data]

    def generate(self):
        """
        Combine all blocks into a single design matrix. Include batch
        intercepts, categorical variables, and numerical variables
        """
        blocks = []

        # Always include batch intercepts (required)
        if hasattr(self, "_b_block"):
            blocks.append(self._b_block)

        # Include non-batch categorical variables if present
        if hasattr(self, "_c_block"):
            blocks.extend(self._c_block)

        # Include numerical variables if present
        if hasattr(self, "_n_block"):
            blocks.extend(self._n_block)

        if not blocks:
            raise ValueError("No data provided to generate design matrix")

        return np.hstack(blocks)

    def _batch_categorical(self, categorical_data, indexes=None):
        """
        Convert categorical data into a big one-hot matrix

        Parameters
        ----------
        categorical_data : pd.DataFrame, np.array, list[list]
            The input categorical data.
        indexes : list[str], optional
            The indexes of columns to include.

        Returns
        -------
        Y : array-like, shape (n_samples, N)
            The one-hot encoded matrix. N is the sum of the
            number of classes for all categorical features.
        """
        if indexes is None:
            indexes = categorical_data.columns.tolist()

        _col = []
        for ix in indexes:
            _enc = np.unique(categorical_data[ix], return_inverse=True)[-1]
            print(f"encoding for {ix} = {categorical_data[ix]} is\n    {_enc}")
            _col.append(_categorical(_enc, len(np.unique(_enc))))

        return np.hstack(_col)


def _categorical(cat, n_class=None):
    """
    Encode categorical vector into on-hot matrix
        Parameters
    ----------
    cat : array-like, shape (n_samples,)
        The input categorical vector.
    n_class : int, optional
        The number of classes. If not provided, it
        will be inferred from the input.
        Returns
    -------
    Y : array-like, shape (n_samples, n_class)
        The one-hot encoded matrix.
    """

    Y = np.zeros((len(cat), n_class or np.max(cat) + 1))
    Y[np.arange(len(cat)), cat] = 1.0
    return Y
