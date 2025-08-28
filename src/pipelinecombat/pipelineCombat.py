import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from pgmpy.models import LinearGaussianBayesianNetwork, DiscreteBayesianNetwork
from pgmpy.estimators import ExpectationMaximization
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.factors.discrete import DiscreteFactor

from pipelinecombat.model.design import DesignMatrix


def pipeline_combat(
    biased_data,
    covariates,
    batch_col_index=0,
    modality_col_index=1,
    numerical_col_indexes=None,
    create_pca_block=True,
    pca_n_components=None,
    batch_links=None
):
    """
    Run pipeline combat.

    Parameters
    ----------
    biased_data : np.ndarray
        The input biased data to correct.
        Shape : (n_samples, n_features)
    covariates : pd.DataFrame
        The input covariates to preserve. Also includes columns
        with batch and modality information Shape : (n_samples, n_categories)
    batch_col_index : int, string
        The index of the batch column in the covariates.
    modality_col_index : int, string
        The index of the modality column in the covariates.
    numerical_col_indexes : list[int], None
        The indexes of the numerical columns in the covariates.
    create_pca_block : bool
        Whether to create a PCA block from the
        biased data in the design matrix.
    pca_n_components : int
        The number of components to keep if PCA is applied.

    Returns
    -------
    design_matrices : list
        Design matrices for all modalities
    models : list
        Model parameters beta for all voxels
    additive : list
        Additive pipeline parameters gamma star for all batches
    multiplicative : list
        Multiplicative pipeline parameters delta star for all batches
    """
    if isinstance(modality_col_index, int):
        modality_col_index = covariates.columns[modality_col_index].to_list()
    elif isinstance(modality_col_index, str):
        modality_col_index = [modality_col_index]

    if numerical_col_indexes is None:
        numerical_col_indexes = []

    if all(isinstance(n, int) for n in numerical_col_indexes):
        numerical_col_indexes = covariates.columns[numerical_col_indexes].to_list()

    categorical_col_indexes = covariates.columns.difference(
        numerical_col_indexes + modality_col_index
    )
    print(f"cov cols {categorical_col_indexes} | {numerical_col_indexes + modality_col_index}")

    # Collect unique modalities, we build estimates from them
    _mods = np.unique(covariates[modality_col_index])
    models = []
    designs = []
    for _mod in _mods:
        _samples = np.where(covariates[modality_col_index] == _mod)[0]

        # Design matrix for the current modality
        design_matrix = generate_design_matrix_j(
            covariates[categorical_col_indexes].iloc[_samples],
            covariates[numerical_col_indexes].iloc[_samples],
            batch_col_index=batch_col_index,
            create_pca_block=create_pca_block,
            pca_n_components=pca_n_components,
            numerical_for_pca=biased_data[_samples]
        )
        _dm = design_matrix.generate()
        designs.append(_dm)

        # Model fits in the current v element
        _models = []
        for _data in biased_data[_samples].T:
            beta, noise_var = estimate_model_v(_data, _dm)
            _models.append({"beta": beta, "noise_variance": noise_var})

        models.append(_models)

    batch_per_sample = covariates[batch_col_index].to_numpy().flatten()
    modality_per_sample = covariates[modality_col_index].to_numpy().flatten()

    # Standardize data for each modality
    standard_data = standardize(
        biased_data,
        designs,
        models,
        modality_per_sample,
        len(np.unique(batch_per_sample))
    )

    # Estimate location/scale parameters
    (
        gamma_hat, gamma_bar, gamma_var_bar,
        delta_hat_var, lambda_bar, theta_bar
    ) = estimate_ls_parameters(standard_data, batch_per_sample, batch_links)

    # Optimize parameters using EM Bayes
    gamma_star, delta_var_star = empirical_bayes_optimizer(
        standard_data,
        gamma_hat, gamma_bar, gamma_var_bar,
        delta_hat_var, lambda_bar, theta_bar,
        batch_per_sample
    )

    return designs, models, gamma_star, delta_var_star


def generate_design_matrix_j(
    categorical_data,
    numerical_data,
    batch_col_index=0,
    create_pca_block=True,
    pca_n_components=None,
    numerical_for_pca=None
):
    """
    Generate the Design Matrix for a measurement `j`. Has the ability to
    generate a PCA block from the numerical data to replace it, if requested.
    In this case, the numerical data provided can have an extra dimension at
    the end for multiple observations each variable, in which case it will be
    used to boost the samples.

    Parameters
    ----------
    categorical_data : pd.DataFrame, np.array, list[list], required
        The input categorical data. At least one column must describe
        the batch of the samples. Shape: (n_samples, n_variable)
    numerical_data : pd.DataFrame, np.array, list[list], optional
        The input numerical data. Shape (n_samples, n_features).
    batch_col_index : any, optional
        The index of the batch column in the categorical data.
    create_pca_block : bool, optional
        Whether to create a PCA block from the numerical data.
    pca_n_components : int, optional
        The number of components to keep if PCA is applied.
    numerical_for_pca : np.array, optional
        The numerical data to use for PCA. If None,
        the original numerical_data is used.

    Returns
    -------
    design_matrix : np.array
        The generated design matrix. Shape: (n_samples, K).
        K is the total number of features after encoding, from
        all categorical and numerical data, plus batch intercept.
    """
    if create_pca_block:
        _data = numerical_data
        if numerical_for_pca is not None:
            _data = numerical_for_pca

        pca = PCA(n_components=pca_n_components or len(categorical_data.index))
        pca.fit(_data.T)

        # Replace the original numerical data with the PCA system matrix
        numerical_data = pca.components_.T

    return DesignMatrix(categorical_data, numerical_data, batch_col_index)


def estimate_model_v(sample, design_matrix):
    """
    Estimate model parameters beta given a system matrix. Also
    estimate the variance of the noise distribution.

    Parameters:
    -----------
    sample : np.array
        The sample data in v to use for estimating model parameters.
    design_matrix : np.array
        The design matrix to use for estimating model parameters.

    Returns
    -------
    beta : np.array
        The estimated beta parameters.
    noise_variance : float
        The estimated variance of the noise distribution.
    """

    # Estimate beta using OLS
    beta, _, _, _ = np.linalg.lstsq(design_matrix, sample, rcond=None)
    noise_variance = np.var(sample - design_matrix @ beta)

    return beta, noise_variance


def bayes_network_optimizer(standard_data, batch_per_sample, batch_links):
    """
    Parameters
    ----------
    standard_data : np.array
        The standardized data to use for estimating parameters.
        Shape : (n_samples, n_features)
    batch_per_sample : np.ndarray
        The batch index for each sample. Shape : (n_samples,)
    batch_links : np.ndarray | None, optional
        Links between batches. A 1 in row i, column j means batch j
        is parent to batch i. Shape : (n_batches, n_batches)

    Returns
    -------
    gamma_opt : np.array
        The optimized gamma parameters. Shape : (n_batches,)
    gamma_var_opt : np.array
        The optimized variance of the gamma parameters. Shape : (n_batches,)
    delta_opt : np.array
        The optimized delta parameters. Shape : (n_batches,)
    delta_var_opt : np.array
        The optimized variance of the delta parameters. Shape : (n_batches,)
    """
    batches, bcounts = np.unique(batch_per_sample, return_counts=True)
    n_batches = len(batches)

    def _optim(_df, _est=ExpectationMaximization):
        _opt = np.zeros((n_batches,))
        _opt_var = np.zeros((n_batches,))
        _bn = DiscreteBayesianNetwork()
        for b in range(n_batches):
            _bn.add_node(f"b{b}")

        for child, p_list in enumerate(batch_links):
            for parent in np.where(p_list)[0]:
                _bn.add_edge(f"b{parent}", f"b{child}")

        _bn.fit(_df)

        _ve = BeliefPropagation(_bn)
        for b in range(n_batches):
            _q = _ve.query(variables=[f"b{b}"])
            if isinstance(_q, DiscreteFactor):
                _opt[b] = np.mean(_q.values)
                _opt_var[b] = np.var(_q.values, ddof=1)

        #_sim = _bn.simulate(1)
        #for b in range(n_batches):
        #    _q = _sim.copy()
        #    del _q[f"b{b}"]
        #    _, _opt[b], _opt_var[b] = _bn.predict(_q)

        return _opt, _opt_var

    # Create pandas dataframe from gammas with nodes names as columns
    gamma_df = pd.DataFrame({f"b{b}": np.mean(standard_data[batch_per_sample == b], axis=0)
                             for b in np.unique(batch_per_sample)})
    print(gamma_df)
    gamma_opt, gamma_var_opt = _optim(gamma_df)
    print(f"Gamma : {gamma_opt} | var : {gamma_var_opt}")

    delta_estim = (standard_data - np.repeat(gamma_opt, bcounts)[:, None]) ** 2.
    delta_df = pd.DataFrame({f"b{b}": np.mean(delta_estim[batch_per_sample == b], axis=0)
                             for b in np.unique(batch_per_sample)})

    print(delta_df)
    delta_opt, delta_var_opt = _optim(delta_df)
    print(f"Delta : {delta_opt} | var : {delta_var_opt}")

    return gamma_opt, gamma_var_opt, delta_opt, delta_var_opt


def estimate_ls_parameters(standard_data, batch_per_sample, batch_links=None):
    """
    Estimate location/scale model parameters using the provided models.
    For now, uses parametric implementation stating that :

        gamma_iv   ~ N(gamma_i, tau_i^2)
        delta_iv^2 ~ InvGamma(lambda_i, theta_i)

    Parameters
    ----------
    standard_data : np.array
        The standardized data to use for estimating parameters.
        Shape : (n_samples, n_features)
    batch_per_sample : np.ndarray
        The batch index for each sample. Shape : (n_samples,)
    batch_links : np.ndarray | None, optional
        Links between batches. A 1 in row i, column j means batch j
        is parent to batch i. Shape : (n_batches, n_batches)

    Returns
    -------
    gamma_hat : np.array
        The estimated gamma parameters for each batch.
        Shape : (n_batches, n_features)
    gamma_bar : np.array
        The estimated gamma parameters. Shape : (n_batches,)
    gamma_var_bar : np.array
        The estimated variance of the gamma parameters.
        Shape : (n_batches,)
    delta_var_hat : np.array
        The estimated variance of the delta parameters.
        Shape : (n_batches, n_features)
    lambda_bar : np.array
        The estimated lambda parameters. Shape : (n_batches,)
    theta_bar : np.array
        The estimated theta parameters. Shape : (n_batches,)
    """
    _batches = np.unique(batch_per_sample)
    n_batches = len(_batches)
    _, n_features = standard_data.shape

    # Compute gamma_hat: average standardized data per batch
    gamma_hat = np.zeros((n_batches, n_features))
    for i, batch_id in enumerate(_batches):
        batch_mask = batch_per_sample == batch_id
        gamma_hat[i] = np.mean(standard_data[batch_mask], axis=0)

    # Compute delta_hat_var: variance of standardized data per batch
    delta_hat_var = np.zeros((n_batches, n_features))
    for i, batch_id in enumerate(_batches):
        batch_mask = batch_per_sample == batch_id
        batch_data = standard_data[batch_mask]
        delta_hat_var[i] = np.var(
            batch_data,
            mean=gamma_hat[i, None, :],
            axis=0,
            ddof=1
        )
    if batch_links is not None:
        print("Applying Bayesian network optimization with sequential"
              " dependencies...")
        (
            gamma_bar,
            gamma_var_bar,
            lambda_bar,
            theta_bar
        ) = bayes_network_optimizer(
            standard_data, batch_per_sample, batch_links
        )
    else:
        # Compute gamma distribution parameters (across features)
        gamma_bar = np.mean(gamma_hat, axis=1)  # (n_batches,)
        gamma_var_bar = np.var(
            gamma_hat, mean=gamma_bar[:, None], axis=1, ddof=1
        )  # (n_batches,)

        # Compute delta distribution parameters (across features)
        average_vox = np.mean(delta_hat_var, axis=1)  # (n_batches,)
        variance_vox = np.var(
            delta_hat_var,
            mean=average_vox[:, None],
            axis=1,
            ddof=1
        )  # (n_batches,)
        zero_var = variance_vox == 0

        # Inverse gamma parameters (avoid division by zero)
        lambda_bar = np.zeros(n_batches)
        theta_bar = np.zeros(n_batches)
        avg_on_var = average_vox[~zero_var] / variance_vox[~zero_var]

        lambda_bar[~zero_var] = avg_on_var + 2.
        theta_bar[~zero_var] = average_vox[~zero_var] * (
            average_vox[~zero_var] * avg_on_var + 1.
        )

    return (
        gamma_hat,
        gamma_bar,
        gamma_var_bar,
        delta_hat_var,
        lambda_bar,
        theta_bar
    )


def standardize(biased_data, designs, models, modality_per_sample, n_batches):
    """
    Standardize the biased data for each modality using the
    design matrices and model parameters.

    Parameters
    ----------
    biased_data : np.array
        The biased data to standardize.
        Shape : (n_samples, n_features)
    designs : list[DesignMatrix]
        The list of design matrices for each modality.
        Shape : (n_modalities,)
    models : list[dict]
        The list of models for each feature in each modality.
        Shape : (n_modalities, n_features_per_modality)
    modality_per_sample : np.array
        The modality index for each sample.
        Shape : (n_samples,)

    Returns
    -------
    np.array
        The standardized data.
        Shape : (n_samples, n_features)
    """

    standard_data = np.zeros_like(biased_data)
    for _mod_ix in np.unique(modality_per_sample):
        # Get samples for this modality
        mod_mask = modality_per_sample == _mod_ix
        mod_data = biased_data[mod_mask]  # (n_samples_mod, n_features)

        # Get model parameters for this modality
        _betas = np.asarray(
            [m["beta"] for m in models[_mod_ix]]
        )  # (n_features, n_coef)
        _vars = np.asarray(
            [m["noise_variance"] for m in models[_mod_ix]]
        )  # (n_features,)

        # Compute residuals: data - design @ beta
        fit = designs[_mod_ix] @ _betas.T  # (n_samples_mod, n_features)
        residuals = (mod_data - fit)  # (n_samples_mod, n_features)

        # Standardize by noise standard deviation (avoid division by zero)
        _zero_var = np.isclose(_vars, 0)
        if np.any(~_zero_var):
            residuals[..., ~_zero_var] /= np.sqrt(_vars[~_zero_var])

        standard_data[mod_mask] = residuals

    return standard_data


def empirical_bayes_optimizer(
    standard_data,
    gamma_hat,
    gamma_bar,
    gamma_var_bar,
    delta_hat_var,
    lambda_bar,
    theta_bar,
    batch_per_sample,
    niter=30
):
    """
    Perform empirical bayes optimization on prior parameters.

    Parameters
    ----------
    standard_data : np.array
        Standardized data for each sample.
        Shape : (n_samples, n_features)
    gamma_hat : np.array
        Initial estimate of the gamma parameters.
        Shape : (n_batches, n_features)
    gamma_bar : np.array
        Prior mean of the gamma parameters.
        Shape : (n_batches,)
    gamma_var_bar : np.array
        Prior variance of the gamma parameters.
        Shape : (n_batches,)
    delta_hat_var : np.array
        Initial estimate of the delta variance parameters.
        Shape : (n_batches, n_features)
    lambda_bar : np.array
        Prior mean of the lambda parameters.
        Shape : (n_batches,)
    theta_bar : np.array
        Prior mean of the theta parameters.
        Shape : (n_batches,)
    batch_per_sample : np.array
        Batch assignment for each sample.
        Shape : (n_samples,)
    niter : int
        Number of iterations for optimization.

    Returns
    -------
    gamma_star : np.array
        Optimized gamma parameters.
        Shape : (n_batches, n_features)
    delta_var_star : np.array
        Optimized delta variance parameters.
        Shape : (n_batches, n_features)
    """
    convergence = False
    i = 0
    batches, bcounts = np.unique(batch_per_sample, return_counts=True)

    gamma_star = np.zeros_like(gamma_hat)  # Shape: (n_batches, n_features)
    delta_var_star = delta_hat_var.copy()  # Shape: (n_batches, n_features)
    
    while not convergence:
        i += 1
        for batch_idx, (_batch, _bcount) in enumerate(zip(batches, bcounts)):
            _bmask = batch_per_sample == _batch
            
            # Update gamma_star for this batch
            gamma_star[batch_idx] = (
                _bcount * gamma_var_bar[batch_idx] * gamma_hat[batch_idx] +
                delta_var_star[batch_idx] * gamma_bar[batch_idx]
            ) / (
                _bcount * gamma_var_bar[batch_idx] + delta_var_star[batch_idx]
            )

            # Update delta_var_star for this batch
            batch_residuals = standard_data[_bmask] - gamma_star[batch_idx]
            residual_sum = 0.5 * np.sum(batch_residuals ** 2, axis=0)
            delta_var_star[batch_idx] = (
                theta_bar[batch_idx] + residual_sum
            ) / (
                _bcount / 2. + lambda_bar[batch_idx] - 1.
            )

        convergence = i == niter

    return gamma_star, delta_var_star
