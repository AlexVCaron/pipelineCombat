# %%matplotlib tk
# %%
# #!/usr/bin/env python3
"""
Demo of the pipeline_combat method using simulated DWI data.

This demo simulates:
- DWI data from a 10x10 voxel patch
- 13 subjects for statistical relevance
- 20 gradient directions (modalities)
- 4 sequential pipeline processing steps (batches) with decreasing noise levels
"""

import numpy as np
import pandas as pd

from pipelinecombat.pipelineCombat import pipeline_combat


# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib
    import seaborn as sns

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Matplotlib/Seaborn not available. Running without visualizations.")
    print("To enable visualizations, install with: uv add matplotlib seaborn")


def simulate_dwi_data(
    n_subjects=13,
    n_voxels=100,
    n_directions=20,
    n_pipeline_steps=4,
    min_add=0.0,
    max_add=0.7,
    min_mult=0.0,
    max_mult=1.0,
):
    """
    Simulate DWI data with pipeline-induced biases.

    Parameters
    ----------
    n_subjects : int
        Number of subjects
    n_voxels : int
        Number of voxels (10x10 = 100)
    n_directions : int
        Number of gradient directions
    n_pipeline_steps : int
        Number of sequential pipeline processing steps

    Returns
    -------
    biased_data : np.ndarray
        Simulated biased data (n_samples, n_voxels)
    covariates : pd.DataFrame
        Covariates including batch and modality information
    true_data : np.ndarray
        Ground truth data without pipeline biases
    """
    np.random.seed(42)  # For reproducibility

    # Generate ground truth DWI signal for each voxel
    # Simulate realistic DWI values (0.5 to 0.7 range)
    true_signal_base = 1000.0 * np.repeat(
        np.random.uniform(0.5, 0.7, n_subjects)[:, None], n_voxels, axis=1
    )

    # Create direction-dependent signal variation
    # Different directions show different signal intensities
    direction_effects = np.random.uniform(0.01, 1.0, n_directions)

    # Initialize arrays
    biased_data = []
    true_data = []
    batch_labels = []
    modality_labels = []
    subject_ids = []

    # Pipeline step noise levels (decreasing with processing depth)
    noise_variance = 0.005 * 0.01 * np.mean(true_signal_base)

    # Pipeline step systematic biases
    additive_biases = 1000.0 * np.linspace(
        min_add, max_add, n_pipeline_steps
    )  # Systematic shifts
    multiplicative_biases = np.linspace(
        min_mult, max_mult, n_pipeline_steps
    )  # Scale changes

    for subject in range(n_subjects):
        for direction in range(n_directions):
            # Base signal for this subject and direction
            base_signal = true_signal_base[subject] * np.exp(
                -direction_effects[direction]
            )

            noise = 1000.0 * np.random.normal(0, noise_variance, n_voxels)

            input_signal = base_signal
            biased_signals = []
            for step in range(n_pipeline_steps):
                # _add = np.random.normal(additive_biases[step], 0.01, n_voxels)
                # _mult = np.random.normal(
                #    multiplicative_biases[step], 0.01, n_voxels
                # )
                # Apply pipeline-specific biases
                # Additive bias (systematic shift)
                biased_signal = (
                    input_signal
                    + multiplicative_biases[step] * noise
                    + additive_biases[step]
                )
                # input_signal = biased_signal

                # Store data
                biased_signals.append(biased_signal)
                batch_labels.append(step)  # Numerical batch ID
                modality_labels.append(direction)  # Numerical modality ID
                subject_ids.append(subject)  # Numerical subject ID

            biased_data.extend(biased_signals[::-1])
            true_data.append(base_signal)

    # Convert to arrays
    biased_data = np.array(biased_data)
    print(biased_data.shape)
    true_data = np.array(true_data)
    print(true_data.shape)

    # Create covariates DataFrame
    covariates = pd.DataFrame(
        {
            "batch": batch_labels,
            "modality": modality_labels,
            "subject": subject_ids,
        }
    )

    print(covariates)

    print(f"Generated data shape: {biased_data.shape}")
    print(f"Unique batches: {sorted(covariates['batch'].unique())}")
    print(
        f"Unique modalities: {len(covariates['modality'].unique())} directions"
    )
    print(f"Subjects: {len(covariates['subject'].unique())}")

    return biased_data, covariates, true_data


def visualize_pipeline_effects(biased_data, covariates, true_data=None):
    """Visualize pipeline-induced biases before correction."""
    if not VISUALIZATION_AVAILABLE:
        print("Visualization skipped - matplotlib/seaborn not available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Pipeline-Induced Biases (Before Correction)", fontsize=16)

    # Mean signal by pipeline step
    batch_means = []
    batch_names = []
    for batch in sorted(covariates["batch"].unique()):
        mask = covariates["batch"] == batch
        batch_means.append(biased_data[mask].mean())
        batch_names.append(f"Step {batch+1}")

    axes[0, 0].bar(range(len(batch_means)), batch_means)
    axes[0, 0].set_title("Mean Signal by Pipeline Step")
    axes[0, 0].set_xlabel("Pipeline Step")
    axes[0, 0].set_ylabel("Mean Signal")
    axes[0, 0].set_xticks(range(len(batch_means)))
    axes[0, 0].set_xticklabels(batch_names)

    # Signal variance by pipeline step
    batch_vars = []
    for batch in sorted(covariates["batch"].unique()):
        mask = covariates["batch"] == batch
        batch_vars.append(biased_data[mask].var())

    axes[0, 1].bar(range(len(batch_vars)), batch_vars)
    axes[0, 1].set_title("Signal Variance by Pipeline Step")
    axes[0, 1].set_xlabel("Pipeline Step")
    axes[0, 1].set_ylabel("Variance")
    axes[0, 1].set_xticks(range(len(batch_vars)))
    axes[0, 1].set_xticklabels(batch_names)

    # Heatmap of mean signal across first 10 voxels by batch
    batch_voxel_means = []
    for batch in sorted(covariates["batch"].unique()):
        mask = covariates["batch"] == batch
        batch_voxel_means.append(biased_data[mask, :10].mean(axis=0))

    sns.heatmap(
        batch_voxel_means,
        xticklabels=[f"Voxel {i+1}" for i in range(10)],
        yticklabels=batch_names,
        ax=axes[1, 0],
        cmap="viridis",
    )
    axes[1, 0].set_title("Mean Signal Heatmap (First 10 Voxels)")

    # Distribution of signals by pipeline step
    for i, batch in enumerate(sorted(covariates["batch"].unique())):
        mask = covariates["batch"] == batch
        print(biased_data[mask].flatten().shape)
        axes[1, 1].hist(
            biased_data[mask].flatten(),
            alpha=0.6,
            label=f"Step {batch+1}",
            bins=30,
        )

    # Add true signal distribution if available
    if true_data is not None:
        print(true_data.flatten().shape)
        axes[1, 1].hist(
            true_data.flatten(),
            alpha=0.8,
            label="True Signal",
            bins=30,
            color="black",
            linestyle="--",
            histtype="step",
            linewidth=2,
        )

    axes[1, 1].set_title("Signal Distribution: Pipeline Steps vs True Signal")
    axes[1, 1].set_xlabel("Signal Value")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


def visualize_correction_results(gamma_star, delta_var_star, covariates):
    """Visualize the correction parameters from pipeline_combat."""
    if not VISUALIZATION_AVAILABLE:
        print("Visualization skipped - matplotlib/seaborn not available")
        return

    n_batches = len(gamma_star)

    fig, axes = plt.subplots(2, 3, figsize=(18, 15))
    fig.suptitle("Pipeline Combat Correction Parameters", fontsize=16)

    # Gamma (additive) parameters
    gamma_array = np.array(gamma_star).T  # Shape: (n_voxels, n_batches)

    # Heatmap of gamma parameters
    sns.heatmap(
        gamma_array[:20],  # Show first 20 voxels
        xticklabels=[f"Step {i+1}" for i in range(n_batches)],
        yticklabels=[f"Voxel {i+1}" for i in range(20)],
        ax=axes[0, 0],
        cmap="RdBu_r",
        center=0,
    )
    axes[0, 0].set_title("Gamma (Additive) Parameters")

    # Mean gamma by batch with error bars
    gamma_means = [np.mean(gamma) for gamma in gamma_star]
    gamma_stds = [np.std(gamma) for gamma in gamma_star]
    bars = axes[0, 1].bar(
        range(n_batches),
        gamma_means,
        color="skyblue",
        yerr=gamma_stds,
        capsize=5,
        alpha=0.8,
    )
    axes[0, 1].set_title(
        "Mean Gamma by Pipeline Step\n(with Standard Deviation)"
    )
    axes[0, 1].set_xlabel("Pipeline Step")
    axes[0, 1].set_ylabel("Mean Gamma")
    axes[0, 1].set_xticks(range(n_batches))
    axes[0, 1].set_xticklabels([f"Step {i+1}" for i in range(n_batches)])
    axes[0, 1].axhline(y=0, color="red", linestyle="--", alpha=0.7)
    axes[0, 1].grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, gamma_means, gamma_stds)):
        height = bar.get_height()
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + 0.01 * abs(height),
            f"{mean:.3f}±{std:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Gamma distribution with enhanced visualization
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    ]  # Default matplotlib colors
    for i, gamma in enumerate(gamma_star):
        color = colors[i % len(colors)]
        # Handle edge cases for histogram bins
        data_range = np.max(gamma) - np.min(gamma)
        if data_range < 1e-10:  # All values essentially the same
            # For constant values, show as vertical line
            axes[0, 2].axvline(
                x=np.mean(gamma),
                color=color,
                linewidth=3,
                alpha=0.8,
                label=f"Step {i+1} (μ={np.mean(gamma):.4f})",
            )
        else:
            # Use adaptive number of bins based on data range
            n_bins = min(30, max(5, int(len(gamma) / 5)))
            n, bins, patches = axes[0, 2].hist(
                gamma,
                alpha=0.7,
                label=f"Step {i+1} (μ={np.mean(gamma):.4f}, σ={np.std(gamma):.4f})",
                bins=n_bins,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )

    axes[0, 2].set_title(
        "Gamma Parameter Distributions\n(Additive Corrections)"
    )
    axes[0, 2].set_xlabel("Gamma Value")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].axvline(
        x=0,
        color="red",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
        label="Zero line",
    )
    axes[0, 2].grid(True, alpha=0.3)

    # Delta (multiplicative) parameters
    delta_array = np.array(delta_var_star).T  # Shape: (n_voxels, n_batches)

    # Heatmap of delta parameters
    sns.heatmap(
        delta_array[:20],  # Show first 20 voxels
        xticklabels=[f"Step {i+1}" for i in range(n_batches)],
        yticklabels=[f"Voxel {i+1}" for i in range(20)],
        ax=axes[1, 0],
        cmap="plasma",
    )
    axes[1, 0].set_title("Delta (Multiplicative Variance) Parameters")

    # Mean delta by batch with error bars
    delta_means = [np.mean(delta) for delta in delta_var_star]
    delta_stds = [np.std(delta) for delta in delta_var_star]
    bars = axes[1, 1].bar(
        range(n_batches),
        delta_means,
        color="lightcoral",
        yerr=delta_stds,
        capsize=5,
        alpha=0.8,
    )
    axes[1, 1].set_title(
        "Mean Delta by Pipeline Step\n(with Standard Deviation)"
    )
    axes[1, 1].set_xlabel("Pipeline Step")
    axes[1, 1].set_ylabel("Mean Delta")
    axes[1, 1].set_xticks(range(n_batches))
    axes[1, 1].set_xticklabels([f"Step {i+1}" for i in range(n_batches)])
    axes[1, 1].axhline(
        y=1, color="red", linestyle="--", alpha=0.7, label="Unity"
    )
    axes[1, 1].grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, delta_means, delta_stds)):
        height = bar.get_height()
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + 0.05 * abs(height),
            f"{mean:.3f}±{std:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Delta distribution with enhanced visualization
    for i, delta in enumerate(delta_var_star):
        color = colors[i % len(colors)]
        # Handle edge cases for histogram bins
        data_range = np.max(delta) - np.min(delta)
        if data_range < 1e-10:  # All values essentially the same
            # For constant values, show as vertical line
            axes[1, 2].axvline(
                x=np.mean(delta),
                color=color,
                linewidth=3,
                alpha=0.8,
                label=f"Step {i+1} (μ={np.mean(delta):.4f})",
            )
        else:
            # Use adaptive number of bins based on data range
            n_bins = min(30, max(5, int(len(delta) / 5)))
            n, bins, patches = axes[1, 2].hist(
                delta,
                alpha=0.7,
                label=f"Step {i+1} (μ={np.mean(delta):.4f}, σ={np.std(delta):.4f})",
                bins=n_bins,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )

    axes[1, 2].set_title(
        "Delta Parameter Distributions\n(Variance Corrections)"
    )
    axes[1, 2].set_xlabel("Delta Value")
    axes[1, 2].set_ylabel("Frequency")
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].axvline(
        x=1,
        color="red",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
        label="Unity line",
    )
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def reconstruct_fitted_data(
    biased_data, designs, models, covariates, modality_col_index="modality"
):
    """
    Reconstruct fitted data from design matrices and model parameters.

    Parameters
    ----------
    biased_data : np.ndarray
        Original biased data (n_samples, n_voxels)
    designs : list
        Design matrices for each modality
    models : list
        Model parameters for each modality
    covariates : pd.DataFrame
        Covariates including modality information
    modality_col_index : str
        Column name for modality information

    Returns
    -------
    fitted_data : np.ndarray
        Reconstructed fitted data (n_samples, n_voxels)
    """
    fitted_data = np.zeros_like(biased_data)
    modality_per_sample = covariates[modality_col_index].to_numpy()

    for _mod_ix in np.unique(modality_per_sample):
        # Get samples for this modality
        mod_mask = modality_per_sample == _mod_ix

        # Get model parameters for this modality
        _betas = np.asarray(
            [m["beta"] for m in models[_mod_ix]]
        )  # (n_voxels, n_coef)

        # Reconstruct fitted values: design @ beta
        fitted_values = (
            designs[_mod_ix] @ _betas.T
        )  # (n_samples_mod, n_voxels)
        fitted_data[mod_mask] = fitted_values

    return fitted_data


def visualize_model_goodness_of_fit(
    biased_data,
    designs,
    models,
    covariates,
    true_data,
    modality_col_index="modality",
):
    """
    Visualize goodness of fit of the models against true data.

    Parameters
    ----------
    biased_data : np.ndarray
        Original biased data
    designs : list
        Design matrices for each modality
    models : list
        Model parameters for each modality
    covariates : pd.DataFrame
        Covariates including modality information
    true_data : np.ndarray
        Ground truth data without biases
    modality_col_index : str
        Column name for modality information
    """
    if not VISUALIZATION_AVAILABLE:
        print("Visualization skipped - matplotlib/seaborn not available")
        return

    # Reconstruct fitted data from models
    fitted_data = reconstruct_fitted_data(
        biased_data, designs, models, covariates, modality_col_index
    )

    # Flatten true data to match fitted data structure
    # true_data has shape (n_subjects * n_directions, n_voxels)
    # We need to expand it to match the biased_data structure
    modality_per_sample = covariates[modality_col_index].to_numpy()
    subject_per_sample = covariates["subject"].to_numpy()

    # Create expanded true data to match sample structure
    expanded_true_data = np.zeros_like(fitted_data)
    for i, (subject, modality) in enumerate(
        zip(subject_per_sample, modality_per_sample)
    ):
        # Find corresponding true data index
        true_idx = subject * len(np.unique(modality_per_sample)) + modality
        if true_idx < len(true_data):
            expanded_true_data[i] = true_data[true_idx]

    # Calculate goodness of fit metrics
    n_voxels = fitted_data.shape[1]
    fitted_data.shape[0]

    # R-squared for each voxel
    r_squared_voxels = []
    mse_voxels = []

    for v in range(min(20, n_voxels)):  # Analyze first 20 voxels
        true_vals = expanded_true_data[:, v]
        fitted_vals = fitted_data[:, v]

        # R-squared
        ss_res = np.sum((true_vals - fitted_vals) ** 2)
        ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r_squared_voxels.append(r2)

        # MSE
        mse = np.mean((true_vals - fitted_vals) ** 2)
        mse_voxels.append(mse)

    # Get batch information for distinguishing batches
    batch_per_sample = covariates["batch"].to_numpy()
    unique_batches = np.unique(batch_per_sample)

    # Color scheme for batches
    batch_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    ]
    batch_markers = ["o", "s", "^", "D", "v", "<"]

    # Calculate overall correlation
    true_flat = expanded_true_data.flatten()
    fitted_flat = fitted_data.flatten()
    biased_flat = biased_data.flatten()

    # Flatten batch information to match data structure
    batch_flat = np.repeat(batch_per_sample, fitted_data.shape[1])

    # Remove any invalid values
    valid_mask = (
        np.isfinite(true_flat)
        & np.isfinite(fitted_flat)
        & np.isfinite(biased_flat)
    )
    true_flat = true_flat[valid_mask]
    fitted_flat = fitted_flat[valid_mask]
    biased_flat = biased_flat[valid_mask]
    batch_flat = batch_flat[valid_mask]

    correlation_fitted = np.corrcoef(true_flat, fitted_flat)[0, 1]
    correlation_biased = np.corrcoef(true_flat, biased_flat)[0, 1]

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Model Goodness of Fit Analysis (Batch-Colored)", fontsize=16)

    # 1. Scatter plot: True vs Fitted (colored by batch)
    sample_size = min(5000, len(true_flat))  # Sample for performance
    sample_idx = np.random.choice(len(true_flat), sample_size, replace=False)

    # Plot fitted vs true by batch
    for i, batch in enumerate(unique_batches):
        batch_mask = batch_flat[sample_idx] == batch
        if np.any(batch_mask):
            axes[0, 0].scatter(
                true_flat[sample_idx][batch_mask],
                fitted_flat[sample_idx][batch_mask],
                alpha=0.6,
                s=15,
                color=batch_colors[i % len(batch_colors)],
                marker=batch_markers[i % len(batch_markers)],
                label=f"{batch+1}",
                edgecolors="black",
                linewidth=0.3,
            )

    # Add diagonal line
    min_val = min(
        np.min(true_flat[sample_idx]), np.min(fitted_flat[sample_idx])
    )
    max_val = max(
        np.max(true_flat[sample_idx]), np.max(fitted_flat[sample_idx])
    )
    axes[0, 0].plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        alpha=0.8,
        label="Perfect fit",
        linewidth=2,
    )

    axes[0, 0].set_xlabel("True Values")
    axes[0, 0].set_ylabel("Fitted Values")
    axes[0, 0].set_title(
        f"True vs Fitted by Batch\n(Overall r={correlation_fitted:.3f})"
    )
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. R-squared by voxel
    voxel_indices = list(range(len(r_squared_voxels)))
    bars = axes[0, 1].bar(
        voxel_indices, r_squared_voxels, color="lightgreen", alpha=0.8
    )
    axes[0, 1].set_xlabel("Voxel Index")
    axes[0, 1].set_ylabel("R-squared")
    axes[0, 1].set_title(
        f"R-squared by Voxel\n(Mean R² = {np.mean(r_squared_voxels):.3f})"
    )
    axes[0, 1].grid(True, alpha=0.3)

    # Add value labels on bars for better readability
    for i, (bar, r2) in enumerate(zip(bars, r_squared_voxels)):
        if i % 2 == 0:  # Show every other label to avoid crowding
            height = bar.get_height()
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{r2:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # 3. MSE by voxel
    bars = axes[0, 2].bar(
        voxel_indices, mse_voxels, color="lightcoral", alpha=0.8
    )
    axes[0, 2].set_xlabel("Voxel Index")
    axes[0, 2].set_ylabel("Mean Squared Error")
    axes[0, 2].set_title(
        f"MSE by Voxel\n(Mean MSE = {np.mean(mse_voxels):.4f})"
    )
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Residuals vs Fitted (colored by batch)
    residuals = true_flat - fitted_flat
    for i, batch in enumerate(unique_batches):
        batch_mask = batch_flat[sample_idx] == batch
        if np.any(batch_mask):
            axes[1, 0].scatter(
                fitted_flat[sample_idx][batch_mask],
                residuals[sample_idx][batch_mask],
                alpha=0.6,
                s=12,
                color=batch_colors[i % len(batch_colors)],
                marker=batch_markers[i % len(batch_markers)],
                label=f"{batch+1}",
                edgecolors="black",
                linewidth=0.2,
            )

    axes[1, 0].axhline(
        y=0, color="red", linestyle="--", alpha=0.8, linewidth=2
    )
    axes[1, 0].set_xlabel("Fitted Values")
    axes[1, 0].set_ylabel("Residuals (True - Fitted)")
    axes[1, 0].set_title("Residuals vs Fitted by Batch")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Residuals distribution by batch
    axes[1, 1].hist(
        residuals,
        bins=30,
        alpha=0.4,
        color="gray",
        edgecolor="black",
        label="Overall",
        density=True,
    )

    # Add individual batch distributions
    for i, batch in enumerate(unique_batches):
        batch_mask = batch_flat == batch
        if np.any(batch_mask):
            batch_residuals = residuals[batch_mask]
            axes[1, 1].hist(
                batch_residuals,
                bins=20,
                alpha=0.6,
                color=batch_colors[i % len(batch_colors)],
                label=f"{batch+1} (n={np.sum(batch_mask)})",
                density=True,
                histtype="step",
                linewidth=2,
            )

    axes[1, 1].axvline(
        x=0, color="red", linestyle="--", alpha=0.8, linewidth=2
    )
    axes[1, 1].set_xlabel("Residuals")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("Residuals Distribution by Batch")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Model performance by batch and modality
    # Calculate metrics by batch
    batch_metrics = {}
    for batch in unique_batches:
        batch_mask = batch_per_sample == batch
        if np.any(batch_mask):
            true_batch = expanded_true_data[batch_mask].flatten()
            fitted_batch = fitted_data[batch_mask].flatten()
            biased_batch = biased_data[batch_mask].flatten()

            # Remove invalid values
            valid_batch = (
                np.isfinite(true_batch)
                & np.isfinite(fitted_batch)
                & np.isfinite(biased_batch)
            )
            true_batch = true_batch[valid_batch]
            fitted_batch = fitted_batch[valid_batch]
            biased_batch = biased_batch[valid_batch]

            if len(true_batch) > 0:
                # R-squared for fitted
                ss_res_fitted = np.sum((true_batch - fitted_batch) ** 2)
                ss_tot = np.sum((true_batch - np.mean(true_batch)) ** 2)
                r2_fitted = 1 - (ss_res_fitted / ss_tot) if ss_tot > 0 else 0

                # R-squared for biased
                ss_res_biased = np.sum((true_batch - biased_batch) ** 2)
                r2_biased = 1 - (ss_res_biased / ss_tot) if ss_tot > 0 else 0

                # Correlations
                corr_fitted = np.corrcoef(true_batch, fitted_batch)[0, 1]
                corr_biased = np.corrcoef(true_batch, biased_batch)[0, 1]

                batch_metrics[batch] = {
                    "r2_fitted": r2_fitted,
                    "r2_biased": r2_biased,
                    "corr_fitted": corr_fitted,
                    "corr_biased": corr_biased,
                    "mse_fitted": np.mean((true_batch - fitted_batch) ** 2),
                    "mse_biased": np.mean((true_batch - biased_batch) ** 2),
                }

    # Plot batch comparison
    batch_names = [f"{b+1}" for b in unique_batches]
    r2_fitted_vals = [batch_metrics[b]["r2_fitted"] for b in unique_batches]
    r2_biased_vals = [batch_metrics[b]["r2_biased"] for b in unique_batches]

    x_pos = np.arange(len(batch_names))
    width = 0.35

    bars1 = axes[1, 2].bar(
        x_pos - width / 2,
        r2_fitted_vals,
        width,
        label="Fitted Model",
        alpha=0.8,
        color=[
            batch_colors[i % len(batch_colors)]
            for i in range(len(unique_batches))
        ],
    )
    bars2 = axes[1, 2].bar(
        x_pos + width / 2,
        r2_biased_vals,
        width,
        label="Biased Data",
        alpha=0.6,
        color=[
            batch_colors[i % len(batch_colors)]
            for i in range(len(unique_batches))
        ],
        hatch="//",
    )

    axes[1, 2].set_xlabel("Pipeline Batch")
    axes[1, 2].set_ylabel("R-squared")
    axes[1, 2].set_title("Model Performance by Batch")
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(batch_names)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1, 2].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("MODEL GOODNESS OF FIT SUMMARY")
    print("=" * 60)
    print(f"Overall Correlation (Fitted vs True): {correlation_fitted:.4f}")
    print(f"Overall Correlation (Biased vs True): {correlation_biased:.4f}")
    print(
        f"Improvement in Correlation: {correlation_fitted - correlation_biased:.4f}"
    )
    print(
        f"Mean R-squared across voxels: {np.mean(r_squared_voxels):.4f} ± {np.std(r_squared_voxels):.4f}"
    )
    print(
        f"Mean MSE across voxels: {np.mean(mse_voxels):.6f} ± {np.std(mse_voxels):.6f}"
    )
    print("Residuals statistics:")
    print(f"  Mean: {np.mean(residuals):.6f}")
    print(f"  Std:  {np.std(residuals):.6f}")
    print(f"  Min:  {np.min(residuals):.6f}")
    print(f"  Max:  {np.max(residuals):.6f}")

    # Print batch-specific statistics
    print("\n" + "-" * 60)
    print("BATCH-SPECIFIC PERFORMANCE")
    print("-" * 60)
    for batch in unique_batches:
        if batch in batch_metrics:
            metrics = batch_metrics[batch]
            print(f"Batch {batch+1}:")
            print(
                f"  Fitted Model - R²: {metrics['r2_fitted']:.4f}, "
                f"Correlation: {metrics['corr_fitted']:.4f}, "
                f"MSE: {metrics['mse_fitted']:.6f}"
            )
            print(
                f"  Biased Data  - R²: {metrics['r2_biased']:.4f}, "
                f"Correlation: {metrics['corr_biased']:.4f}, "
                f"MSE: {metrics['mse_biased']:.6f}"
            )
            improvement_r2 = metrics["r2_fitted"] - metrics["r2_biased"]
            improvement_corr = metrics["corr_fitted"] - metrics["corr_biased"]
            improvement_mse = (
                metrics["mse_biased"] - metrics["mse_fitted"]
            )  # Lower MSE is better
            print(
                f"  Improvement  - R²: {improvement_r2:+.4f}, "
                f"Correlation: {improvement_corr:+.4f}, "
                f"MSE: {improvement_mse:+.6f}"
            )
            print()


def visualize_gamma_parameters(gamma_star, covariates):
    """
    Simple visualization of gamma parameters with 6 subplots in 2x3 grid.
    
    Parameters
    ----------
    gamma_star : list
        List of gamma parameter arrays for each batch
    covariates : pd.DataFrame
        Covariates including batch information
    """
    if not VISUALIZATION_AVAILABLE:
        print("Visualization skipped - matplotlib/seaborn not available")
        return

    print("Creating gamma parameter analysis...")
    
    # Extract batch information
    unique_batches = sorted(covariates['batch'].unique())
    n_batches = len(unique_batches)
    batch_colors = plt.cm.Set3(np.linspace(0, 1, n_batches))
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Overall gamma distribution
    all_gamma_values = np.concatenate(gamma_star)
    axes[0, 0].hist(all_gamma_values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Zero correction')
    axes[0, 0].set_xlabel('Gamma Values')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Overall Gamma Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Gamma values by batch (boxplot)
    batch_gamma_data = []
    batch_labels = []
    for i, batch in enumerate(unique_batches):
        batch_gamma_data.append(gamma_star[i])
        batch_labels.append(f'{batch+1}')
    
    bp = axes[0, 1].boxplot(batch_gamma_data, labels=batch_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], batch_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[0, 1].axhline(0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    axes[0, 1].set_xlabel('Pipeline Batch')
    axes[0, 1].set_ylabel('Gamma Values')
    axes[0, 1].set_title('Gamma Distribution by Batch')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Batch statistics comparison
    batch_means = [np.mean(gamma) for gamma in gamma_star]
    batch_stds = [np.std(gamma) for gamma in gamma_star]
    
    x_pos = np.arange(len(unique_batches))
    width = 0.35
    
    bars1 = axes[0, 2].bar(x_pos - width/2, batch_means, width, 
                           label='Mean', alpha=0.8, color=batch_colors)
    bars2 = axes[0, 2].bar(x_pos + width/2, batch_stds, width, 
                           label='Std Dev', alpha=0.6, color=batch_colors, hatch='//')
    
    axes[0, 2].axhline(0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    axes[0, 2].set_xlabel('Pipeline Batch')
    axes[0, 2].set_ylabel('Gamma Statistics')
    axes[0, 2].set_title('Mean and Standard Deviation by Batch')
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(batch_labels)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Gamma range by batch
    gamma_ranges = [np.max(gamma) - np.min(gamma) for gamma in gamma_star]
    gamma_mins = [np.min(gamma) for gamma in gamma_star]
    gamma_maxs = [np.max(gamma) for gamma in gamma_star]
    
    axes[1, 0].bar(batch_labels, gamma_ranges, alpha=0.7, color=batch_colors)
    axes[1, 0].set_xlabel('Pipeline Batch')
    axes[1, 0].set_ylabel('Gamma Range')
    axes[1, 0].set_title('Gamma Parameter Range by Batch')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Cumulative gamma effects
    cumulative_gamma = np.zeros_like(gamma_star[0])
    cumulative_data = []
    cumulative_labels = []
    
    for i, gamma in enumerate(gamma_star):
        cumulative_gamma += gamma
        cumulative_data.append(cumulative_gamma.copy())
        cumulative_labels.append(f'Up to Batch {unique_batches[i]+1}')
    
    for i, (cum_gamma, label, color) in enumerate(zip(cumulative_data, cumulative_labels, batch_colors)):
        axes[1, 1].plot(sorted(cum_gamma), alpha=0.7, color=color, linewidth=2, label=label)
    
    axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    axes[1, 1].set_xlabel('Voxel Index (sorted)')
    axes[1, 1].set_ylabel('Cumulative Gamma')
    axes[1, 1].set_title('Cumulative Gamma Effects')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Gamma magnitude comparison
    gamma_magnitudes = [np.mean(np.abs(gamma)) for gamma in gamma_star]
    
    axes[1, 2].bar(batch_labels, gamma_magnitudes, alpha=0.7, color=batch_colors)
    axes[1, 2].set_xlabel('Pipeline Batch')
    axes[1, 2].set_ylabel('Mean |Gamma|')
    axes[1, 2].set_title('Average Gamma Magnitude by Batch')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    # Print gamma statistics
    print("\n" + "=" * 60)
    print("GAMMA PARAMETER SUMMARY")
    print("=" * 60)
    print(f"Overall gamma mean: {np.mean(all_gamma_values):.6f}")
    print(f"Overall gamma std: {np.std(all_gamma_values):.6f}")
    print(f"Overall gamma range: [{np.min(all_gamma_values):.6f}, {np.max(all_gamma_values):.6f}]")
    
    print("\nBatch-specific statistics:")
    for i, batch in enumerate(unique_batches):
        gamma = gamma_star[i]
        print(f"  Batch {batch+1}: mean={np.mean(gamma):.6f}, "
              f"std={np.std(gamma):.6f}, "
              f"range=[{np.min(gamma):.6f}, {np.max(gamma):.6f}]")


def main():
    """Main demo function."""
    print("🧠 Pipeline Combat Demo - DWI Data Harmonization")
    print("=" * 60)

    # Simulate DWI data with pipeline biases
    print("\n1. Simulating DWI data with pipeline-induced biases...")
    n_pipeline_steps = 15

    biased_data, covariates, true_data = simulate_dwi_data(
        n_subjects=13,
        n_voxels=1000,
        n_directions=20,
        n_pipeline_steps=n_pipeline_steps,
    )

    # Visualize pipeline effects before correction
    print("\n2. Visualizing pipeline-induced biases...")
    visualize_pipeline_effects(biased_data, covariates, true_data)

    # Create batch dependency links for sequential pipeline
    print("\n3. Setting up pipeline dependencies...")

    # Create batch_links matrix for sequential dependencies:
    # Step 0 → Step 1 → Step 2 → Step 3
    batch_links = np.zeros((n_pipeline_steps, n_pipeline_steps), dtype=int)

    # Sequential dependencies: each step depends on the previous step
    for step in range(1, n_pipeline_steps):
        batch_links[step, step - 1] = 1  # step depends on (step-1)

    print("Pipeline dependency structure:")
    print("  Step 0 → Step 1 → Step 2 → Step 3")
    print(f"Batch links matrix:\n{batch_links}")
    print("Note: batch_links[i,j]=1 means step j is parent of step i")

    # Run pipeline combat with Bayesian network optimization
    print("\n4. Running Pipeline Combat with Bayesian Network optimization...")
    try:
        print(f"directions {covariates['modality']}")
        designs, models, gamma_star, delta_var_star = pipeline_combat(
            biased_data=biased_data,
            covariates=covariates,
            batch_col_index="batch",
            modality_col_index="modality",
            numerical_col_indexes=None,
            create_pca_block=True,
            pca_n_components=17,
            batch_links=None,  # batch_links,  # Enable Bayesian network optimization
        )

        print("✅ Pipeline Combat completed successfully!")
        print(f"Number of design matrices: {len(designs)}")
        print(f"Number of model sets: {len(models)}")
        print(f"Gamma parameters shape: {len(gamma_star)} batches")
        print(f"Delta parameters shape: {len(delta_var_star)} batches")

    except Exception as e:
        print(f"❌ Pipeline Combat failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # Visualize correction results
    print("\n5. Visualizing correction parameters...")
    visualize_correction_results(gamma_star, delta_var_star, covariates)

    # Visualize model goodness of fit
    print("\n6. Analyzing model goodness of fit against true data...")
    visualize_model_goodness_of_fit(
        biased_data, designs, models, covariates, true_data
    )

    # Visualize gamma parameters
    print("\n7. Analyzing gamma parameter corrections...")
    visualize_gamma_parameters(gamma_star, covariates)

    # Summary statistics
    print("\n8. Summary of correction parameters:")
    print("-" * 40)

    for i, (gamma, delta) in enumerate(zip(gamma_star, delta_var_star)):
        print(f"Pipeline Step {i+1}:")
        gamma_mean, gamma_std = np.mean(gamma), np.std(gamma)
        delta_mean, delta_std = np.mean(delta), np.std(delta)
        print(
            f"  Gamma (additive): mean={gamma_mean:.4f}, "
            f"std={gamma_std:.4f}"
        )
        print(
            f"  Delta (variance): mean={delta_mean:.4f}, "
            f"std={delta_std:.4f}"
        )

    print("\n" + "=" * 60)
    print("Demo completed! 🎉")
    print("\nInterpretation:")
    print(
        "- Gamma parameters show additive corrections for each pipeline step"
    )
    print("- Delta parameters show multiplicative variance corrections")
    print("- Bayesian network optimization models sequential dependencies")
    print("  between pipeline steps (Step0 → Step1 → Step2 → Step3)")
    print("- These parameters can be used to harmonize new data processed")
    print("  through the same pipeline steps")
    print("\nBayesian Network Benefits:")
    print("- Incorporates prior knowledge about pipeline dependencies")
    print("- Improves parameter estimation through structured priors")
    print("- More robust corrections for sequential processing workflows")


if __name__ == "__main__":
    main()

# %%
