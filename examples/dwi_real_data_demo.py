#!/usr/bin/env python3
# %%
"""
Real DWI Data PCA Demo.

This demo demonstrates PCA analysis on real diffusion-weighted imaging
(DWI) data from a single brain slice acquired with multiple gradient
directions.

The analysis treats:
- Voxels as features (spatial locations in the brain slice)
- Gradient directions as samples (different diffusion measurements)

This provides insight into the principal diffusion patterns across
the brain slice.
"""

import matplotlib.pyplot as plt
import numpy as np
from dipy.data import fetch_stanford_hardi, read_stanford_hardi

from pipelinecombat.model.pca import PCA


def fetch_real_dwi_data(normalize_by_b0=True):
    """
    Fetch real DWI data using DIPY's Stanford HARDI dataset.
    Optionally normalize diffusion-weighted signals by b0 (baseline) signal.

    Parameters
    ----------
    normalize_by_b0 : bool, optional
        If True, normalize DWI signals by averaged b0 signal and return only
        diffusion-weighted volumes. If False, return all volumes including b0.
        Default is True.

    Returns
    -------
    dwi_data : ndarray
        DWI data from a single slice. If normalize_by_b0=True:
        (n_dwi_directions, n_voxels) - b0 normalized, DWI only.
        If normalize_by_b0=False: (n_all_directions, n_voxels) - raw data.
    gtab : GradientTable
        Gradient table. If normalize_by_b0=True: only DWI directions.
        If normalize_by_b0=False: all directions including b0.
    """
    print("🔄 Fetching Stanford HARDI dataset...")

    # Fetch the dataset (publicly available, no registration needed)
    fetch_stanford_hardi()
    img, gtab = read_stanford_hardi()

    # Get the DWI data
    dwi_data = img.get_fdata()  # Shape: (x, y, z, directions)
    print(f"Original DWI shape: {dwi_data.shape}")

    # Select a representative axial slice (middle of the brain)
    z_slice = dwi_data.shape[2] // 2
    dwi_slice_4d = dwi_data[:, :, z_slice, :]  # Shape: (x, y, directions)

    # Reshape to (directions, voxels) - treating voxels as features
    n_x, n_y, n_directions = dwi_slice_4d.shape
    dwi_slice = dwi_slice_4d.reshape(n_x * n_y, n_directions).T

    # Remove background voxels (very low signal)
    mean_signal = np.mean(dwi_slice, axis=0)
    signal_threshold = np.percentile(mean_signal, 10)  # Keep 90% of voxels
    valid_voxels = mean_signal > signal_threshold
    dwi_slice = dwi_slice[:, valid_voxels]

    print(
        f"Selected slice {z_slice}, shape before processing: {dwi_slice.shape}"
    )

    if not normalize_by_b0:
        # Return raw data without normalization
        print("Using raw DWI data (no b0 normalization)")
        print(
            f"({dwi_slice.shape[0]} gradient directions, "
            f"{dwi_slice.shape[1]} brain voxels)"
        )
        print(
            f"B-values range: {gtab.bvals.min():.0f} - "
            f"{gtab.bvals.max():.0f} s/mm²"
        )
        print(
            f"Raw signal range: {dwi_slice.min():.1f} - {dwi_slice.max():.1f}"
        )
        return dwi_slice, gtab

    # Normalize by b0 (original behavior)
    print("Applying b0 normalization...")

    # Separate b0 and diffusion-weighted volumes
    b0_mask = gtab.bvals <= 50  # b0 volumes (allowing small tolerance)
    dwi_mask = gtab.bvals > 50  # diffusion-weighted volumes

    b0_volumes = dwi_slice[b0_mask, :]
    dwi_volumes = dwi_slice[dwi_mask, :]

    print(
        f"Found {np.sum(b0_mask)} b0 volumes and "
        f"{np.sum(dwi_mask)} DWI volumes"
    )

    # Average b0 volumes if multiple exist
    if b0_volumes.shape[0] > 1:
        b0_mean = np.mean(b0_volumes, axis=0)
        print(f"Averaged {b0_volumes.shape[0]} b0 volumes")
    else:
        b0_mean = b0_volumes[0, :]
        print("Using single b0 volume")

    # Normalize DWI signals by b0 (avoid division by zero)
    # Add small epsilon to prevent division by very small numbers
    epsilon = 1e-6
    b0_mean_safe = np.maximum(b0_mean, epsilon)

    # Calculate normalized signal: DWI / b0
    dwi_normalized = dwi_volumes / b0_mean_safe[np.newaxis, :]

    # Create gradient table for only DWI volumes
    gtab_dwi = gtab[dwi_mask]

    print(f"Final normalized DWI shape: {dwi_normalized.shape}")
    print(
        f"({dwi_normalized.shape[0]} gradient directions, "
        f"{dwi_normalized.shape[1]} brain voxels)"
    )
    print(
        f"B-values range: {gtab_dwi.bvals.min():.0f} - "
        f"{gtab_dwi.bvals.max():.0f} s/mm²"
    )
    print(
        f"Normalized signal range: {dwi_normalized.min():.4f} - "
        f"{dwi_normalized.max():.4f}"
    )

    return dwi_normalized, gtab_dwi


def visualize_comprehensive_dwi_pca(
    dwi_data, gtab, pca_model, n_components=20
):
    """
    Create comprehensive PCA analysis visualization for DWI data.

    Parameters
    ----------
    dwi_data : ndarray
        DWI data matrix (n_directions, n_voxels)
    gtab : GradientTable
        Gradient table information
    pca_model : PCA
        Fitted PCA model
    n_components : int
        Number of components to analyze
    """
    print("Creating comprehensive DWI PCA analysis...")

    # Generate the model to access components and variances properly
    model = pca_model.generate()

    # Get components from model - component 0 is the mean
    components = model.I  # Shape: (n_components+1, n_features)
    all_variances = model.variances

    # Skip the mean component (index 0) for variance analysis
    explained_variance = all_variances[1:]  # Actual PCA component variances
    n_pca_components = len(explained_variance)

    # Create transformed data
    transformed_data = []
    for i in range(dwi_data.shape[0]):
        beta, _ = model.fit(dwi_data[i, :])
        transformed_data.append(beta)
    transformed_data = np.array(transformed_data)

    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "🧠 DWI Data - Principal Component Analysis Results\n"
        "Gradient Directions as Samples, Brain Voxels as Features",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Get variance ratios
    total_variance = np.sum(explained_variance)
    explained_variance_ratio = explained_variance / total_variance
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # Show first n_components components
    n_show = min(n_components, n_pca_components)
    x_pos = np.arange(n_show)

    # 1. Explained Variance Ratio
    bars = axes[0, 0].bar(
        x_pos, explained_variance_ratio[:n_show], alpha=0.7, color="steelblue"
    )
    axes[0, 0].set_xlabel("Principal Component")
    axes[0, 0].set_ylabel("Explained Variance Ratio")
    axes[0, 0].set_title(
        "Individual Component Variance\n(DWI Diffusion Patterns)"
    )
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels([f"PC{i + 1}" for i in x_pos])
    axes[0, 0].grid(visible=True, alpha=0.3)

    # Add value labels on bars
    for _i, (bar, ratio) in enumerate(
        zip(bars, explained_variance_ratio[:n_show], strict=False)
    ):
        if ratio > 0.01:  # Only label if > 1%
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{ratio:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # 2. Cumulative Explained Variance
    axes[0, 1].plot(
        range(1, n_show + 1),
        cumulative_variance_ratio[:n_show],
        "o-",
        color="darkgreen",
        linewidth=2,
        markersize=6,
    )
    axes[0, 1].axhline(
        y=0.8, color="red", linestyle="--", alpha=0.7, label="80% threshold"
    )
    axes[0, 1].axhline(
        y=0.95,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label="95% threshold",
    )
    axes[0, 1].set_xlabel("Number of Components")
    axes[0, 1].set_ylabel("Cumulative Explained Variance")
    axes[0, 1].set_title(
        "Cumulative Variance Explained\n(Diffusion Information Retention)"
    )
    axes[0, 1].legend()
    axes[0, 1].grid(visible=True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1.05)

    # 3. Component Loadings Heatmap
    n_comp_heatmap = min(6, n_pca_components)
    n_feat_heatmap = min(100, components.shape[1])

    # Skip mean component (index 0) and use actual PCA components (1:)
    pca_components = components[1: n_comp_heatmap + 1, :n_feat_heatmap]

    # Calculate symmetric limits for color scale
    max_val = np.max(np.abs(pca_components))

    im = axes[0, 2].imshow(
        pca_components,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-max_val,
        vmax=max_val,
    )

    axes[0, 2].set_xlabel("Brain Voxels (subset)", fontsize=10)
    axes[0, 2].set_ylabel("Principal Components", fontsize=10)
    axes[0, 2].set_title(
        f"Component Loadings\n(First {n_comp_heatmap} PCs - Spatial Patterns)",
        fontsize=11,
    )
    axes[0, 2].set_yticks(range(n_comp_heatmap))
    axes[0, 2].set_yticklabels(
        [f"PC{i + 1}" for i in range(n_comp_heatmap)], fontsize=8
    )

    # Add colorbar
    plt.colorbar(im, ax=axes[0, 2], shrink=0.8, label="Loading Coefficient")

    # 4. PC1 vs PC2 Scatter Plot (if we have at least 2 components)
    if n_pca_components >= 2:
        b_values = gtab.bvals
        scatter = axes[1, 0].scatter(
            transformed_data[:, 1],  # PC1 (skip mean at index 0)
            transformed_data[:, 2],  # PC2
            alpha=0.6,
            s=30,
            c=b_values[: len(transformed_data)],
            cmap="viridis",
        )
        axes[1, 0].set_xlabel(
            f"PC1 ({explained_variance_ratio[0]:.1%} variance)"
        )
        axes[1, 0].set_ylabel(
            f"PC2 ({explained_variance_ratio[1]:.1%} variance)"
        )
        axes[1, 0].set_title("PC1 vs PC2 Distribution\n(Colored by B-value)")
        axes[1, 0].grid(visible=True, alpha=0.3)
        plt.colorbar(
            scatter, ax=axes[1, 0], shrink=0.8, label="B-value (s/mm²)"
        )

    # 5. B-value Distribution Analysis
    b_values = gtab.bvals
    unique_b_vals, b_counts = np.unique(b_values, return_counts=True)

    bars = axes[1, 1].bar(
        range(len(unique_b_vals)),
        b_counts,
        color=plt.cm.plasma(np.linspace(0, 1, len(unique_b_vals))),
        alpha=0.8,
    )
    axes[1, 1].set_xlabel("B-value Shell")
    axes[1, 1].set_ylabel("Number of Directions")
    axes[1, 1].set_title(
        "B-value Shell Distribution\n(Diffusion Encoding Strength)"
    )
    axes[1, 1].set_xticks(range(len(unique_b_vals)))
    axes[1, 1].set_xticklabels(
        [f"{int(b)}" for b in unique_b_vals], rotation=45
    )
    axes[1, 1].grid(visible=True, alpha=0.3)

    # Add value labels on bars
    for bar, count in zip(bars, b_counts, strict=False):
        height = bar.get_height()
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 6. Gradient Direction Analysis
    if hasattr(gtab, "bvecs"):
        bvecs = gtab.bvecs
        valid_dirs = b_values > 100  # Non-b0 volumes
        if np.sum(valid_dirs) > 0:
            valid_bvecs = bvecs[valid_dirs]
            valid_b_vals = b_values[valid_dirs]

            scatter = axes[1, 2].scatter(
                valid_bvecs[:, 0],
                valid_bvecs[:, 1],
                c=valid_b_vals,
                cmap="viridis",
                alpha=0.7,
                s=50,
            )

            axes[1, 2].set_xlabel("Gradient X-component")
            axes[1, 2].set_ylabel("Gradient Y-component")
            axes[1, 2].set_title("Gradient Directions\n(X-Y Projection)")
            axes[1, 2].set_aspect("equal")
            axes[1, 2].grid(visible=True, alpha=0.3)

            # Add unit circle
            theta = np.linspace(0, 2 * np.pi, 100)
            axes[1, 2].plot(
                np.cos(theta), np.sin(theta), "k--", alpha=0.3, linewidth=1
            )

            # Add colorbar
            plt.colorbar(
                scatter, ax=axes[1, 2], shrink=0.8, label="B-value (s/mm²)"
            )

    plt.tight_layout()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("DWI PCA ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Original data shape: {dwi_data.shape}")
    print(f"Number of components: {n_pca_components}")
    print(f"Total explained variance: {total_variance:.4f}")

    # Find components for variance thresholds
    var_80_idx = np.where(cumulative_variance_ratio >= 0.8)[0]
    var_95_idx = np.where(cumulative_variance_ratio >= 0.95)[0]

    if len(var_80_idx) > 0:
        print(f"Components for 80% variance: {var_80_idx[0] + 1}")
    if len(var_95_idx) > 0:
        print(f"Components for 95% variance: {var_95_idx[0] + 1}")

    print(f"\nTop {min(5, n_pca_components)} components variance ratios:")
    for i in range(min(5, n_pca_components)):
        print(f"  PC{i + 1}: {explained_variance_ratio[i]:.4f}")

    # Additional DWI-specific information
    print("\nDWI Acquisition Details:")
    print(f"  Gradient directions: {dwi_data.shape[0]}")
    print(f"  Brain voxels analyzed: {dwi_data.shape[1]}")
    print(
        f"  B-value range: {gtab.bvals.min():.0f} - "
        f"{gtab.bvals.max():.0f} s/mm²"
    )
    print(f"  Number of b0 volumes: {np.sum(gtab.bvals < 100)}")

    return fig


def visualize_dwi_loadings_analysis(pca_model, dwi_data, gtab):
    """
    Visualize PCA component loadings in detail for DWI data.

    Parameters
    ----------
    pca_model : PCA
        The fitted PCA model
    dwi_data : ndarray
        DWI data matrix (n_directions, n_voxels)
    gtab : GradientTable
        Gradient table information
    """
    print("Creating detailed DWI loadings analysis...")

    # Use the model's public interface
    model = pca_model.generate()
    components = model.I  # Shape: (n_components+1, n_features)
    all_variances = model.variances

    # Skip mean component for analysis
    pca_components = components[1:7, :]  # First 6 PCA components
    pca_variances = all_variances[1:7]  # Corresponding variances
    total_var = np.sum(
        all_variances[1:]
    )  # Total PCA variance (excluding mean)
    explained_variance_ratio = pca_variances / total_var

    n_comp = min(6, pca_components.shape[0])
    n_features_per_component = 10

    # Create figure with subplots for each component
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "🧠 DWI Component Loadings Analysis\nSpatial Diffusion Patterns",
        fontsize=16,
        fontweight="bold",
    )

    axes = axes.flatten()

    for i in range(n_comp):
        ax = axes[i]
        component_loadings = pca_components[i, :]  # Get component loadings

        # Get top positive and negative loadings
        abs_loadings = np.abs(component_loadings)
        top_indices = np.argsort(abs_loadings)[-n_features_per_component:][
            ::-1
        ]

        # Create horizontal bar plot
        top_loadings = component_loadings[top_indices]
        colors = ["red" if x < 0 else "blue" for x in top_loadings]

        bars = ax.barh(
            range(len(top_loadings)), top_loadings, color=colors, alpha=0.7
        )

        # Set labels (use voxel indices as feature names)
        labels = [f"Voxel {idx}" for idx in top_indices]

        ax.set_yticks(range(len(top_loadings)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Loading Value", fontsize=10)
        ax.set_title(
            f"PC{i + 1} ({explained_variance_ratio[i]:.1%} var)\n"
            "Top Voxel Loadings",
            fontweight="bold",
        )
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.5)
        ax.grid(visible=True, alpha=0.3)

        # Add value labels
        for _j, (bar, loading) in enumerate(
            zip(bars, top_loadings, strict=False)
        ):
            ax.text(
                loading + (0.001 if loading >= 0 else -0.001),
                bar.get_y() + bar.get_height() / 2,
                f"{loading:.3f}",
                ha="left" if loading >= 0 else "right",
                va="center",
                fontsize=8,
            )

    # Hide unused subplots (shouldn't be any for 6 components)
    for i in range(n_comp, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    # Print detailed loading statistics
    print("\n" + "=" * 70)
    print("DWI COMPONENT LOADINGS ANALYSIS")
    print("=" * 70)

    for i in range(n_comp):
        component_loadings = pca_components[i, :]
        variance_pct = explained_variance_ratio[i] * 100

        print(f"\nPC{i + 1} ({variance_pct:.1f}% variance):")
        print(
            f"  Range: [{np.min(component_loadings):.3f}, "
            f"{np.max(component_loadings):.3f}]"
        )
        mean_abs_loading = np.mean(np.abs(component_loadings))
        print(f"  Mean absolute loading: {mean_abs_loading:.3f}")
        print(f"  Std of loadings: {np.std(component_loadings):.3f}")

        # Top positive and negative loadings
        pos_loadings = component_loadings[component_loadings > 0]
        neg_loadings = component_loadings[component_loadings < 0]

        if len(pos_loadings) > 0:
            max_pos_idx = np.argmax(component_loadings)
            print(
                f"  Strongest positive loading: Voxel {max_pos_idx} "
                f"({component_loadings[max_pos_idx]:.3f})"
            )

        if len(neg_loadings) > 0:
            max_neg_idx = np.argmin(component_loadings)
            print(
                f"  Strongest negative loading: Voxel {max_neg_idx} "
                f"({component_loadings[max_neg_idx]:.3f})"
            )

    # Additional DWI-specific analysis
    print("\nDWI-Specific Loading Patterns:")
    print(f"  Total brain voxels: {pca_components.shape[1]}")
    print(f"  Gradient directions: {dwi_data.shape[0]}")
    print(
        f"  B-value range: {gtab.bvals.min():.0f} - {gtab.bvals.max():.0f} "
        "s/mm²"
    )

    return fig


def visualize_dwi_data_distribution(dwi_data, gtab, pca_model):
    """
    Visualize data distribution before/after PCA transformation for DWI data.

    Parameters
    ----------
    dwi_data : ndarray
        DWI data matrix (n_directions, n_voxels)
    gtab : GradientTable
        Gradient table information
    pca_model : PCA
        The fitted PCA model
    """
    print("Creating DWI data distribution analysis...")

    # Use model's public interface
    model = pca_model.generate()
    all_variances = model.variances
    explained_variance_ratio = all_variances[1:] / np.sum(all_variances[1:])

    # Generate transformed data
    transformed_data = []
    for i in range(dwi_data.shape[0]):
        beta, _ = model.fit(dwi_data[i, :])
        transformed_data.append(beta)
    transformed_data = np.array(transformed_data)

    n_components = transformed_data.shape[1] - 1  # Exclude mean component

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "🧠 DWI Data Distribution Analysis\n"
        "Before and After PCA Transformation",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Original data distribution (first few voxels)
    n_voxels_to_show = min(5, dwi_data.shape[1])
    voxel_indices = np.linspace(
        0, dwi_data.shape[1] - 1, n_voxels_to_show, dtype=int
    )

    for _i, voxel_idx in enumerate(voxel_indices):
        axes[0, 0].hist(
            dwi_data[:, voxel_idx],
            bins=20,
            alpha=0.6,
            label=f"Voxel {voxel_idx}",
            density=True,
        )
    axes[0, 0].set_xlabel("Signal Intensity")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title("Original DWI Signal Distribution\n(Sample Voxels)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(visible=True, alpha=0.3)

    # 2. Transformed data distribution (first few components, skip mean)
    n_comp_to_show = min(5, n_components)
    for i in range(n_comp_to_show):
        axes[0, 1].hist(
            transformed_data[:, i + 1],  # Skip mean component at index 0
            bins=20,
            alpha=0.6,
            label=f"PC{i + 1}",
            density=True,
        )
    axes[0, 1].set_xlabel("Coefficient Value")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("PCA Coefficients Distribution")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(visible=True, alpha=0.3)

    # 3. PC1 vs PC2 colored by B-value
    if n_components >= 2:
        b_values = gtab.bvals
        scatter = axes[0, 2].scatter(
            transformed_data[:, 1],  # PC1 (skip mean at index 0)
            transformed_data[:, 2],  # PC2
            c=b_values[: len(transformed_data)],
            cmap="viridis",
            alpha=0.7,
            s=40,
        )
        axes[0, 2].set_xlabel(f"PC1 ({explained_variance_ratio[0]:.1%})")
        axes[0, 2].set_ylabel(f"PC2 ({explained_variance_ratio[1]:.1%})")
        axes[0, 2].set_title("PC1 vs PC2 Distribution\n(Colored by B-value)")
        axes[0, 2].grid(visible=True, alpha=0.3)
        plt.colorbar(
            scatter, ax=axes[0, 2], shrink=0.8, label="B-value (s/mm²)"
        )

    # 4. Voxel variance before PCA
    voxel_variances = np.var(dwi_data, axis=0)
    n_voxels_show = min(50, len(voxel_variances))
    voxel_subset = np.linspace(
        0, len(voxel_variances) - 1, n_voxels_show, dtype=int
    )

    axes[1, 0].bar(
        range(n_voxels_show),
        voxel_variances[voxel_subset],
        alpha=0.7,
        color="orange",
    )
    axes[1, 0].set_xlabel("Voxel Index (subset)")
    axes[1, 0].set_ylabel("Signal Variance")
    axes[1, 0].set_title("Voxel Signal Variances\n(Spatial Variability)")
    axes[1, 0].grid(visible=True, alpha=0.3)

    # 5. PC coefficient variance
    pc_variances = np.var(
        transformed_data[:, 1: n_comp_to_show + 1], axis=0
    )  # Skip mean
    axes[1, 1].bar(
        range(len(pc_variances)), pc_variances, alpha=0.7, color="green"
    )
    axes[1, 1].set_xlabel("Principal Component")
    axes[1, 1].set_ylabel("Coefficient Variance")
    axes[1, 1].set_title("PC Coefficient Variances")
    axes[1, 1].set_xticks(range(len(pc_variances)))
    axes[1, 1].set_xticklabels(
        [f"PC{i + 1}" for i in range(len(pc_variances))]
    )
    axes[1, 1].grid(visible=True, alpha=0.3)

    # 6. B-value vs Signal Analysis
    b_values = gtab.bvals
    unique_b_vals = np.unique(b_values)

    # Calculate mean signal for each b-value shell
    b_val_signals = []
    b_val_errors = []

    for b_val in unique_b_vals:
        b_indices = b_values == b_val
        if np.sum(b_indices) > 0:
            b_signals = np.mean(
                dwi_data[b_indices, :], axis=1
            )  # Mean across voxels
            b_val_signals.append(np.mean(b_signals))
            b_val_errors.append(np.std(b_signals))

    axes[1, 2].errorbar(
        unique_b_vals,
        b_val_signals,
        yerr=b_val_errors,
        marker="o",
        capsize=5,
        capthick=2,
        linewidth=2,
        alpha=0.8,
    )
    axes[1, 2].set_xlabel("B-value (s/mm²)")
    axes[1, 2].set_ylabel("Mean Signal Intensity")
    axes[1, 2].set_title("Signal Attenuation vs B-value\n(Diffusion Decay)")
    axes[1, 2].grid(visible=True, alpha=0.3)

    # Add exponential decay reference line if we have different b-values
    if len(unique_b_vals) > 1:
        b_range = np.linspace(unique_b_vals.min(), unique_b_vals.max(), 100)
        # Simple mono-exponential model: S = S0 * exp(-b * ADC)
        if len(b_val_signals) >= 2:
            # Estimate ADC from first two points
            s0, s1 = b_val_signals[0], b_val_signals[1]
            b0, b1 = unique_b_vals[0], unique_b_vals[1]
            if s1 > 0 and b1 > b0:
                adc_est = -np.log(s1 / s0) / (b1 - b0)
                decay_model = s0 * np.exp(-b_range * adc_est)
                axes[1, 2].plot(
                    b_range,
                    decay_model,
                    "--",
                    alpha=0.5,
                    label=f"Mono-exp (ADC≈{adc_est:.3f})",
                    color="red",
                )
                axes[1, 2].legend(fontsize=8)

    plt.tight_layout()

    # Print distribution statistics
    print("\n" + "=" * 60)
    print("DWI DATA DISTRIBUTION SUMMARY")
    print("=" * 60)
    print(f"Original data shape: {dwi_data.shape}")
    print(f"Transformed data shape: {transformed_data.shape}")

    print("\nOriginal data statistics:")
    print(f"  Mean: {np.mean(dwi_data):.4f}")
    print(f"  Std: {np.std(dwi_data):.4f}")
    print(f"  Min: {np.min(dwi_data):.4f}")
    print(f"  Max: {np.max(dwi_data):.4f}")

    print("\nTransformed data statistics (PCA coefficients):")
    pca_coeffs = transformed_data[:, 1:]  # Exclude mean component
    print(f"  Mean: {np.mean(pca_coeffs):.4f}")
    print(f"  Std: {np.std(pca_coeffs):.4f}")
    print(f"  Min: {np.min(pca_coeffs):.4f}")
    print(f"  Max: {np.max(pca_coeffs):.4f}")

    print("\nB-value distribution:")
    for b_val, count in zip(
        *np.unique(b_values, return_counts=True), strict=False
    ):
        print(f"  {int(b_val):4d} s/mm²: {count:2d} directions")

    return fig


def visualize_dwi_reconstruction_and_distances(
    pca_model, dwi_data, n_samples_for_distances=50
):
    """
    Visualize DWI PCA reconstruction quality and distance preservation.

    Top row: Reconstructed DWI signals using components that explain
    50%, 80%, 100% variance
    Bottom row: Distance preservation plots for each variance threshold

    Parameters
    ----------
    pca_model : PCA
        The fitted PCA model
    dwi_data : np.ndarray
        Original DWI data before PCA transformation
    n_samples_for_distances : int, optional
        Number of directions for distance calculations (for performance)
    """
    print("Creating DWI reconstruction and distance preservation analysis...")

    # Get model components and variances
    model = pca_model.generate()
    # model.X is the design matrix: (n_features, n_components+1) including mean

    # Adjust for the fact that component 0 is now the mean
    # Skip the mean component (index 0) for variance calculations
    actual_variances = model.variances[1:]  # Skip mean variance
    explained_variance_ratio = actual_variances / np.sum(actual_variances)
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Find number of components for different variance thresholds
    # Add 1 because we need to account for the mean component (index 0)
    n_comp_50 = np.where(cumulative_variance >= 0.5)[0]
    n_comp_80 = np.where(cumulative_variance >= 0.8)[0]
    n_comp_100 = len(explained_variance_ratio)

    n_comp_50 = (
        (n_comp_50[0] + 1) + 1 if len(n_comp_50) > 0 else n_comp_100 + 1
    )
    n_comp_80 = (
        (n_comp_80[0] + 1) + 1 if len(n_comp_80) > 0 else n_comp_100 + 1
    )
    n_comp_100 = n_comp_100 + 1  # +1 for mean component

    thresholds = [50, 80, 100]
    n_components_list = [n_comp_50, n_comp_80, n_comp_100]

    # Transform data to PCA space
    transformed_data = []
    for i in range(dwi_data.shape[0]):
        beta, _ = model.fit(dwi_data[i, :])
        transformed_data.append(beta)
    transformed_data = np.array(transformed_data)

    # Create 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "🧠 DWI Reconstruction Quality and Distance Preservation",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Subsample data for distance calculations (performance)
    n_samples = min(n_samples_for_distances, dwi_data.shape[0])
    sample_indices = np.random.choice(
        dwi_data.shape[0], n_samples, replace=False
    )
    original_subset = dwi_data[sample_indices]

    # Calculate original pairwise distances
    from scipy.spatial.distance import pdist

    original_distances = pdist(original_subset, metric="euclidean")

    for i, (threshold, n_comp) in enumerate(
        zip(thresholds, n_components_list, strict=False)
    ):
        # Top row: Reconstruction visualization
        ax_top = axes[0, i]

        # Reconstruct data using first n_comp components (including mean at 0)
        # Correct reconstruction: coefficients @ design_matrix.T
        # model.X is (n_features, n_components), so we need .T
        reconstructed_data = (
            transformed_data[:, :n_comp] @ model.X[:, :n_comp].T
        )
        reconstructed_subset = reconstructed_data[sample_indices]

        # Calculate reconstruction error
        mse = np.mean((original_subset - reconstructed_subset) ** 2)
        # Calculate explained variance (excluding mean component)
        actual_n_comp = n_comp - 1  # -1 because component 0 is mean
        explained_var = (
            cumulative_variance[actual_n_comp - 1]
            if actual_n_comp > 0
            else 1.0
        )

        # Plot original vs reconstructed (first gradient direction as example)
        gradient_indices = np.linspace(
            0, original_subset.shape[1] - 1, 100, dtype=int
        )
        example_idx = 0

        ax_top.plot(
            gradient_indices,
            original_subset[example_idx, gradient_indices],
            "o-",
            alpha=0.7,
            label="Original DWI",
            color="blue",
            markersize=3,
        )
        ax_top.plot(
            gradient_indices,
            reconstructed_subset[example_idx, gradient_indices],
            "x-",
            alpha=0.7,
            label="Reconstructed",
            color="red",
            markersize=3,
        )

        ax_top.set_xlabel("Gradient Direction Index", fontsize=10)
        ax_top.set_ylabel("Signal Intensity", fontsize=10)
        ax_top.set_title(
            f"{threshold}% Variance\n({actual_n_comp} PCs, MSE: {mse:.4f})",
            fontsize=11,
        )
        ax_top.legend(fontsize=9)
        ax_top.grid(visible=True, alpha=0.3)
        ax_top.tick_params(axis="both", labelsize=8)

        # Bottom row: Distance preservation
        ax_bottom = axes[1, i]

        # Calculate reconstructed pairwise distances
        reconstructed_distances = pdist(
            reconstructed_subset, metric="euclidean"
        )

        # Create scatter plot of original vs reconstructed distances
        ax_bottom.scatter(
            original_distances,
            reconstructed_distances,
            alpha=0.5,
            s=20,
            color="purple",
        )

        # Add perfect correlation line
        min_dist = min(
            np.min(original_distances), np.min(reconstructed_distances)
        )
        max_dist = max(
            np.max(original_distances), np.max(reconstructed_distances)
        )
        ax_bottom.plot(
            [min_dist, max_dist],
            [min_dist, max_dist],
            "r--",
            alpha=0.8,
            label="Perfect preservation",
        )

        # Calculate correlation
        correlation = np.corrcoef(original_distances, reconstructed_distances)[
            0, 1
        ]

        ax_bottom.set_xlabel("Original Distances", fontsize=10)
        ax_bottom.set_ylabel("Reconstructed Distances", fontsize=10)
        ax_bottom.set_title(
            f"Distance Preservation\n(r = {correlation:.3f})", fontsize=11
        )
        ax_bottom.legend(fontsize=9)
        ax_bottom.grid(visible=True, alpha=0.3)
        ax_bottom.tick_params(axis="both", labelsize=8)

    plt.tight_layout()

    # Print summary
    print("\n" + "=" * 70)
    print("DWI RECONSTRUCTION AND DISTANCE PRESERVATION SUMMARY")
    print("=" * 70)
    print(f"Analysis based on {n_samples} gradient directions")

    for _i, (threshold, n_comp) in enumerate(
        zip(thresholds, n_components_list, strict=False)
    ):
        # Calculate metrics with proper reconstruction (including mean)
        reconstructed_data = (
            transformed_data[:, :n_comp] @ model.X[:, :n_comp].T
        )
        reconstructed_subset = reconstructed_data[sample_indices]

        mse = np.mean((original_subset - reconstructed_subset) ** 2)
        actual_n_comp = n_comp - 1  # -1 because component 0 is mean
        explained_var = (
            cumulative_variance[actual_n_comp - 1]
            if actual_n_comp > 0
            else 1.0
        )

        reconstructed_distances = pdist(
            reconstructed_subset, metric="euclidean"
        )
        correlation = np.corrcoef(original_distances, reconstructed_distances)[
            0, 1
        ]

        print(f"\n{threshold}% Variance Threshold:")
        print(f"  Components used: {actual_n_comp} (+ mean)")
        print(f"  Actual variance explained: {explained_var:.3f}")
        print(f"  Reconstruction MSE: {mse:.6f}")
        print(f"  Distance correlation: {correlation:.3f}")

    return fig


def main():
    """Main function to run the real DWI PCA demo."""
    print("🧠 Real DWI Data PCA Demo")
    print("=" * 50)

    # Control parameters
    use_b0_normalization = True  # Set to False to use raw DWI data

    print(
        "📊 B0 normalization: "
        f"{'Enabled' if use_b0_normalization else 'Disabled'}"
    )

    try:
        # Fetch real DWI data
        dwi_data, gtab = fetch_real_dwi_data(
            normalize_by_b0=use_b0_normalization
        )

        # Apply PCA
        print("\n🔄 Applying PCA to DWI data...")
        n_components = min(12, dwi_data.shape[0] - 1)  # Don't exceed samples
        pca_dwi = PCA(numerical_data=dwi_data, n_components=n_components)

        print(f"PCA model created with {len(pca_dwi._n_variance)} components")

        # Create visualizations (matching pca_demo structure)
        print("\n1. Creating comprehensive DWI PCA analysis...")
        visualize_comprehensive_dwi_pca(dwi_data, gtab, pca_dwi, n_components)

        print("\n2. Creating detailed component loadings analysis...")
        visualize_dwi_loadings_analysis(pca_dwi, dwi_data, gtab)

        print("\n3. Creating data distribution analysis...")
        visualize_dwi_data_distribution(dwi_data, gtab, pca_dwi)

        print(
            "\n4. Creating reconstruction and distance preservation plots..."
        )
        visualize_dwi_reconstruction_and_distances(pca_dwi, dwi_data)

        # Show all plots
        plt.show()

        print("\n🎉 Real DWI PCA Demo completed successfully!")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Error in DWI demo: {e!s}")
        raise


if __name__ == "__main__":
    main()

# %%
