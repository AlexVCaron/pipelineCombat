# %%
#!/usr/bin/env python3
"""
PCA visualization functions for Pipeline Combat.

This module provides comprehensive visualization functions for analyzing
Principal Component Analysis results in the context of neuroimaging data
harmonization, following the patterns established in the demo visualizations.
"""

import numpy as np

# Optional visualization imports
try:
    import matplotlib.pyplot as plt

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Matplotlib not available. Running without visualizations.")
    print("To enable visualizations, install with: uv add matplotlib")


def visualize_pca_analysis(
    pca_model,
    original_data,
    transformed_data=None,
    feature_names=None,
    n_components_to_show=10,
):
    """
    Create comprehensive PCA analysis visualization.

    Parameters
    ----------
    pca_model : PCA
        The fitted PCA model from pipelinecombat.model.pca
    original_data : np.ndarray
        Original data before PCA transformation, shape (n_samples, n_features)
    transformed_data : np.ndarray, optional
        PCA-transformed data, shape (n_samples, n_components)
    feature_names : list, optional
        Names of original features
    n_components_to_show : int, optional
        Number of components to show in detailed plots
    """
    if not VISUALIZATION_AVAILABLE:
        print("Visualization not available. Install matplotlib.")
        return

    print("Creating comprehensive PCA analysis...")

    # Generate the model to access components and variances properly
    model = pca_model.generate()

    # Get components from model - now component 0 is the mean
    components = model.I  # Shape: (n_components+1, n_features)
    all_variances = model.variances

    # Skip the mean component (index 0) for variance analysis
    explained_variance = all_variances[1:]  # Actual PCA component variances
    n_components = len(explained_variance)

    # Create transformed data if not provided
    if transformed_data is None:
        # Transform using the model's fit method per sample
        transformed_data = []
        for i in range(original_data.shape[0]):
            beta, _ = model.fit(original_data[i, :])
            transformed_data.append(beta)
        transformed_data = np.array(transformed_data)

    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Principal Component Analysis Results",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # 1. Explained Variance Ratio (skip mean component)
    total_variance = np.sum(explained_variance)
    explained_variance_ratio = explained_variance / total_variance
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # Show first n_components_to_show components
    n_show = min(n_components_to_show, n_components)
    x_pos = np.arange(n_show)

    bars = axes[0, 0].bar(
        x_pos, explained_variance_ratio[:n_show], alpha=0.7, color="steelblue"
    )
    axes[0, 0].set_xlabel("Principal Component")
    axes[0, 0].set_ylabel("Explained Variance Ratio")
    axes[0, 0].set_title("Individual Component Variance")
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels([f"PC{i + 1}" for i in x_pos])
    axes[0, 0].grid(visible=True, alpha=0.3)

    # Add value labels on bars
    for _, (bar, ratio) in enumerate(
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
    axes[0, 1].set_title("Cumulative Variance Explained")
    axes[0, 1].legend()
    axes[0, 1].grid(visible=True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1.05)

    # 3. Component Loadings Heatmap (first few components, skip mean)
    n_comp_heatmap = min(6, n_components)
    n_feat_heatmap = min(20, components.shape[1])  # (n_comp, n_feat)

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

    axes[0, 2].set_xlabel("Features", fontsize=10)
    axes[0, 2].set_ylabel("Principal Components", fontsize=10)
    axes[0, 2].set_title(
        f"Component Loadings\n(First {n_comp_heatmap} PCs)", fontsize=11
    )
    axes[0, 2].set_yticks(range(n_comp_heatmap))
    axes[0, 2].set_yticklabels(
        [f"PC{i + 1}" for i in range(n_comp_heatmap)], fontsize=8
    )

    if feature_names is not None and len(feature_names) >= n_feat_heatmap:
        axes[0, 2].set_xticks(range(n_feat_heatmap))
        axes[0, 2].set_xticklabels(
            feature_names[:n_feat_heatmap], rotation=45, ha="right"
        )

    # Add colorbar
    plt.colorbar(im, ax=axes[0, 2], shrink=0.8)

    # 4. PC1 vs PC2 Scatter Plot
    if n_components >= 2:
        scatter = axes[1, 0].scatter(
            transformed_data[:, 0],
            transformed_data[:, 1],
            alpha=0.6,
            s=30,
            c=range(len(transformed_data)),
            cmap="viridis",
        )
        axes[1, 0].set_xlabel(
            f"PC1 ({explained_variance_ratio[0]:.1%} variance)"
        )
        axes[1, 0].set_ylabel(
            f"PC2 ({explained_variance_ratio[1]:.1%} variance)"
        )
        axes[1, 0].set_title("PC1 vs PC2 Sample Distribution")
        axes[1, 0].grid(visible=True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], shrink=0.8, label="Sample Index")

    # 5. Component Magnitude Analysis
    component_magnitudes = np.linalg.norm(components, axis=1)
    bars = axes[1, 1].bar(
        range(n_show), component_magnitudes[:n_show], alpha=0.7, color="coral"
    )
    axes[1, 1].set_xlabel("Principal Component")
    axes[1, 1].set_ylabel("Component Magnitude (L2 norm)")
    axes[1, 1].set_title("Component Vector Magnitudes")
    axes[1, 1].set_xticks(range(n_show))
    axes[1, 1].set_xticklabels([f"PC{i + 1}" for i in range(n_show)])
    axes[1, 1].grid(visible=True, alpha=0.3)

    # Expected magnitude is 1.0 for normalized components
    axes[1, 1].axhline(
        y=1.0, color="red", linestyle="--", alpha=0.7, label="Expected (1.0)"
    )
    axes[1, 1].legend()

    # 6. Reconstruction Error Analysis
    # Calculate reconstruction error for different numbers of components
    if n_components > 1:
        reconstruction_errors = []
        n_components_range = range(1, min(15, n_components) + 1)

        for n_comp in n_components_range:
            # Reconstruct with first n_comp components
            # components is (n_components, n_features), so:
            reconstructed = (
                transformed_data[:, :n_comp] @ components[:n_comp, :]
            )
            mse = np.mean((original_data - reconstructed) ** 2)
            reconstruction_errors.append(mse)

        axes[1, 2].semilogy(
            n_components_range,
            reconstruction_errors,
            "o-",
            color="purple",
            linewidth=2,
            markersize=6,
        )
        axes[1, 2].set_xlabel("Number of Components")
        axes[1, 2].set_ylabel("Mean Squared Error (log scale)")
        axes[1, 2].set_title("Reconstruction Error vs Components")
        axes[1, 2].grid(visible=True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("PCA ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Original data shape: {original_data.shape}")
    print(f"Number of components: {n_components}")
    print(f"Total explained variance: {total_variance:.4f}")

    # Find components for different variance thresholds
    var_80_idx = np.where(cumulative_variance_ratio >= 0.8)[0]
    var_95_idx = np.where(cumulative_variance_ratio >= 0.95)[0]

    if len(var_80_idx) > 0:
        print(f"Components for 80% variance: {var_80_idx[0] + 1}")
    if len(var_95_idx) > 0:
        print(f"Components for 95% variance: {var_95_idx[0] + 1}")

    print(f"\nTop {min(5, n_components)} components variance ratios:")
    for i in range(min(5, n_components)):
        print(f"  PC{i + 1}: {explained_variance_ratio[i]:.4f}")

    print("\nComponent magnitudes (should be ~1.0):")
    for i in range(min(5, n_components)):
        print(f"  PC{i + 1}: {component_magnitudes[i]:.6f}")


def visualize_pca_loadings(
    pca_model,
    feature_names=None,
    n_components=6,
    n_features_per_component=10,
):
    """
    Visualize PCA component loadings in detail.

    Parameters
    ----------
    pca_model : PCA
        The fitted PCA model
    feature_names : list, optional
        Names of original features
    n_components : int, optional
        Number of components to analyze
    n_features_per_component : int, optional
        Number of top features to show per component
    """
    if not VISUALIZATION_AVAILABLE:
        print("Visualization not available. Install matplotlib and seaborn.")
        return

    print("Creating detailed PCA loadings analysis...")

    # Use the model's public interface
    model = pca_model.generate()
    components = model.I  # Shape: (n_components, n_features)
    explained_variance_ratio = model.variances / np.sum(model.variances)
    n_comp = min(n_components, len(explained_variance_ratio))

    # Create figure with subplots for each component
    fig, axes = plt.subplots(
        2, 3, figsize=(18, 12) if n_comp <= 6 else (24, 16)
    )
    axes = axes.flatten() if n_comp <= 6 else axes.reshape(-1)
    fig.suptitle("PCA Component Loadings Analysis", fontsize=16)

    for i in range(n_comp):
        ax = axes[i] if n_comp <= len(axes) else plt.subplot(3, 3, i + 1)
        component_loadings = components[i, :]  # Get row i for component i

        # Get top positive and negative loadings
        abs_loadings = np.abs(component_loadings)
        top_indices = np.argsort(abs_loadings)[-n_features_per_component:][
            ::-1
        ]

        # Create horizontal bar plot
        top_loadings = component_loadings[top_indices]
        colors = ["red" if x < 0 else "blue" for x in top_loadings]

        bars = ax.barh(range(len(top_loadings)), top_loadings, color=colors)

        # Set labels
        if feature_names is not None:
            labels = [feature_names[idx] for idx in top_indices]
        else:
            labels = [f"Feature {idx}" for idx in top_indices]

        ax.set_yticks(range(len(top_loadings)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Loading Value")
        ax.set_title(f"PC{i + 1} ({explained_variance_ratio[i]:.1%} var)")
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.5)
        ax.grid(visible=True, alpha=0.3)

        # Add value labels
        for _, (bar, loading) in enumerate(
            zip(bars, top_loadings, strict=False)
        ):
            ax.text(
                loading + (0.01 if loading >= 0 else -0.01),
                bar.get_y() + bar.get_height() / 2,
                f"{loading:.3f}",
                ha="left" if loading >= 0 else "right",
                va="center",
                fontsize=8,
            )

    # Hide unused subplots
    for i in range(n_comp, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()

    # Print detailed loading statistics
    print("\n" + "=" * 60)
    print("COMPONENT LOADINGS ANALYSIS")
    print("=" * 60)

    for i in range(n_comp):
        component_loadings = components[i, :]  # Get row i for component i
        print(f"\nPC{i + 1} ({explained_variance_ratio[i]:.3f} variance):")
        print(
            f"  Loading range: [{np.min(component_loadings):.3f}, "
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
                f"  Strongest positive loading: Feature {max_pos_idx} "
                f"({component_loadings[max_pos_idx]:.3f})"
            )

        if len(neg_loadings) > 0:
            max_neg_idx = np.argmin(component_loadings)
            print(
                f"  Strongest negative loading: Feature {max_neg_idx} "
                f"({component_loadings[max_neg_idx]:.3f})"
            )


def visualize_pca_data_distribution(
    original_data,
    transformed_data,
    pca_model,
    group_labels=None,
    sample_names=None,
):
    """
    Visualize data distribution before and after PCA transformation.

    Parameters
    ----------
    original_data : np.ndarray
        Original data, shape (n_samples, n_features)
    transformed_data : np.ndarray
        PCA-transformed data, shape (n_samples, n_components)
    pca_model : PCA
        The fitted PCA model
    group_labels : array-like, optional
        Group labels for samples (e.g., site, diagnosis)
    sample_names : list, optional
        Names or IDs for samples
    """
    if not VISUALIZATION_AVAILABLE:
        print("Visualization not available. Install matplotlib and seaborn.")
        return

    print("Creating PCA data distribution analysis...")

    # Use model's public interface
    model = pca_model.generate()
    n_components = transformed_data.shape[1]
    explained_variance_ratio = model.variances / np.sum(model.variances)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("PCA Data Distribution Analysis", fontsize=16)

    # 1. Original data distribution (first few features)
    n_features_to_show = min(5, original_data.shape[1])
    for i in range(n_features_to_show):
        axes[0, 0].hist(
            original_data[:, i], bins=20, alpha=0.6, label=f"Feature {i + 1}"
        )
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Original Features Distribution")
    axes[0, 0].legend()
    axes[0, 0].grid(visible=True, alpha=0.3)

    # 2. Transformed data distribution (first few components)
    n_comp_to_show = min(5, n_components)
    for i in range(n_comp_to_show):
        axes[0, 1].hist(
            transformed_data[:, i], bins=20, alpha=0.6, label=f"PC{i + 1}"
        )
    axes[0, 1].set_xlabel("Value")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("PC Scores Distribution")
    axes[0, 1].legend()
    axes[0, 1].grid(visible=True, alpha=0.3)

    # 3. PC1 vs PC2 with optional grouping
    if n_components >= 2:
        if group_labels is not None:
            unique_groups = np.unique(group_labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_groups)))

            for group_idx, group in enumerate(unique_groups):
                mask = group_labels == group
                axes[0, 2].scatter(
                    transformed_data[mask, 0],
                    transformed_data[mask, 1],
                    c=[colors[group_idx]],
                    label=f"Group {group}",
                    alpha=0.7,
                    s=40,
                )
            axes[0, 2].legend()
        else:
            axes[0, 2].scatter(
                transformed_data[:, 0],
                transformed_data[:, 1],
                alpha=0.7,
                s=40,
                c="steelblue",
            )

        axes[0, 2].set_xlabel(f"PC1 ({explained_variance_ratio[0]:.1%})")
        axes[0, 2].set_ylabel(f"PC2 ({explained_variance_ratio[1]:.1%})")
        axes[0, 2].set_title("PC1 vs PC2 Distribution")
        axes[0, 2].grid(visible=True, alpha=0.3)

    # 4. Feature variance before PCA
    feature_variances = np.var(original_data, axis=0)
    axes[1, 0].bar(
        range(min(20, len(feature_variances))),
        feature_variances[:20],
        alpha=0.7,
        color="orange",
    )
    axes[1, 0].set_xlabel("Feature Index")
    axes[1, 0].set_ylabel("Variance")
    axes[1, 0].set_title("Original Feature Variances")
    axes[1, 0].grid(visible=True, alpha=0.3)

    # 5. PC score variance
    pc_variances = np.var(transformed_data, axis=0)
    axes[1, 1].bar(
        range(len(pc_variances)), pc_variances, alpha=0.7, color="green"
    )
    axes[1, 1].set_xlabel("Principal Component")
    axes[1, 1].set_ylabel("Score Variance")
    axes[1, 1].set_title("PC Score Variances")
    axes[1, 1].set_xticks(range(len(pc_variances)))
    axes[1, 1].set_xticklabels(
        [f"PC{i + 1}" for i in range(len(pc_variances))]
    )
    axes[1, 1].grid(visible=True, alpha=0.3)

    # 6. Sample distances in original vs PC space
    if original_data.shape[0] <= 100:  # Only for reasonable number of samples
        # Calculate pairwise distances
        from scipy.spatial.distance import pdist

        original_distances = pdist(original_data)
        pc_distances = pdist(transformed_data)

        axes[1, 2].scatter(original_distances, pc_distances, alpha=0.5, s=20)
        axes[1, 2].plot(
            [0, np.max(original_distances)],
            [0, np.max(pc_distances)],
            "r--",
            alpha=0.5,
        )
        axes[1, 2].set_xlabel("Original Space Distance")
        axes[1, 2].set_ylabel("PC Space Distance")
        axes[1, 2].set_title("Distance Preservation")
        axes[1, 2].grid(visible=True, alpha=0.3)

        # Calculate correlation
        distance_corr = np.corrcoef(original_distances, pc_distances)[0, 1]
        axes[1, 2].text(
            0.05,
            0.95,
            f"r = {distance_corr:.3f}",
            transform=axes[1, 2].transAxes,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "alpha": 0.8
            },
            verticalalignment="top",
        )

    plt.tight_layout()
    plt.show()

    # Print distribution statistics
    print("\n" + "=" * 60)
    print("DATA DISTRIBUTION SUMMARY")
    print("=" * 60)
    print(f"Original data shape: {original_data.shape}")
    print(f"Transformed data shape: {transformed_data.shape}")

    print("\nOriginal data statistics:")
    print(f"  Mean: {np.mean(original_data):.4f}")
    print(f"  Std: {np.std(original_data):.4f}")
    print(f"  Min: {np.min(original_data):.4f}")
    print(f"  Max: {np.max(original_data):.4f}")

    print("\nTransformed data statistics:")
    print(f"  Mean: {np.mean(transformed_data):.4f}")
    print(f"  Std: {np.std(transformed_data):.4f}")
    print(f"  Min: {np.min(transformed_data):.4f}")
    print(f"  Max: {np.max(transformed_data):.4f}")

    if group_labels is not None:
        unique_groups = np.unique(group_labels)
        print("\nGroup distribution:")
        for group in unique_groups:
            count = np.sum(group_labels == group)
            print(f"  Group {group}: {count} samples")


def visualize_pca_reconstruction_and_distances(
    pca_model, original_data, n_samples_for_distances=50
):
    """
    Visualize PCA reconstruction quality and distance preservation.

    Top row: Reconstructed data using components that explain
    50%, 80%, 100% variance
    Bottom row: Distance preservation plots for each variance threshold

    Parameters
    ----------
    pca_model : PCA
        The fitted PCA model
    original_data : np.ndarray
        Original data before PCA transformation
    n_samples_for_distances : int, optional
        Number of samples to use for distance calculations (for performance)
    """
    if not VISUALIZATION_AVAILABLE:
        print("Visualization not available. Install matplotlib and seaborn.")
        return

    print("Creating PCA reconstruction and distance preservation analysis...")

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
    for i in range(original_data.shape[0]):
        beta, _ = model.fit(original_data[i, :])
        transformed_data.append(beta)
    transformed_data = np.array(transformed_data)

    # Create 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "PCA Reconstruction Quality and Distance Preservation",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Subsample data for distance calculations (performance)
    n_samples = min(n_samples_for_distances, original_data.shape[0])
    sample_indices = np.random.choice(
        original_data.shape[0], n_samples, replace=False
    )
    original_subset = original_data[sample_indices]

    # Calculate original pairwise distances
    from scipy.spatial.distance import pdist

    original_distances = pdist(original_subset, metric="euclidean")

    for comp_idx, (threshold, n_comp) in enumerate(
        zip(thresholds, n_components_list, strict=False)
    ):
        # Top row: Reconstruction visualization
        ax_top = axes[0, comp_idx]

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

        # Plot original vs reconstructed (first two features as example)
        ax_top.scatter(
            original_subset[:, 0],
            original_subset[:, 1],
            alpha=0.6,
            s=30,
            label="Original",
            color="blue",
        )
        ax_top.scatter(
            reconstructed_subset[:, 0],
            reconstructed_subset[:, 1],
            alpha=0.6,
            s=30,
            label="Reconstructed",
            color="red",
            marker="x",
        )

        ax_top.set_xlabel("Feature 1", fontsize=10)
        ax_top.set_ylabel("Feature 2", fontsize=10)
        ax_top.set_title(
            f"{threshold}% Variance\n({actual_n_comp} PCs, MSE: {mse:.4f})",
            fontsize=11,
        )
        ax_top.legend(fontsize=9)
        ax_top.grid(visible=True, alpha=0.3)
        ax_top.tick_params(axis="both", labelsize=8)

        # Bottom row: Distance preservation
        ax_bottom = axes[1, comp_idx]

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
    plt.show()

    # Print summary
    print("\n" + "=" * 70)
    print("RECONSTRUCTION AND DISTANCE PRESERVATION SUMMARY")
    print("=" * 70)
    print(f"Analysis based on {n_samples} samples")

    for _, (threshold, n_comp) in enumerate(
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


def create_pca_demo_with_neuroimaging_data():
    """
    Create a complete PCA demonstration using simulated neuroimaging data.

    This function demonstrates the full PCA workflow with realistic
    neuroimaging data patterns, following the demo patterns from the project.
    """
    if not VISUALIZATION_AVAILABLE:
        print("Visualization not available. Install matplotlib.")
        return None

    # Import PCA locally to avoid module-level import issues
    from pipelinecombat.model.pca import PCA

    print("🧠 PCA Neuroimaging Data Demo")
    print("=" * 50)

    # Simulate multi-site DTI data
    np.random.seed(42)
    n_sites = 3
    n_subjects_per_site = 25
    n_regions = 68  # Desikan-Killiany atlas regions

    print(f"Simulating DTI data from {n_sites} sites...")

    all_data = []
    all_site_labels = []
    region_names = [f"Region_{i + 1}" for i in range(n_regions)]

    for site_id in range(n_sites):
        # Each site has different scanner characteristics
        site_bias = np.random.normal(0, 0.05, n_regions)
        site_scale = np.random.uniform(0.95, 1.05, n_regions)

        # Generate realistic FA values (0.2 to 0.8)
        site_fa = np.random.beta(2, 3, (n_subjects_per_site, n_regions)) * 0.6
        site_fa = site_fa * site_scale + site_bias

        # Add some anatomical structure (neighboring regions correlated)
        for subject in range(n_subjects_per_site):
            for region in range(1, n_regions - 1):
                site_fa[subject, region] = (
                    0.6 * site_fa[subject, region]
                    + 0.2 * site_fa[subject, region - 1]
                    + 0.2 * site_fa[subject, region + 1]
                )

        all_data.append(site_fa)
        all_site_labels.extend([site_id] * n_subjects_per_site)

    # Combine all data
    combined_data = np.vstack(all_data)
    site_labels = np.array(all_site_labels)

    print(f"Combined data shape: {combined_data.shape}")
    print(f"Site distribution: {np.bincount(site_labels)}")

    # Apply PCA
    print("\nApplying PCA for dimensionality reduction...")
    pca_model = PCA(numerical_data=combined_data, n_components=20)

    # Transform the data using the model's fit method
    # This is the proper way to use the Model interface
    model = pca_model.generate()
    transformed_data = []
    for i in range(combined_data.shape[0]):
        beta, _ = model.fit(combined_data[i, :])
        transformed_data.append(beta)
    transformed_data = np.array(transformed_data)

    print(f"PCA model created with {len(model.variances)} components")

    # Create comprehensive visualizations
    print("\n1. Creating comprehensive PCA analysis...")
    visualize_pca_analysis(
        pca_model=pca_model,
        original_data=combined_data,
        transformed_data=transformed_data,
        feature_names=region_names,
        n_components_to_show=15,
    )

    print("\n2. Creating detailed component loadings analysis...")
    visualize_pca_loadings(
        pca_model=pca_model,
        feature_names=region_names,
        n_components=6,
        n_features_per_component=8,
    )

    print("\n3. Creating data distribution analysis...")
    visualize_pca_data_distribution(
        original_data=combined_data,
        transformed_data=transformed_data,
        pca_model=pca_model,
        group_labels=site_labels,
    )

    print("\n4. Creating reconstruction and distance preservation analysis...")
    visualize_pca_reconstruction_and_distances(
        pca_model=pca_model,
        original_data=combined_data,
        n_samples_for_distances=50,
    )

    print("\n🎉 PCA Demo completed successfully!")
    print("=" * 50)

    return pca_model, combined_data, transformed_data, site_labels


if __name__ == "__main__":
    # Run the complete demo
    create_pca_demo_with_neuroimaging_data()

# %%
