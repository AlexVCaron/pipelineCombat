#!/usr/bin/env python3
"""
Example script demonstrating the Pipeline Combat toolkit functionality.
"""

import logging

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main example function demonstrating all modules."""

    print("🧠 Pipeline Combat Example")
    print("=" * 50)

    # Import modules
    try:
        from pipelinecombat.diffusion import create_example_dwi_data
        from pipelinecombat.harmonization import create_example_data
        from pipelinecombat.statistics import (
            NeuroStatAnalyzer,
            create_example_behavioral_data,
        )

        print("✅ All modules imported successfully!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please run 'uv sync' to install dependencies")
        return

    # Example 1: Harmonization
    print("\n📊 Example 1: Data Harmonization (Concept Demo)")
    print("-" * 30)

    try:
        # Create example data with site effects
        data, covars = create_example_data(n_subjects=50, n_features=20, n_sites=3)
        print(f"Created data: {data.shape[0]} subjects, {data.shape[1]} features")
        print(f"Sites: {covars['SITE'].unique()}")

        # Show site effects before harmonization
        site_means = covars.groupby("SITE").apply(
            lambda x: data[covars["SITE"] == x.name].mean(axis=0).mean(),
            include_groups=False,
        )
        print(f"Mean values by site (before): {site_means.to_dict()}")

        # Simple demonstration of harmonization concept
        # Note: For real harmonization, use the harmonize_data function with proper setup
        print("Note: This demonstrates site effects. For full harmonization, use:")
        print(
            "  harmonized_data, model = harmonize_data(data, covars, batch_col='SITE')"
        )
        print("  (requires proper data formatting for neuroHarmonize)")

    except Exception as e:
        print(f"Harmonization example failed: {e}")

    # Example 2: Statistical Analysis
    print("\n📈 Example 2: Statistical Analysis")
    print("-" * 30)

    try:
        # Create example brain and behavioral data
        brain_data, behavioral_scores, covariates = create_example_behavioral_data(
            n_subjects=80
        )
        print(f"Brain data: {brain_data.shape}")
        print(f"Behavioral scores: {behavioral_scores.shape}")

        # Initialize analyzer
        analyzer = NeuroStatAnalyzer()

        # Correlation analysis
        corr_results = analyzer.correlation_analysis(
            brain_data, behavioral_scores, method="pearson"
        )

        # Find significant correlations
        significant_corr = np.abs(corr_results["correlations"]) > 0.3
        n_significant = np.sum(significant_corr)
        print(f"Found {n_significant} regions with |r| > 0.3")

        # Group comparison (ANOVA)
        anova_results = analyzer.anova_analysis(brain_data, covariates["group"])

        # Apply multiple comparisons correction
        reject, pvals_corrected, _, _ = analyzer.multiple_comparisons_correction(
            anova_results["pvalues"], method="fdr_bh"
        )

        n_significant_anova = np.sum(reject)
        print(
            f"Found {n_significant_anova} significant group differences (FDR corrected)"
        )

    except Exception as e:
        print(f"Statistical analysis example failed: {e}")

    # Example 3: Diffusion Imaging (synthetic data only)
    print("\n🧠 Example 3: Diffusion Imaging")
    print("-" * 30)

    try:
        # Create synthetic DWI data
        dwi_data, bvals, bvecs = create_example_dwi_data(shape=(30, 30, 10))
        print(f"DWI data shape: {dwi_data.shape}")
        print(f"Number of b-values: {len(bvals)}")
        print(f"Unique b-values: {np.unique(bvals)}")

        print("Note: This is synthetic data. For real analysis, use:")
        print("  processor = DiffusionProcessor()")
        print("  processor.load_data(dwi_path, bvals_path, bvecs_path)")
        print("  fa, md, ad, rd = processor.fit_dti()")

    except Exception as e:
        print(f"Diffusion imaging example failed: {e}")

    # Example 4: Integration Example
    print("\n🔄 Example 4: Integration Workflow")
    print("-" * 30)

    try:
        # Simulate a complete workflow
        print("1. Loading multi-site DTI data...")

        # Simulate DTI metrics from multiple sites
        np.random.seed(123)
        n_subjects_per_site = 20
        n_sites = 3
        n_regions = 68  # Desikan-Killiany atlas

        all_fa_data = []
        all_covars = []

        for site_id in range(n_sites):
            # Simulate site-specific FA values
            site_fa = np.random.beta(2, 3, (n_subjects_per_site, n_regions)) * 0.8
            # Add site bias
            site_fa += np.random.normal(0, 0.05, (1, n_regions))

            site_covars = pd.DataFrame(
                {
                    "SITE": [f"Site_{site_id}"] * n_subjects_per_site,
                    "age": np.random.uniform(20, 70, n_subjects_per_site),
                    "sex": np.random.choice(["M", "F"], n_subjects_per_site),
                }
            )

            all_fa_data.append(site_fa)
            all_covars.append(site_covars)

        # Combine all data
        fa_data = np.vstack(all_fa_data)
        covars_df = pd.concat(all_covars, ignore_index=True)

        print(f"Combined data: {fa_data.shape[0]} subjects, {fa_data.shape[1]} regions")

        print("2. Harmonizing across sites...")
        print("Note: Using statistical concept demo. For real harmonization:")
        print(
            "  harmonized_fa, _ = harmonize_data(fa_data, covars_df, batch_col='SITE')"
        )

        # Demonstrate statistical analysis instead
        harmonized_fa = fa_data  # Use original data for demo

        print("3. Statistical analysis...")
        analyzer = NeuroStatAnalyzer()

        # Age correlation analysis
        age_corr = analyzer.correlation_analysis(
            harmonized_fa, covars_df["age"], method="pearson"
        )

        # Multiple comparisons correction
        reject, _, _, _ = analyzer.multiple_comparisons_correction(
            age_corr["pvalues"], method="fdr_bh", alpha=0.05
        )

        significant_regions = np.sum(reject)
        print(f"Found {significant_regions} regions significantly correlated with age")

        print("4. Workflow completed successfully! 🎉")

    except Exception as e:
        print(f"Integration workflow failed: {e}")

    print("\n" + "=" * 50)
    print("Example completed! Check the modules for more detailed functionality.")
    print(
        "For real data analysis, replace synthetic data with actual neuroimaging files."
    )


if __name__ == "__main__":
    main()
