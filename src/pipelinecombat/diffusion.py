"""
Diffusion imaging processing module using DIPY for analyzing
diffusion-weighted MRI data.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    from dipy.core.gradients import gradient_table
    from dipy.denoise.localpca import localpca
    from dipy.io.image import load_nifti, save_nifti
    from dipy.reconst.dki import DiffusionKurtosisModel
    from dipy.reconst.dti import TensorModel
    from dipy.segment.mask import median_otsu

    DIPY_AVAILABLE = True
except ImportError:
    logger.warning("DIPY not available. Install with: uv add dipy")
    DIPY_AVAILABLE = False


class DiffusionProcessor:
    """Class for processing diffusion MRI data using DIPY."""

    def __init__(self):
        if not DIPY_AVAILABLE:
            raise ImportError("DIPY is required for diffusion processing")

        self.data = None
        self.affine = None
        self.bvals = None
        self.bvecs = None
        self.gtab = None
        self.mask = None

    def load_data(
        self,
        dwi_path: str | Path,
        bvals_path: str | Path,
        bvecs_path: str | Path,
    ) -> None:
        """
        Load diffusion MRI data and gradient information.

        Args:
            dwi_path: Path to diffusion-weighted images (NIfTI)
            bvals_path: Path to b-values file
            bvecs_path: Path to b-vectors file
        """
        logger.info("Loading diffusion MRI data...")

        # Load DWI data
        self.data, self.affine = load_nifti(str(dwi_path))

        # Load gradient information
        self.bvals = np.loadtxt(str(bvals_path))
        self.bvecs = np.loadtxt(str(bvecs_path)).T

        # Create gradient table
        self.gtab = gradient_table(self.bvals, bvecs=self.bvecs)

        logger.info(f"Loaded data shape: {self.data.shape}")
        logger.info(f"Number of gradients: {len(self.bvals)}")

    def denoise(
        self, sigma: float | None = None
    ) -> tuple[np.ndarray, np.ndarray | int | float]:
        """
        Denoise diffusion data using local PCA.

        Args:
            sigma: Noise standard deviation (estimated if None)

        Returns:
            Tuple of (denoised_data, sigma)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.info("Denoising diffusion data...")

        result = localpca(self.data, sigma=sigma)
        # localpca may return (denoised_data, sigma) or just denoised_data
        if isinstance(result, tuple):
            denoised_data, estimated_sigma = result
        else:
            denoised_data = result
            estimated_sigma = sigma if sigma is not None else 0.0

        logger.info("Denoising completed")
        return denoised_data, estimated_sigma

    def create_mask(
        self, median_radius: int = 4, numpass: int = 1
    ) -> np.ndarray:
        """
        Create brain mask using median Otsu algorithm.

        Args:
            median_radius: Radius for median filtering
            numpass: Number of passes for mask refinement

        Returns:
            Binary brain mask
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if self.gtab is None:
            raise ValueError("No gradient table. Call load_data() first.")

        logger.info("Creating brain mask...")

        # Use mean b=0 image for masking
        b0_mask = self.gtab.b0s_mask
        mean_b0 = np.mean(self.data[..., b0_mask], axis=3)

        _, self.mask = median_otsu(
            mean_b0, median_radius=median_radius, numpass=numpass
        )

        logger.info(f"Mask created with {np.sum(self.mask)} voxels")
        return self.mask

    def fit_dti(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit diffusion tensor model and compute DTI metrics.

        Returns:
            Tuple of (FA, MD, AD, RD) maps
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if self.mask is None:
            logger.warning("No mask provided, creating one...")
            self.create_mask()

        logger.info("Fitting diffusion tensor model...")

        # Fit DTI model
        tenmodel = TensorModel(self.gtab)
        tenfit = tenmodel.fit(self.data, mask=self.mask)

        # Compute DTI metrics
        fa = tenfit.fa
        md = tenfit.md
        ad = tenfit.ad
        rd = tenfit.rd

        # Clean up NaN values
        fa[np.isnan(fa)] = 0
        md[np.isnan(md)] = 0
        ad[np.isnan(ad)] = 0
        rd[np.isnan(rd)] = 0

        logger.info("DTI fitting completed")

        return fa, md, ad, rd

    def fit_dki(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit diffusion kurtosis model and compute DKI metrics.

        Returns:
            Tuple of (MK, AK, RK) maps
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if self.mask is None:
            logger.warning("No mask provided, creating one...")
            self.create_mask()

        logger.info("Fitting diffusion kurtosis model...")

        # Fit DKI model
        dkimodel = DiffusionKurtosisModel(self.gtab)
        dkifit = dkimodel.fit(self.data, mask=self.mask)

        if dkifit is None:
            raise RuntimeError("DKI fitting failed.")

        # Compute DKI metrics
        mk = dkifit.mk()  # Mean kurtosis
        ak = dkifit.ak()  # Axial kurtosis
        rk = dkifit.rk()  # Radial kurtosis

        # Clean up NaN values
        mk[np.isnan(mk)] = 0
        ak[np.isnan(ak)] = 0
        rk[np.isnan(rk)] = 0

        logger.info("DKI fitting completed")

        return mk, ak, rk

    def save_metrics(
        self, metrics: dict, output_dir: str | Path, prefix: str = ""
    ) -> None:
        """
        Save computed metrics to NIfTI files.

        Args:
            metrics: Dictionary of metric name -> data array
            output_dir: Output directory
            prefix: Prefix for output filenames
        """
        if self.affine is None:
            raise ValueError("No affine information available")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, data in metrics.items():
            filename = f"{prefix}{name}.nii.gz" if prefix else f"{name}.nii.gz"
            filepath = output_dir / filename
            save_nifti(str(filepath), data.astype(np.float32), self.affine)
            logger.info(f"Saved {name} to {filepath}")


def create_example_dwi_data(
    shape: tuple[int, int, int] = (50, 50, 20)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create example diffusion MRI data for testing.

    Args:
        shape: Spatial dimensions of the data

    Returns:
        Tuple of (dwi_data, bvals, bvecs)
    """
    np.random.seed(42)

    # Create synthetic DWI data
    n_gradients = 64
    dwi_shape = shape + (n_gradients,)
    dwi_data = np.random.randn(*dwi_shape) * 100 + 1000

    # Create b-values (one b=0, rest at b=1000)
    bvals = np.zeros(n_gradients)
    bvals[1:] = 1000

    # Create random b-vectors
    bvecs = np.random.randn(n_gradients, 3)
    bvecs[0] = [0, 0, 0]  # b=0 direction
    # Normalize non-zero b-vectors
    for i in range(1, n_gradients):
        bvecs[i] = bvecs[i] / np.linalg.norm(bvecs[i])

    logger.info(f"Created example DWI data: {dwi_shape}")

    return dwi_data, bvals, bvecs
