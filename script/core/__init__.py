"""
NanoStack 3D Core Research Package.
Provides geometry, feature extraction, candidate generation, and model inference modules.
"""

from .geometry import (
    validate_structure,
    wrap_positions_custom,
    unwrap_continuous_molecule,
    min_image_displacement,
    partition_layers_dynamically,
    calculate_layer_centroids,
    VDW_RADII,
    DEFAULT_VDW_CUTOFFS
)

from .features import (
    extract_interface_features,
    compute_n_layer_features
)

from .models import (
    StackingPredictor,
    compute_metrics,
    run_group_aware_cv
)

from .generator import (
    check_clash_pbc_jit,
    get_min_dist_jit,
    atomic_write_cif,
    generate_candidates_for_parent
)

__all__ = [
    "validate_structure",
    "wrap_positions_custom",
    "unwrap_continuous_molecule",
    "min_image_displacement",
    "partition_layers_dynamically",
    "calculate_layer_centroids",
    "VDW_RADII",
    "DEFAULT_VDW_CUTOFFS",
    "extract_interface_features",
    "compute_n_layer_features",
    "StackingPredictor",
    "compute_metrics",
    "run_group_aware_cv",
    "check_clash_pbc_jit",
    "get_min_dist_jit",
    "atomic_write_cif",
    "generate_candidates_for_parent"
]
