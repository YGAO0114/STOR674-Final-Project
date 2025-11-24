"""
Configuration file for Statistical Analysis
Centralizes all parameters, paths, and constants for reproducibility
"""

from pathlib import Path

# ==============================================================================
# PROJECT PATHS
# ==============================================================================

# Project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / 'Data' / 'validation_data' / 'feature_extraction'
FEATURE_DIR = DATA_DIR / 'feature' / 'p2-p7'

# Results directory
RESULTS_DIR = PROJECT_ROOT / 'results' / 'statistical_analysis'

# ==============================================================================
# TIME POINTS
# ==============================================================================

# Developmental time points to analyze
TIME_POINTS = ['p2', 'p3', 'p4', 'p5', 'p6', 'p7']

# Mapping of time point names to numeric days
TIME_POINT_TO_DAY = {
    'p2': 2.0,
    'p3': 3.0,
    'p4': 4.0,
    'p5': 5.0,
    'p6': 6.0,
    'p7': 7.0
}

# ==============================================================================
# FEATURE FILE CONFIGURATIONS
# ==============================================================================

# Feature file patterns
FEATURE_FILES = {
    'density': 'vessel_{tp}_vascular_density.xlsx',
    'fractal': 'vessel_{tp}_fractal_dimension.xlsx',
    'angle': 'vessel_{tp}_angle_statistics.xlsx',
    'betti': 'vessel_{tp}_betti_numbers.xlsx',
    'regional': 'vessel_{tp}_regional_area.xlsx',
    'alldata': 'vessel_{tp}_alldata.xlsx',
    'degreedata': 'vessel_{tp}_degreedata.xlsx'
}

# Excel sheet names for each feature type
FEATURE_SHEETS = {
    'density': 'Global',      # Vascular density has 'Global' and 'Regional' sheets
    'fractal': None,          # Single sheet (default)
    'angle': 'Statistics',    # Angle has 'Statistics' and 'Detailed' sheets
    'betti': 'Summary',       # Betti has 'Summary' and 'Voids' sheets
    'regional': None,         # Single sheet (default)
    'alldata': None,          # Single sheet (default)
    'degreedata': None        # Single sheet (default)
}

# ==============================================================================
# STATISTICAL PARAMETERS
# ==============================================================================

# Significance level for hypothesis testing
ALPHA = 0.05

# Minimum data points required for regression
MIN_POINTS_FOR_REGRESSION = 3

# ==============================================================================
# VISUALIZATION PARAMETERS
# ==============================================================================

# Figure size for temporal trends plot (width, height in inches)
TEMPORAL_TRENDS_FIGSIZE = (18, 14)

# Figure size for correlation matrix (width, height in inches)
CORRELATION_FIGSIZE = (14, 12)

# Plot resolution (dots per inch)
PLOT_DPI = 300

# Plot style
PLOT_STYLE = 'seaborn-v0_8-whitegrid'  # Falls back to 'default' if unavailable

# Font size for plots
PLOT_FONT_SIZE = 11

# Colors for different metrics (optional, can customize)
METRIC_COLORS = {
    'vad': 'blue',
    'vld': 'green',
    'vessel_fd': 'red',
    'skeleton_fd': 'orange',
    'mean_angle': 'purple',
    'beta_0': 'brown',
    'beta_1': 'teal',
    'loops_per_component': 'magenta',
    'void_fraction': 'navy',
    'branch_points': 'darkgreen',
    'tortuosity': 'crimson',
    'vessel_diameter': 'darkblue',
    'segment_count': 'darkorange'
}

# Murray's Law optimal branching angle (degrees)
MURRAY_LAW_ANGLE = 37.5

# ==============================================================================
# METRICS TO ANALYZE
# ==============================================================================

# Metrics to extract from each feature type
METRICS = {
    'density': [
        'vessel_area_density',
        'vessel_length_density'
    ],
    'fractal': [
        'vessel_fractal_dimension',
        'skeleton_fractal_dimension',
        'vessel_r_squared'
    ],
    'angle': [
        'mean_angle',
        'median_angle',
        'std_angle',
        'num_branch_points'
    ],
    'betti': [
        'beta_0',
        'beta_1',
        'beta_2',
        'loops_per_component',
        'loops_per_component',
        'void_area_fraction'
    ],
    'alldata': [
        'tortuosity',
        'vessel_diameter',
        'segment_count'
    ]
}

# Metrics for trend analysis (subset of all metrics)
TREND_METRICS = [
    'vessel_area_density',
    'vessel_length_density',
    'vessel_fractal_dimension',
    'mean_angle',
    'beta_0',
    'beta_1',
    'loops_per_component',
    'beta_1',
    'loops_per_component',
    'void_area_fraction',
    'tortuosity',
    'vessel_diameter',
    'segment_count'
]

# ==============================================================================
# OUTPUT FILES
# ==============================================================================

OUTPUT_FILES = {
    'summary_table': 'summary_statistics.xlsx',
    'temporal_trends': 'temporal_trends.png',
    'trend_analysis': 'trend_analysis.xlsx',
    'correlation_matrix': 'correlation_matrix.png',
    'correlations': 'correlations.xlsx',
    'correlation_matrix': 'correlation_matrix.png',
    'correlations': 'correlations.xlsx',
    'analysis_report': 'analysis_report.txt',
    'pca_plot': 'pca_analysis.png'
}

# ==============================================================================
# VALIDATION PARAMETERS
# ==============================================================================

# Expected number of files per feature type
EXPECTED_FILES_PER_TYPE = len(TIME_POINTS)  # Should be 7

# Expected number of total feature files
EXPECTED_TOTAL_FILES = len(FEATURE_FILES) * EXPECTED_FILES_PER_TYPE  # Should be 35

# Valid ranges for metrics (for data validation)
METRIC_RANGES = {
    'vessel_area_density': (0.0, 1.0),     # Should be 0-100% (as decimal)
    'vessel_length_density': (0.0, 0.1),   # Reasonable upper bound
    'vessel_fractal_dimension': (1.0, 2.0), # Fractal dimension range
    'mean_angle': (0.0, 180.0),            # Degrees
    'beta_0': (0, 10000),                  # Reasonable range for components
    'beta_1': (0, 10000),                  # Reasonable range for loops
    'void_area_fraction': (0.0, 1.0),      # Should be 0-100% (as decimal)
    'tortuosity': (1.0, 10.0),             # Ratio >= 1.0
    'vessel_diameter': (0.0, 1000.0)       # Pixels^2 (area)
}

# ==============================================================================
# LOGGING & DISPLAY
# ==============================================================================

# Whether to display plots interactively
SHOW_PLOTS = True

# Whether to print verbose output
VERBOSE = True

# Separator length for console output
SEPARATOR_LENGTH = 70

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_feature_file_path(time_point: str, feature_type: str) -> Path:
    """
    Get the full path to a feature file

    Parameters:
    -----------
    time_point : str
        Time point identifier (e.g., 'p2', 'p7x')
    feature_type : str
        Feature type ('density', 'fractal', 'angle', 'betti', 'regional')

    Returns:
    --------
    Path : Full path to the feature file
    """
    if feature_type not in FEATURE_FILES:
        raise ValueError(f"Unknown feature type: {feature_type}")

    filename = FEATURE_FILES[feature_type].format(tp=time_point)
    return FEATURE_DIR / filename


def get_output_file_path(output_type: str) -> Path:
    """
    Get the full path to an output file

    Parameters:
    -----------
    output_type : str
        Output type (e.g., 'summary_table', 'temporal_trends')

    Returns:
    --------
    Path : Full path to the output file
    """
    if output_type not in OUTPUT_FILES:
        raise ValueError(f"Unknown output type: {output_type}")

    return RESULTS_DIR / OUTPUT_FILES[output_type]


def validate_metric_value(metric_name: str, value: float) -> bool:
    """
    Validate that a metric value is within expected range

    Parameters:
    -----------
    metric_name : str
        Name of the metric
    value : float
        Value to validate

    Returns:
    --------
    bool : True if value is valid, False otherwise
    """
    if metric_name not in METRIC_RANGES:
        return True  # No validation range defined

    min_val, max_val = METRIC_RANGES[metric_name]
    return min_val <= value <= max_val


# ==============================================================================
# CONFIGURATION VALIDATION
# ==============================================================================

def validate_config():
    """
    Validate that all configured paths exist

    Raises:
    -------
    FileNotFoundError : If required directories don't exist
    """
    # Check data directory exists
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    # Check feature directory exists
    if not FEATURE_DIR.exists():
        raise FileNotFoundError(f"Feature directory not found: {FEATURE_DIR}")

    # Create results directory if it doesn't exist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"✓ Configuration validated successfully")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Feature directory: {FEATURE_DIR}")
    print(f"  Results directory: {RESULTS_DIR}")


# Run validation when module is imported (optional)
if __name__ == '__main__':
    validate_config()
