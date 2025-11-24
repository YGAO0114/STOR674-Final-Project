"""
Unit Tests for Statistical Analysis
Tests data loading, analysis functions, and output generation

Author: Statistical Analysis Team
Date: 2025-11-17

Usage:
    # Run all tests
    pytest test_statistical_analysis.py -v

    # Run with coverage
    pytest test_statistical_analysis.py --cov=statistical_analysis --cov-report=html

    # Run specific test
    pytest test_statistical_analysis.py::TestDataLoading::test_load_all_features -v
"""

import unittest
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent))

from statistical_analysis import VascularDevelopmentAnalysis
import config

# Suppress warnings during testing
warnings.filterwarnings('ignore')


class TestConfiguration(unittest.TestCase):
    """Test configuration file and paths"""

    def test_config_paths_exist(self):
        """Test that all configured paths exist"""
        self.assertTrue(config.PROJECT_ROOT.exists(), "Project root should exist")
        self.assertTrue(config.DATA_DIR.exists(), "Data directory should exist")
        self.assertTrue(config.FEATURE_DIR.exists(), "Feature directory should exist")

    def test_time_points_defined(self):
        """Test that time points are properly defined"""
        self.assertEqual(len(config.TIME_POINTS), 7, "Should have 7 time points")
        self.assertIn('p2', config.TIME_POINTS)
        self.assertIn('p7x', config.TIME_POINTS)

    def test_time_point_to_day_mapping(self):
        """Test time point to day conversion"""
        self.assertEqual(config.TIME_POINT_TO_DAY['p2'], 2.0)
        self.assertEqual(config.TIME_POINT_TO_DAY['p7'], 7.0)
        self.assertEqual(config.TIME_POINT_TO_DAY['p7x'], 7.5)

    def test_expected_files_count(self):
        """Test expected file counts"""
        self.assertEqual(config.EXPECTED_FILES_PER_TYPE, 7)
        self.assertEqual(config.EXPECTED_TOTAL_FILES, 35)


class TestDataLoading(unittest.TestCase):
    """Test data loading functionality"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests"""
        cls.analyzer = VascularDevelopmentAnalysis(config.DATA_DIR)

    def test_analyzer_initialization(self):
        """Test that analyzer initializes correctly"""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.data_dir, config.DATA_DIR)
        self.assertTrue(self.analyzer.results_dir.exists())

    def test_load_all_features(self):
        """Test loading all feature files"""
        self.analyzer.load_all_features()

        # Check that all DataFrames are populated
        self.assertEqual(len(self.analyzer.density_df), 7, "Should load 7 density records")
        self.assertEqual(len(self.analyzer.fractal_df), 7, "Should load 7 fractal records")
        self.assertEqual(len(self.analyzer.angle_df), 7, "Should load 7 angle records")
        self.assertEqual(len(self.analyzer.betti_df), 7, "Should load 7 betti records")

    def test_density_data_columns(self):
        """Test that density data has required columns"""
        self.analyzer.load_all_features()

        required_cols = ['time_point', 'day', 'vessel_area_density', 'vessel_length_density']
        for col in required_cols:
            self.assertIn(col, self.analyzer.density_df.columns, f"Missing column: {col}")

    def test_fractal_data_columns(self):
        """Test that fractal data has required columns"""
        self.analyzer.load_all_features()

        required_cols = ['time_point', 'day', 'vessel_fractal_dimension', 'vessel_r_squared']
        for col in required_cols:
            self.assertIn(col, self.analyzer.fractal_df.columns, f"Missing column: {col}")

    def test_angle_data_columns(self):
        """Test that angle data has required columns"""
        self.analyzer.load_all_features()

        required_cols = ['time_point', 'day', 'mean_angle', 'num_branch_points']
        for col in required_cols:
            self.assertIn(col, self.analyzer.angle_df.columns, f"Missing column: {col}")

    def test_betti_data_columns(self):
        """Test that betti data has required columns"""
        self.analyzer.load_all_features()

        required_cols = ['time_point', 'day', 'beta_0', 'beta_1', 'beta_2',
                        'loops_per_component', 'void_area_fraction']
        for col in required_cols:
            self.assertIn(col, self.analyzer.betti_df.columns, f"Missing column: {col}")

    def test_time_points_complete(self):
        """Test that all time points are loaded"""
        self.analyzer.load_all_features()

        loaded_timepoints = set(self.analyzer.density_df['time_point'])
        expected_timepoints = set(config.TIME_POINTS)

        self.assertEqual(loaded_timepoints, expected_timepoints,
                        "All time points should be loaded")

    def test_day_values_correct(self):
        """Test that day values are correctly assigned"""
        self.analyzer.load_all_features()

        for _, row in self.analyzer.density_df.iterrows():
            tp = row['time_point']
            day = row['day']
            expected_day = config.TIME_POINT_TO_DAY[tp]
            self.assertEqual(day, expected_day,
                           f"Day mismatch for {tp}: got {day}, expected {expected_day}")


class TestDataValidation(unittest.TestCase):
    """Test data validation and quality checks"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.analyzer = VascularDevelopmentAnalysis(config.DATA_DIR)
        cls.analyzer.load_all_features()

    def test_vad_range(self):
        """Test that vessel area density is in valid range [0, 1]"""
        vad_values = self.analyzer.density_df['vessel_area_density']
        self.assertTrue((vad_values >= 0).all(), "VAD should be >= 0")
        self.assertTrue((vad_values <= 1).all(), "VAD should be <= 1")

    def test_vld_positive(self):
        """Test that vessel length density is positive"""
        vld_values = self.analyzer.density_df['vessel_length_density']
        self.assertTrue((vld_values > 0).all(), "VLD should be positive")

    def test_fractal_dimension_range(self):
        """Test that fractal dimension is in valid range [1, 2]"""
        fd_values = self.analyzer.fractal_df['vessel_fractal_dimension']
        self.assertTrue((fd_values >= 1).all(), "FD should be >= 1")
        self.assertTrue((fd_values <= 2).all(), "FD should be <= 2")

    def test_r_squared_range(self):
        """Test that R² is in valid range [0, 1]"""
        r2_values = self.analyzer.fractal_df['vessel_r_squared']
        self.assertTrue((r2_values >= 0).all(), "R² should be >= 0")
        self.assertTrue((r2_values <= 1).all(), "R² should be <= 1")

    def test_angles_range(self):
        """Test that angles are in valid range [0, 180]"""
        angle_values = self.analyzer.angle_df['mean_angle']
        self.assertTrue((angle_values >= 0).all(), "Angles should be >= 0")
        self.assertTrue((angle_values <= 180).all(), "Angles should be <= 180")

    def test_betti_numbers_positive(self):
        """Test that Betti numbers are non-negative"""
        beta_0 = self.analyzer.betti_df['beta_0']
        beta_1 = self.analyzer.betti_df['beta_1']
        beta_2 = self.analyzer.betti_df['beta_2']

        self.assertTrue((beta_0 >= 0).all(), "β₀ should be non-negative")
        self.assertTrue((beta_1 >= 0).all(), "β₁ should be non-negative")
        self.assertTrue((beta_2 >= 0).all(), "β₂ should be non-negative")

    def test_void_fraction_range(self):
        """Test that void fraction is in valid range [0, 1]"""
        void_values = self.analyzer.betti_df['void_area_fraction']
        self.assertTrue((void_values >= 0).all(), "Void fraction should be >= 0")
        self.assertTrue((void_values <= 1).all(), "Void fraction should be <= 1")

    def test_no_missing_values(self):
        """Test that there are no missing values in key columns"""
        self.assertFalse(self.analyzer.density_df['vessel_area_density'].isna().any(),
                        "VAD should not have missing values")
        self.assertFalse(self.analyzer.fractal_df['vessel_fractal_dimension'].isna().any(),
                        "FD should not have missing values")


class TestAnalysisFunctions(unittest.TestCase):
    """Test statistical analysis functions"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.analyzer = VascularDevelopmentAnalysis(config.DATA_DIR)
        cls.analyzer.load_all_features()

    def test_create_summary_table(self):
        """Test summary table creation"""
        summary = self.analyzer.create_summary_table()

        # Check shape
        self.assertEqual(len(summary), 7, "Summary should have 7 rows")

        # Check required columns
        self.assertIn('time_point', summary.columns)
        self.assertIn('day', summary.columns)
        self.assertIn('vessel_area_density', summary.columns)
        self.assertIn('vessel_fractal_dimension', summary.columns)

    def test_perform_trend_analysis(self):
        """Test trend analysis"""
        self.analyzer.create_summary_table()  # Need summary first
        trends = self.analyzer.perform_trend_analysis()

        # Check that we have results
        self.assertIsNotNone(trends)
        self.assertGreater(len(trends), 0, "Should have trend results")

        # Check required columns
        required_cols = ['Metric', 'Slope', 'R²', 'P-value', 'Trend', 'Significant']
        for col in required_cols:
            self.assertIn(col, trends.columns, f"Missing column: {col}")

    def test_trend_analysis_metrics(self):
        """Test that trend analysis includes expected metrics"""
        self.analyzer.create_summary_table()
        trends = self.analyzer.perform_trend_analysis()

        metrics = trends['Metric'].tolist()
        expected_metrics = ['vessel_area_density', 'vessel_fractal_dimension',
                          'mean_angle', 'beta_1']

        for metric in expected_metrics:
            self.assertIn(metric, metrics, f"Missing metric: {metric}")

    def test_r_squared_valid(self):
        """Test that R² values are valid"""
        self.analyzer.create_summary_table()
        trends = self.analyzer.perform_trend_analysis()

        r2_values = trends['R²']
        self.assertTrue((r2_values >= 0).all(), "R² should be >= 0")
        self.assertTrue((r2_values <= 1).all(), "R² should be <= 1")

    def test_p_values_valid(self):
        """Test that p-values are valid"""
        self.analyzer.create_summary_table()
        trends = self.analyzer.perform_trend_analysis()

        p_values = trends['P-value']
        self.assertTrue((p_values >= 0).all(), "P-value should be >= 0")
        self.assertTrue((p_values <= 1).all(), "P-value should be <= 1")


class TestOutputGeneration(unittest.TestCase):
    """Test output file generation"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.analyzer = VascularDevelopmentAnalysis(config.DATA_DIR)

    def test_results_directory_created(self):
        """Test that results directory is created"""
        self.assertTrue(self.analyzer.results_dir.exists(),
                       "Results directory should be created")

    def test_summary_table_saved(self):
        """Test that summary table is saved"""
        self.analyzer.load_all_features()
        self.analyzer.create_summary_table()

        output_file = config.get_output_file_path('summary_table')
        self.assertTrue(output_file.exists(),
                       f"Summary table should be saved: {output_file}")

    def test_trend_analysis_saved(self):
        """Test that trend analysis is saved"""
        self.analyzer.load_all_features()
        self.analyzer.create_summary_table()
        self.analyzer.perform_trend_analysis()

        output_file = config.get_output_file_path('trend_analysis')
        self.assertTrue(output_file.exists(),
                       f"Trend analysis should be saved: {output_file}")

    def test_correlation_matrix_saved(self):
        """Test that correlation matrix is saved"""
        self.analyzer.load_all_features()
        self.analyzer.create_summary_table()
        self.analyzer.create_correlation_matrix()

        output_file = config.get_output_file_path('correlation_matrix')
        self.assertTrue(output_file.exists(),
                       f"Correlation matrix should be saved: {output_file}")


class TestReproducibility(unittest.TestCase):
    """Test reproducibility of results"""

    def test_deterministic_loading(self):
        """Test that loading data twice gives same results"""
        analyzer1 = VascularDevelopmentAnalysis(config.DATA_DIR)
        analyzer1.load_all_features()

        analyzer2 = VascularDevelopmentAnalysis(config.DATA_DIR)
        analyzer2.load_all_features()

        # Compare DataFrames
        pd.testing.assert_frame_equal(analyzer1.density_df, analyzer2.density_df)
        pd.testing.assert_frame_equal(analyzer1.fractal_df, analyzer2.fractal_df)

    def test_deterministic_summary(self):
        """Test that summary table is deterministic"""
        analyzer1 = VascularDevelopmentAnalysis(config.DATA_DIR)
        analyzer1.load_all_features()
        summary1 = analyzer1.create_summary_table()

        analyzer2 = VascularDevelopmentAnalysis(config.DATA_DIR)
        analyzer2.load_all_features()
        summary2 = analyzer2.create_summary_table()

        pd.testing.assert_frame_equal(summary1, summary2)

    def test_deterministic_trends(self):
        """Test that trend analysis is deterministic"""
        analyzer1 = VascularDevelopmentAnalysis(config.DATA_DIR)
        analyzer1.load_all_features()
        analyzer1.create_summary_table()
        trends1 = analyzer1.perform_trend_analysis()

        analyzer2 = VascularDevelopmentAnalysis(config.DATA_DIR)
        analyzer2.load_all_features()
        analyzer2.create_summary_table()
        trends2 = analyzer2.perform_trend_analysis()

        # Compare numeric columns with tolerance
        for col in ['Slope', 'R²', 'P-value']:
            np.testing.assert_allclose(trends1[col], trends2[col], rtol=1e-10,
                                      err_msg=f"{col} should be deterministic")


class TestBiologicalInterpretation(unittest.TestCase):
    """Test biological interpretation and expected trends"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.analyzer = VascularDevelopmentAnalysis(config.DATA_DIR)
        cls.analyzer.load_all_features()
        cls.analyzer.create_summary_table()

    def test_vad_increasing_trend(self):
        """Test that VAD increases from P2 to P7 (angiogenesis)"""
        vad_p2 = self.analyzer.density_df[
            self.analyzer.density_df['time_point'] == 'p2'
        ]['vessel_area_density'].values[0]

        vad_p7x = self.analyzer.density_df[
            self.analyzer.density_df['time_point'] == 'p7x'
        ]['vessel_area_density'].values[0]

        self.assertGreater(vad_p7x, vad_p2,
                          "VAD should increase from P2 to P7 (angiogenesis)")

    def test_fractal_dimension_increases(self):
        """Test that fractal dimension increases (network complexity)"""
        fd_p2 = self.analyzer.fractal_df[
            self.analyzer.fractal_df['time_point'] == 'p2'
        ]['vessel_fractal_dimension'].values[0]

        fd_p7x = self.analyzer.fractal_df[
            self.analyzer.fractal_df['time_point'] == 'p7x'
        ]['vessel_fractal_dimension'].values[0]

        self.assertGreater(fd_p7x, fd_p2,
                          "FD should increase from P2 to P7 (complexity)")

    def test_loops_increase(self):
        """Test that number of loops increases (meshwork formation)"""
        beta1_p2 = self.analyzer.betti_df[
            self.analyzer.betti_df['time_point'] == 'p2'
        ]['beta_1'].values[0]

        beta1_p7x = self.analyzer.betti_df[
            self.analyzer.betti_df['time_point'] == 'p7x'
        ]['beta_1'].values[0]

        self.assertGreater(beta1_p7x, beta1_p2,
                          "Loops (β₁) should increase from P2 to P7")

    def test_r_squared_quality(self):
        """Test that fractal dimension fits are high quality"""
        r2_values = self.analyzer.fractal_df['vessel_r_squared']
        self.assertTrue((r2_values > 0.95).all(),
                       "Fractal R² should be > 0.95 (good power-law fit)")


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

class TestFullPipeline(unittest.TestCase):
    """Integration test for full analysis pipeline"""

    def test_full_pipeline_runs(self):
        """Test that full pipeline runs without errors"""
        analyzer = VascularDevelopmentAnalysis(config.DATA_DIR)

        # This should run without raising exceptions
        try:
            analyzer.load_all_features()
            analyzer.create_summary_table()
            analyzer.perform_trend_analysis()
            analyzer.create_correlation_matrix()
            analyzer.generate_report()
        except Exception as e:
            self.fail(f"Full pipeline failed with exception: {e}")

    def test_all_outputs_generated(self):
        """Test that all expected outputs are generated"""
        analyzer = VascularDevelopmentAnalysis(config.DATA_DIR)
        analyzer.run_full_analysis()

        # Check all expected output files exist
        for output_type in ['summary_table', 'trend_analysis',
                           'correlation_matrix', 'correlations', 'analysis_report']:
            output_file = config.get_output_file_path(output_type)
            self.assertTrue(output_file.exists(),
                           f"Output file should exist: {output_file}")


# ==============================================================================
# TEST RUNNER
# ==============================================================================

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
