"""
Statistical Analysis of Vascular Development (P2-P7)

This script performs statistical analysis on extracted vascular features
from mouse retinal vessels across postnatal days P2-P7.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.stats import f_oneway, kruskal, mannwhitneyu, shapiro, levene
import warnings
warnings.filterwarnings('ignore')

# Set visualization parameters
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')


class VascularDevelopmentAnalysis:
    """
    Comprehensive statistical analysis of vascular development
    across postnatal days P2-P7
    """

    def __init__(self, data_dir):
        """
        Initialize the analysis with data directory

        Parameters:
        -----------
        data_dir : str or Path
            Path to the feature extraction directory containing p2-p7 folder
        """
        self.data_dir = Path(data_dir)
        self.feature_dir = self.data_dir / 'feature' / 'p2-p7'
        self.results_dir = Path('results/statistical_analysis')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Time points
        self.time_points = ['p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p7x']

        # Initialize data containers
        self.density_data = []
        self.fractal_data = []
        self.angle_data = []
        self.betti_data = []
        self.regional_data = []

        print("="*70)
        print("Vascular Development Statistical Analysis")
        print("="*70)
        print(f"Data directory: {self.data_dir}")
        print(f"Feature directory: {self.feature_dir}")
        print(f"Results directory: {self.results_dir}")
        print(f"Time points: {', '.join(self.time_points)}")
        print("="*70)

    def load_all_features(self):
        """Load all feature files for P2-P7"""
        print("\n[1] Loading feature data...")

        for tp in self.time_points:
            # Vascular Density
            density_file = self.feature_dir / f'vessel_{tp}_vascular_density.xlsx'
            if density_file.exists():
                df = pd.read_excel(density_file, sheet_name='Global')
                df['time_point'] = tp
                df['day'] = int(tp[1]) if tp != 'p7x' else 7.5
                self.density_data.append(df)

            # Fractal Dimension
            fractal_file = self.feature_dir / f'vessel_{tp}_fractal_dimension.xlsx'
            if fractal_file.exists():
                df = pd.read_excel(fractal_file)
                df['time_point'] = tp
                df['day'] = int(tp[1]) if tp != 'p7x' else 7.5
                self.fractal_data.append(df)

            # Angle Statistics
            angle_file = self.feature_dir / f'vessel_{tp}_angle_statistics.xlsx'
            if angle_file.exists():
                df = pd.read_excel(angle_file, sheet_name='Statistics')
                df['time_point'] = tp
                df['day'] = int(tp[1]) if tp != 'p7x' else 7.5
                self.angle_data.append(df)

            # Betti Numbers
            betti_file = self.feature_dir / f'vessel_{tp}_betti_numbers.xlsx'
            if betti_file.exists():
                df = pd.read_excel(betti_file, sheet_name='Summary')
                df['time_point'] = tp
                df['day'] = int(tp[1]) if tp != 'p7x' else 7.5
                self.betti_data.append(df)

            # Regional Area (sheet is 'Sheet1', not 'Summary')
            regional_file = self.feature_dir / f'vessel_{tp}_regional_area.xlsx'
            if regional_file.exists():
                df = pd.read_excel(regional_file, sheet_name='Sheet1')
                df['time_point'] = tp
                df['day'] = int(tp[1]) if tp != 'p7x' else 7.5
                self.regional_data.append(df)

        # Combine data
        self.density_df = pd.concat(self.density_data, ignore_index=True)
        self.fractal_df = pd.concat(self.fractal_data, ignore_index=True)
        self.angle_df = pd.concat(self.angle_data, ignore_index=True)
        self.betti_df = pd.concat(self.betti_data, ignore_index=True)
        self.regional_df = pd.concat(self.regional_data, ignore_index=True)

        print(f"  ✓ Loaded {len(self.density_df)} density records")
        print(f"  ✓ Loaded {len(self.fractal_df)} fractal records")
        print(f"  ✓ Loaded {len(self.angle_df)} angle records")
        print(f"  ✓ Loaded {len(self.betti_df)} betti records")
        print(f"  ✓ Loaded {len(self.regional_df)} regional records")

        return self

    def create_summary_table(self):
        """Create comprehensive summary statistics table"""
        print("\n[2] Creating summary statistics table...")

        summary = pd.DataFrame({
            'time_point': self.time_points,
            'day': [int(tp[1]) if tp != 'p7x' else 7.5 for tp in self.time_points]
        })

        # Add density metrics
        if not self.density_df.empty:
            cols_to_merge = ['time_point', 'vessel_area_density', 'vessel_length_density']
            # Add regional_asymmetry only if it exists
            if 'regional_asymmetry' in self.density_df.columns:
                cols_to_merge.append('regional_asymmetry')
            summary = summary.merge(
                self.density_df[cols_to_merge],
                on='time_point', how='left'
            )

        # Add fractal dimension
        if not self.fractal_df.empty:
            summary = summary.merge(
                self.fractal_df[['time_point', 'vessel_fractal_dimension', 'skeleton_fractal_dimension', 'vessel_r_squared']],
                on='time_point', how='left'
            )

        # Add angle statistics
        if not self.angle_df.empty:
            summary = summary.merge(
                self.angle_df[['time_point', 'mean_angle', 'median_angle', 'std_angle', 'num_branch_points']],
                on='time_point', how='left'
            )

        # Add betti numbers
        if not self.betti_df.empty:
            summary = summary.merge(
                self.betti_df[['time_point', 'beta_0', 'beta_1', 'beta_2', 'loops_per_component', 'void_area_fraction']],
                on='time_point', how='left'
            )

        self.summary_df = summary

        # Save to Excel
        output_file = self.results_dir / 'summary_statistics.xlsx'
        summary.to_excel(output_file, index=False)
        print(f"  ✓ Summary table saved to: {output_file}")

        # Display table
        print("\n" + "="*70)
        print("SUMMARY STATISTICS TABLE")
        print("="*70)
        print(summary.to_string(index=False))
        print("="*70)

        return summary

    def plot_temporal_trends(self):
        """Plot temporal trends of key vascular features"""
        print("\n[3] Generating temporal trend plots...")

        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle('Vascular Development: Temporal Trends (P2-P7)',
                     fontsize=16, fontweight='bold', y=0.995)

        # 1. Vessel Area Density
        ax = axes[0, 0]
        if not self.density_df.empty:
            ax.plot(self.density_df['day'], self.density_df['vessel_area_density'],
                   'o-', linewidth=2, markersize=8, label='VAD')
            ax.set_xlabel('Postnatal Day')
            ax.set_ylabel('Vessel Area Density (VAD)')
            ax.set_title('Vessel Coverage Over Time')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # 2. Vessel Length Density
        ax = axes[0, 1]
        if not self.density_df.empty:
            ax.plot(self.density_df['day'], self.density_df['vessel_length_density'],
                   'o-', linewidth=2, markersize=8, color='green', label='VLD')
            ax.set_xlabel('Postnatal Day')
            ax.set_ylabel('Vessel Length Density (VLD)')
            ax.set_title('Vessel Length Density Over Time')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # 3. Fractal Dimension
        ax = axes[0, 2]
        if not self.fractal_df.empty:
            ax.plot(self.fractal_df['day'], self.fractal_df['vessel_fractal_dimension'],
                   'o-', linewidth=2, markersize=8, color='red', label='Vessel FD')
            ax.plot(self.fractal_df['day'], self.fractal_df['skeleton_fractal_dimension'],
                   's--', linewidth=2, markersize=8, color='orange', label='Skeleton FD')
            ax.set_xlabel('Postnatal Day')
            ax.set_ylabel('Fractal Dimension')
            ax.set_title('Complexity Over Time')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # 4. Branching Angle
        ax = axes[1, 0]
        if not self.angle_df.empty:
            ax.plot(self.angle_df['day'], self.angle_df['mean_angle'],
                   'o-', linewidth=2, markersize=8, color='purple', label='Mean Angle')
            ax.axhline(y=37.5, color='gray', linestyle='--', label="Murray's Law (37.5°)")
            ax.set_xlabel('Postnatal Day')
            ax.set_ylabel('Mean Branching Angle (degrees)')
            ax.set_title('Branching Optimization Over Time')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # 5. Beta 0 (Components)
        ax = axes[1, 1]
        if not self.betti_df.empty:
            ax.plot(self.betti_df['day'], self.betti_df['beta_0'],
                   'o-', linewidth=2, markersize=8, color='brown', label='β₀')
            ax.set_xlabel('Postnatal Day')
            ax.set_ylabel('β₀ (Connected Components)')
            ax.set_title('Network Fragmentation Over Time')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # 6. Beta 1 (Loops)
        ax = axes[1, 2]
        if not self.betti_df.empty:
            ax.plot(self.betti_df['day'], self.betti_df['beta_1'],
                   'o-', linewidth=2, markersize=8, color='teal', label='β₁')
            ax.set_xlabel('Postnatal Day')
            ax.set_ylabel('β₁ (Loops)')
            ax.set_title('Network Loops Over Time')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # 7. Loops per Component
        ax = axes[2, 0]
        if not self.betti_df.empty:
            ax.plot(self.betti_df['day'], self.betti_df['loops_per_component'],
                   'o-', linewidth=2, markersize=8, color='magenta', label='Loops/Component')
            ax.set_xlabel('Postnatal Day')
            ax.set_ylabel('Loops per Component')
            ax.set_title('Network Redundancy Over Time')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # 8. Void Area Fraction
        ax = axes[2, 1]
        if not self.betti_df.empty:
            ax.plot(self.betti_df['day'], self.betti_df['void_area_fraction'],
                   'o-', linewidth=2, markersize=8, color='navy', label='Void Fraction')
            ax.set_xlabel('Postnatal Day')
            ax.set_ylabel('Void Area Fraction')
            ax.set_title('Avascular Regions Over Time')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # 9. Number of Branch Points
        ax = axes[2, 2]
        if not self.angle_df.empty:
            ax.plot(self.angle_df['day'], self.angle_df['num_branch_points'],
                   'o-', linewidth=2, markersize=8, color='darkgreen', label='Branch Points')
            ax.set_xlabel('Postnatal Day')
            ax.set_ylabel('Number of Branch Points')
            ax.set_title('Branching Complexity Over Time')
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        output_file = self.results_dir / 'temporal_trends.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Temporal trends plot saved to: {output_file}")
        plt.show()
        plt.close()

    def perform_trend_analysis(self):
        """Perform statistical tests for temporal trends"""
        print("\n[4] Performing trend analysis...")

        results = []

        # Define metrics to test
        metrics = [
            ('vessel_area_density', self.density_df),
            ('vessel_length_density', self.density_df),
            ('vessel_fractal_dimension', self.fractal_df),
            ('mean_angle', self.angle_df),
            ('beta_0', self.betti_df),
            ('beta_1', self.betti_df),
            ('loops_per_component', self.betti_df),
            ('void_area_fraction', self.betti_df)
        ]

        for metric_name, df in metrics:
            if df.empty or metric_name not in df.columns:
                continue

            # Linear regression for trend
            x = df['day'].values
            y = df[metric_name].values

            if len(x) > 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                results.append({
                    'Metric': metric_name,
                    'Slope': slope,
                    'Intercept': intercept,
                    'R²': r_value**2,
                    'P-value': p_value,
                    'Trend': 'Increasing' if slope > 0 else 'Decreasing',
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                })

        trend_df = pd.DataFrame(results)

        # Save results
        output_file = self.results_dir / 'trend_analysis.xlsx'
        trend_df.to_excel(output_file, index=False)
        print(f"  ✓ Trend analysis saved to: {output_file}")

        # Display results
        print("\n" + "="*70)
        print("TREND ANALYSIS RESULTS")
        print("="*70)
        print(trend_df.to_string(index=False))
        print("="*70)

        return trend_df

    def create_correlation_matrix(self):
        """Create correlation matrix of key features"""
        print("\n[5] Creating correlation matrix...")

        # Combine key features
        combined = self.summary_df.copy()

        # Select numeric columns for correlation
        numeric_cols = combined.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'day']

        if len(numeric_cols) > 1:
            corr_matrix = combined[numeric_cols].corr()

            # Plot correlation matrix
            fig, ax = plt.subplots(figsize=(14, 12))
            im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Correlation', rotation=270, labelpad=20)

            # Add annotations
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix.columns)):
                    text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)

            # Set ticks and labels
            ax.set_xticks(np.arange(len(corr_matrix.columns)))
            ax.set_yticks(np.arange(len(corr_matrix.index)))
            ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr_matrix.index)

            plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
            plt.tight_layout()

            output_file = self.results_dir / 'correlation_matrix.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  ✓ Correlation matrix saved to: {output_file}")
            plt.show()
            plt.close()

            # Save correlation values
            corr_file = self.results_dir / 'correlations.xlsx'
            corr_matrix.to_excel(corr_file)
            print(f"  ✓ Correlation values saved to: {corr_file}")

    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n[6] Generating comprehensive report...")

        report = []
        report.append("="*70)
        report.append("STATISTICAL ANALYSIS REPORT")
        report.append("Vascular Development Analysis (P2-P7)")
        report.append("="*70)
        report.append("")

        # Summary statistics
        report.append("1. SUMMARY STATISTICS")
        report.append("-"*70)
        report.append(self.summary_df.to_string(index=False))
        report.append("")

        # Key findings
        report.append("2. KEY FINDINGS")
        report.append("-"*70)

        if not self.density_df.empty:
            vad_change = ((self.density_df.iloc[-1]['vessel_area_density'] -
                          self.density_df.iloc[0]['vessel_area_density']) /
                          self.density_df.iloc[0]['vessel_area_density'] * 100)
            report.append(f"• Vessel Area Density: {vad_change:+.1f}% change from P2 to P7")

        if not self.fractal_df.empty:
            fd_change = ((self.fractal_df.iloc[-1]['vessel_fractal_dimension'] -
                         self.fractal_df.iloc[0]['vessel_fractal_dimension']) /
                         self.fractal_df.iloc[0]['vessel_fractal_dimension'] * 100)
            report.append(f"• Fractal Dimension: {fd_change:+.1f}% change from P2 to P7")

        if not self.betti_df.empty:
            loops_change = ((self.betti_df.iloc[-1]['beta_1'] -
                            self.betti_df.iloc[0]['beta_1']) /
                            self.betti_df.iloc[0]['beta_1'] * 100)
            report.append(f"• Network Loops (β₁): {loops_change:+.1f}% change from P2 to P7")

        report.append("")

        # Save report
        report_text = "\n".join(report)
        output_file = self.results_dir / 'analysis_report.txt'
        with open(output_file, 'w') as f:
            f.write(report_text)

        print(report_text)
        print(f"\n  ✓ Full report saved to: {output_file}")

    def run_full_analysis(self):
        """Run complete statistical analysis pipeline"""
        print("\n" + "="*70)
        print("RUNNING FULL STATISTICAL ANALYSIS PIPELINE")
        print("="*70)

        self.load_all_features()
        self.create_summary_table()
        self.plot_temporal_trends()
        self.perform_trend_analysis()
        self.create_correlation_matrix()
        self.generate_report()

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"All results saved to: {self.results_dir}")
        print("="*70)


def main():
    """Main execution function"""
    # Set data directory
    data_dir = Path('Data/validation_data/feature_extraction')

    # Create analyzer
    analyzer = VascularDevelopmentAnalysis(data_dir)

    # Run analysis
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
