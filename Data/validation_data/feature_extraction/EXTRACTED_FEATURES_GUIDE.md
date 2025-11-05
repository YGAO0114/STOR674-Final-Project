# Complete Guide to Extracted Vascular Network Features

## Overview

The enhanced feature extraction system analyzes vascular networks from multiple perspectives:

- **Morphological**: Physical structure (size, shape, density)
- **Geometric**: Branching patterns and angles
- **Topological**: Network connectivity and loops
- **Fractal**: Multi-scale complexity

**Input:** Binary vessel segmentation images (TIF format)  
**Output:** 8 files per image (7 Excel files + 1 visualization)  
**Processing time:** ~25 seconds per image

---

## Output Files Summary

For each input image `IMAGE.tif`, the system generates:

| File | Description | Key Metrics |
|------|-------------|-------------|
| `IMAGE_alldata.xlsx` | Enhanced edge features | Mean area, perimeter, length per vessel segment |
| `IMAGE_degreedata.xlsx` | Enhanced node features | Degree, coordinates per junction/endpoint |
| `IMAGE_network.png` | Network visualization | Color-coded graph structure |
| `IMAGE_regional_area.xlsx` | Regional analysis | Area distribution across image regions |
| `IMAGE_angle_statistics.xlsx` | Branching angles | Individual angles at each junction |
| `IMAGE_vascular_density.xlsx` | Density metrics | Coverage percentages and length density |
| `IMAGE_fractal_dimension.xlsx` | Fractal analysis | Complexity measures and power law fit |
| `IMAGE_betti_numbers.xlsx` | Topological invariants | Components, loops, and voids |

---

## Feature 1: Enhanced Edge Data

**File:** `IMAGE_alldata.xlsx`

### What It Contains

This file provides **detailed information about each vessel segment** (edge) in the network.

### Columns

| Column | Description | Units | Typical Range |
|--------|-------------|-------|---------------|
| `index1` | Starting node ID | - | 0 to N |
| `index2` | Ending node ID | - | 0 to N |
| `length` | Euclidean distance between nodes | pixels | 1-100 |
| `mean_area` | Average cross-sectional area | pixels² | 100-400 |
| `std_area` | Std deviation of area | pixels² | 10-100 |
| `perimeter` | Total perimeter of segment | pixels | 50-500 |

### What It Measures

**Mean Area:**
- Average vessel caliber along the segment
- Reflects vessel diameter
- Changes indicate stenosis or dilation

**Standard Deviation of Area:**
- Uniformity of vessel width
- Low std = uniform caliber
- High std = irregular/tortuous vessel

**Perimeter:**
- Total boundary length
- Related to vessel surface area
- Useful for perfusion calculations

### Example Data

```
index1  index2  length  mean_area  std_area  perimeter
0       1       45.2    234.5      12.3      156.8
0       2       38.7    198.2      18.9      142.3
1       3       52.1    245.7      8.4       168.2
```

### Interpretation

**Healthy vessel segment:**
```
length: 40-60 pixels
mean_area: 200-300 pixels²
std_area: <20 pixels² (uniform)
```

**Stenotic vessel:**
```
mean_area: <150 pixels² (narrow)
std_area: >30 pixels² (irregular)
```

### Clinical Use

1. **Identify narrowed vessels:** `mean_area < 150`
2. **Detect irregularity:** `std_area > 25`
3. **Track caliber changes:** Compare longitudinally
4. **Calculate flow resistance:** Use Poiseuille's law with area

---

## Feature 2: Enhanced Node Data

**File:** `IMAGE_degreedata.xlsx`

### What It Contains

Information about each **junction point and endpoint** (node) in the network.

### Columns

| Column | Description | Units | Node Type |
|--------|-------------|-------|-----------|
| `node_id` | Unique node identifier | - | All |
| `x` | X coordinate | pixels | All |
| `y` | Y coordinate | pixels | All |
| `degree` | Number of connected vessels | count | All |

### Node Types by Degree

| Degree | Type | Biological Meaning | Typical % |
|--------|------|-------------------|-----------|
| 1 | **Endpoint** | Terminal capillary, vessel end | 20-40% |
| 2 | **Continuation** | Mid-vessel point (rare in skeleton) | 0-5% |
| 3 | **Y-junction** | Bifurcation, branching point | 50-70% |
| 4 | **X-junction** | Crossing vessels, anastomosis | 10-30% |
| 5+ | **Complex junction** | Multiple vessel meeting | <5% |

### Example Data

```
node_id    x     y    degree
0         125   456    1      # Endpoint
1         234   567    3      # Y-junction
2         456   789    4      # X-junction
3         678   901    3      # Y-junction
```

### Interpretation

**Degree Distribution:**
- High proportion of degree 3: Tree-like branching structure
- High proportion of degree 4: Many crossings/anastomoses
- Degree 5+: Complex, highly interconnected regions

### Clinical Use

1. **Network topology:** Count degree types
2. **Anastomosis detection:** Find degree 4+ nodes
3. **Endpoint analysis:** Identify terminal vessels
4. **Spatial distribution:** Map junction locations

---

## Feature 3: Regional Area Analysis

**File:** `IMAGE_regional_area.xlsx`

### What It Contains

**Spatial distribution of vessel areas** across different regions of the image.

### Sheet: "Regional"

| Column | Description | Units |
|--------|-------------|-------|
| `region_y` | Region row index | - |
| `region_x` | Region column index | - |
| `total_area` | Total vessel area in region | pixels² |
| `mean_area` | Average vessel area in region | pixels² |
| `density` | Vessel density (area/region area) | 0-1 |

### Sheet: "Summary"

Global statistics:

| Metric | Description |
|--------|-------------|
| `global_mean_area` | Overall mean vessel area |
| `global_std_area` | Overall std of vessel area |
| `min_regional_mean` | Smallest regional mean |
| `max_regional_mean` | Largest regional mean |
| `regional_variation` | Coefficient of variation |

### What It Measures

**Regional Heterogeneity:**
- Uniform network: Low variation between regions
- Heterogeneous network: High variation
- Identifies areas with larger/smaller vessels

**Spatial Patterns:**
- Central vs peripheral differences
- Superior vs inferior differences
- Anatomical asymmetries

### Example Data

**Regional Sheet:**
```
region_y  region_x  total_area  mean_area  density
0         0         45678       234.5      0.182
0         1         52341       245.8      0.210
0         2         38902       198.3      0.155
```

**Summary Sheet:**
```
global_mean_area:      217.67
global_std_area:       45.32
min_regional_mean:     198.30
max_regional_mean:     245.80
regional_variation:    10.23%
```

### Interpretation

**Uniform Network:**
```
regional_variation: <15%
All regions have similar mean_area
```

**Heterogeneous Network:**
```
regional_variation: >25%
Large differences between regions
```

### Clinical Use

1. **Identify ischemic zones:** Low density regions
2. **Detect asymmetry:** Compare left/right, top/bottom
3. **Map perfusion:** Regional density as proxy
4. **Track progression:** Monitor regional changes

---

## Feature 4: Branching Angle Statistics

**File:** `IMAGE_angle_statistics.xlsx`

### What It Contains

**Angles between vessel branches** at each junction point.

### Sheet: "Statistics"

Global statistics about all branching angles:

| Metric | Description | Units |
|--------|-------------|-------|
| `num_branch_points` | Total junctions with 3+ branches | count |
| `num_sampled` | Number of junctions analyzed | count |
| `num_angles` | Total angles measured | count |
| `mean_angle` | Average branching angle | degrees |
| `median_angle` | Median branching angle | degrees |
| `std_angle` | Standard deviation | degrees |
| `min_angle` | Smallest angle | degrees |
| `max_angle` | Largest angle | degrees |
| `acute_count` | Angles < 60° | count |
| `medium_count` | Angles 60-120° | count |
| `obtuse_count` | Angles > 120° | count |
| `acute_percent` | % acute angles | % |
| `medium_percent` | % medium angles | % |
| `obtuse_percent` | % obtuse angles | % |

### Sheet: "Detailed"

Individual angles at each junction:

| Column | Description |
|--------|-------------|
| `x` | X coordinate of junction |
| `y` | Y coordinate of junction |
| `angle_index` | Index of angle at this junction (0, 1, 2...) |
| `angle` | Angle value in degrees |

### What It Measures

**Branching Angle:**
- Angle between two vessel branches
- At a 3-branch junction: 3 angles (C(3,2))
- At a 4-branch junction: 6 angles (C(4,2))
- At a 5-branch junction: 10 angles (C(5,2))

**Murray's Law:**
- Optimal branching angle ≈ 37.5° (75° between branches)
- Minimizes energy cost of flow
- Healthy networks approach this optimal value

### Example Data

**Statistics Sheet:**
```
num_branch_points:    12411
num_sampled:          500
num_angles:           1523
mean_angle:           35.32°
median_angle:         32.18°
acute_count:          1117 (73.3%)
medium_count:         384 (25.2%)
obtuse_count:         22 (1.4%)
```

**Detailed Sheet:**
```
x     y     angle_index  angle
1247  1868  0            141.08°
1247  1868  1            140.08°
1247  1868  2            1.01°
1572  1159  0            3.95°
1572  1159  1            162.78°
1572  1159  2            158.84°
```

### Interpretation

**Angle Categories:**

| Range | Category | Meaning | Optimal % |
|-------|----------|---------|-----------|
| 0-30° | Very acute | Sharp branching | 20-40% |
| 30-60° | Acute | Moderate branching | 40-60% |
| 60-120° | Medium | Wide branching | 20-40% |
| 120-180° | Obtuse | Very wide/backward | <10% |

**Healthy Network:**
```
Mean angle: 30-45°
Acute percent: 60-80%
Distribution: Bell-shaped around 35°
```

**Abnormal Patterns:**
```
Mean angle < 25°: Too sharp (high resistance)
Mean angle > 50°: Too wide (suboptimal)
High obtuse %: Disorganized branching
```

### Clinical Use

1. **Network optimization:** Compare to Murray's law (37.5°)
2. **Angiogenesis quality:** Tumor vessels have abnormal angles
3. **Disease progression:** Angles change with remodeling
4. **Treatment response:** Normalization of angles

---

## Feature 5: Vascular Density

**File:** `IMAGE_vascular_density.xlsx`

### What It Contains

**Coverage metrics** - how much of the tissue is occupied by vessels.

### Sheet: "Global"

Overall density metrics:

| Metric | Description | Units | Typical Range |
|--------|-------------|-------|---------------|
| `vessel_area_density` (VAD) | % of image occupied by vessels | 0-1 | 0.10-0.30 |
| `vessel_length_density` (VLD) | Total skeleton length / image area | pixels⁻¹ | 0.01-0.03 |
| `total_vessel_area` | Total pixels in vessels | pixels² | 100k-1M |
| `total_vessel_length` | Total skeleton length | pixels | 10k-100k |
| `image_area` | Total image area | pixels² | millions |
| `num_regions` | Number of spatial regions | count | 4-9 |
| `regional_asymmetry` | Ratio of max/min regional VAD | ratio | 1.5-3.0 |

### Sheet: "Regional"

Density by spatial region:

| Column | Description |
|--------|-------------|
| `region_y` | Region row index |
| `region_x` | Region column index |
| `vessel_area_density` | VAD for this region |
| `vessel_length_density` | VLD for this region |

### What It Measures

**Vessel Area Density (VAD):**
```
VAD = (Total vessel pixels) / (Total image pixels)
```
- Indicates tissue coverage
- Higher VAD = better perfusion potential
- Range: 0 (no vessels) to 1 (all vessels)

**Vessel Length Density (VLD):**
```
VLD = (Total skeleton length) / (Image area)
```
- Indicates network extent
- Higher VLD = more extensive network
- Units: pixels⁻¹ or mm⁻¹

**Regional Asymmetry:**
```
Asymmetry = max(regional VAD) / min(regional VAD)
```
- Measures spatial heterogeneity
- 1.0 = perfectly uniform
- >2.0 = significant variation

### Example Data

**Global Sheet:**
```
vessel_area_density:      0.1765 (17.65%)
vessel_length_density:    0.014148
total_vessel_area:        1,766,235 pixels²
total_vessel_length:      141,654 pixels
image_area:              10,000,000 pixels²
regional_asymmetry:       2.32
```

**Regional Sheet:**
```
region_y  region_x  vessel_area_density  vessel_length_density
0         0         0.1504               0.0127
0         1         0.1823               0.0145
0         2         0.1967               0.0158
1         0         0.1635               0.0132
```

### Interpretation

**VAD Categories:**

| VAD | Coverage | Clinical Interpretation |
|-----|----------|------------------------|
| <0.10 | Very low (<10%) | Severe ischemia, poor perfusion |
| 0.10-0.15 | Low (10-15%) | Mild hypoperfusion |
| 0.15-0.25 | Normal (15-25%) | Adequate perfusion |
| 0.25-0.35 | High (25-35%) | Excellent perfusion |
| >0.35 | Very high (>35%) | Hypervascularity (e.g., tumor) |

**Regional Asymmetry:**

| Asymmetry | Interpretation |
|-----------|----------------|
| 1.0-1.5 | Uniform distribution |
| 1.5-2.5 | Moderate variation (normal) |
| 2.5-4.0 | High variation (concern) |
| >4.0 | Severe heterogeneity (pathological) |

### Clinical Use

1. **Perfusion assessment:** VAD as proxy for blood supply
2. **Disease severity:** Lower VAD in diabetic retinopathy
3. **Tumor detection:** Very high VAD (>0.35)
4. **Ischemia identification:** Low VAD regions
5. **Treatment response:** VAD increase = improvement

---

## Feature 6: Fractal Dimension

**File:** `IMAGE_fractal_dimension.xlsx`

### What It Contains

**Complexity metrics** using fractal analysis - how the network fills space across different scales.

### Columns

| Column | Description | Range | Healthy Range |
|--------|-------------|-------|---------------|
| `vessel_fractal_dimension` | FD of original vessels | 1.0-2.0 | 1.6-1.8 |
| `skeleton_fractal_dimension` | FD of skeleton | 1.0-2.0 | 1.5-1.7 |
| `vessel_r_squared` | Fit quality (vessels) | 0-1 | >0.90 |
| `skeleton_r_squared` | Fit quality (skeleton) | 0-1 | >0.90 |
| `power_law_exponent` | Slope of log-log plot | - | - |

### What It Measures

**Fractal Dimension (FD):**
- Measures space-filling complexity
- Box-counting method: Count boxes needed to cover image at different scales
- Higher FD = more complex, space-filling pattern

**FD Range:**
- FD = 1.0: Straight line (minimal complexity)
- FD = 1.5: Moderate complexity
- FD = 2.0: Completely fills 2D space

**Biological Significance:**
- Healthy retinal vessels: FD ≈ 1.65-1.75
- Lung capillaries: FD ≈ 1.70-1.80
- Tumor vessels: Often abnormal FD

### Example Data

```
vessel_fractal_dimension:     1.8174
skeleton_fractal_dimension:   1.5724
vessel_r_squared:             0.9999
skeleton_r_squared:           0.9987
power_law_exponent:          -1.8174
```

### Interpretation

**FD Categories:**

| FD | Complexity | Clinical Meaning |
|----|------------|-----------------|
| <1.4 | Very low | Severe rarefaction, minimal branching |
| 1.4-1.6 | Low | Reduced complexity, vessel loss |
| 1.6-1.8 | Normal | Healthy complexity |
| 1.8-2.0 | High | Dense, complex branching |
| >1.9 | Very high | Abnormal (tumor angiogenesis) |

**R² (Fit Quality):**
- R² > 0.95: Excellent power-law fit (true fractal)
- R² = 0.90-0.95: Good fit
- R² < 0.90: Poor fit (not truly fractal)

**Vessel vs Skeleton FD:**
- Vessel FD > Skeleton FD (usually by 0.1-0.3)
- Vessel FD captures caliber variation
- Skeleton FD captures topology only

### Clinical Use

1. **Disease staging:**
   - Diabetic retinopathy: FD decreases over time
   - Early: FD ≈ 1.75
   - Late: FD ≈ 1.55

2. **Tumor detection:**
   - Abnormally high FD (>1.85)
   - Poor R² (disorganized)

3. **Aging:**
   - FD decreases with age
   - ~0.01 per decade

4. **Treatment monitoring:**
   - Anti-VEGF: May normalize FD
   - Revascularization: FD increases

### Mathematical Details

**Box-Counting Method:**
```
1. Overlay grid of boxes (size r)
2. Count boxes containing vessels: N(r)
3. Repeat for different box sizes
4. Plot log(N(r)) vs log(1/r)
5. Slope = Fractal Dimension
```

**Power Law:**
```
N(r) = k × r^(-FD)
log(N(r)) = log(k) - FD × log(r)
```

---

## Feature 7: Betti Numbers

**File:** `IMAGE_betti_numbers.xlsx`

### What It Contains

**Topological invariants** - properties that don't change with stretching or bending, only with cutting or gluing.

### Sheet: "Summary"

| Metric | Description | Typical Range |
|--------|-------------|---------------|
| `beta_0` | Number of connected components | 1-50 |
| `beta_1` | Number of independent loops | 100-5000 |
| `beta_2` | Number of voids (holes) | 100-5000 |
| `euler_characteristic` | χ = β₀ - β₁ + β₂ | Negative |
| `loops_per_component` | β₁ / β₀ | 50-200 |
| `void_area_total` | Total area of voids | pixels² |
| `void_area_fraction` | Voids / total area | 0.05-0.20 |
| `average_void_size` | Mean void area | pixels² |

### Sheet: "Components"

Details of each connected component:

| Column | Description |
|--------|-------------|
| `component_id` | Component identifier |
| `size` | Number of pixels |
| `centroid_x` | Center X coordinate |
| `centroid_y` | Center Y coordinate |

### Sheet: "Voids"

Details of each void (avascular region):

| Column | Description |
|--------|-------------|
| `void_id` | Void identifier |
| `area` | Area in pixels² |
| `centroid_x` | Center X coordinate |
| `centroid_y` | Center Y coordinate |

### What It Measures

**β₀ (Beta-0): Connected Components**
- Number of separate vessel networks
- Ideally β₀ = 1 (fully connected)
- High β₀ = fragmented network

**β₁ (Beta-1): Independent Loops**
- Number of cycles/redundant pathways
- Collateral circulation capacity
- Higher β₁ = more redundancy

**β₂ (Beta-2): Voids**
- Number of enclosed avascular regions
- Ischemic or non-perfused areas
- Higher β₂ = more gaps in coverage

**Euler Characteristic:**
```
χ = β₀ - β₁ + β₂
```
- Topological invariant
- Negative for vascular networks
- More negative = more complex

### Example Data

**Summary Sheet:**
```
beta_0:                    15
beta_1:                    1808
beta_2:                    1800
euler_characteristic:      7
loops_per_component:       120.5
void_area_total:           1,234,567 pixels²
void_area_fraction:        0.1235 (12.35%)
average_void_size:         686 pixels²
```

### Interpretation

**β₀ (Components):**

| β₀ | Interpretation | Clinical Meaning |
|----|----------------|------------------|
| 1 | Fully connected | Excellent integration |
| 2-5 | Mildly fragmented | Minor peripheral dropout |
| 5-20 | Moderately fragmented | Concern for ischemia |
| >20 | Severely fragmented | Poor network integrity |

**β₁ (Loops):**

| Loops/Component | Interpretation | Clinical Meaning |
|-----------------|----------------|------------------|
| <20 | Very few | Poor collaterals, tree-like |
| 20-50 | Low | Limited redundancy |
| 50-150 | Normal | Good collateral circulation |
| >150 | Excellent | Highly redundant |

**β₂ (Voids):**

| Void Fraction | Interpretation | Clinical Meaning |
|---------------|----------------|------------------|
| <5% | Minimal | Dense coverage |
| 5-15% | Moderate | Normal avascular spaces |
| 15-25% | Significant | Concerning for ischemia |
| >25% | Severe | Large non-perfused areas |

### Clinical Use

1. **Network Fragmentation:**
   ```
   if beta_0 > 10:
       print("⚠️ Fragmented network - check for vessel dropout")
   ```

2. **Collateral Capacity:**
   ```
   redundancy = beta_1 / beta_0
   if redundancy > 100:
       print("✓ Excellent collateral circulation")
   elif redundancy < 30:
       print("⚠️ Limited collateral pathways")
   ```

3. **Ischemic Area:**
   ```
   if void_area_fraction > 0.20:
       print("⚠️ Significant avascular regions")
   ```

4. **Longitudinal Tracking:**
   ```
   Diabetic retinopathy progression:
   - β₀ increases (fragmentation)
   - β₁ decreases (loop loss)
   - β₂ increases (void expansion)
   ```

### Mathematical Details

**Homology Theory:**
- β₀ = rank of H₀ (0-dimensional homology)
- β₁ = rank of H₁ (1-dimensional homology)
- β₂ = rank of H₂ (2-dimensional homology)

**Computation:**
- Simplicial complex construction
- Boundary matrix calculation
- Kernel and image computation
- Rank determination

---

## Interpretation Guide

### Comprehensive Network Assessment

Use multiple features together for complete characterization:

#### 1. **Vessel Caliber Assessment**
```
Feature: alldata.xlsx (mean_area)
Normal: 200-300 pixels²
Stenosis: <150 pixels²
Dilation: >400 pixels²
```

#### 2. **Coverage Assessment**
```
Feature: vascular_density.xlsx (VAD)
Poor: <0.15 (15%)
Normal: 0.15-0.25 (15-25%)
Excellent: >0.25 (25%)
```

#### 3. **Complexity Assessment**
```
Feature: fractal_dimension.xlsx
Low: FD < 1.6
Normal: FD = 1.6-1.8
High: FD > 1.8
```

#### 4. **Topology Assessment**
```
Feature: betti_numbers.xlsx
Fragmentation: β₀ (want = 1)
Collaterals: β₁/β₀ (want > 50)
Ischemia: void_fraction (want < 0.15)
```

#### 5. **Optimization Assessment**
```
Feature: angle_statistics.xlsx
Optimal: mean_angle ≈ 35° (Murray's law)
Suboptimal: mean_angle < 25° or > 50°
```

### Multi-Feature Patterns

**Healthy Network:**
```
✓ VAD = 0.18-0.25 (good coverage)
✓ FD = 1.7-1.8 (high complexity)
✓ β₀ = 1-5 (minimal fragmentation)
✓ β₁/β₀ > 80 (excellent collaterals)
✓ void_fraction < 0.12 (minimal voids)
✓ mean_angle = 30-40° (optimal)
```

**Diabetic Retinopathy:**
```
⚠️ VAD decreasing (vessel loss)
⚠️ FD decreasing (reduced complexity)
⚠️ β₀ increasing (fragmentation)
⚠️ β₁ decreasing (loop loss)
⚠️ void_fraction increasing (ischemia)
⚠️ Angles more variable
```

**Tumor Angiogenesis:**
```
⚠️ VAD very high (>0.30)
⚠️ FD abnormal (>1.85 or poor R²)
⚠️ β₁ very high (chaotic loops)
⚠️ Angles disorganized
⚠️ High regional asymmetry
```

---

## Clinical Applications

### 1. Disease Detection

**Diabetic Retinopathy Screening:**
```python
# Red flags
if VAD < 0.15:
    risk_score += 2
if FD < 1.6:
    risk_score += 2
if beta_0 > 10:
    risk_score += 1
if void_fraction > 0.15:
    risk_score += 2

# Risk stratification
if risk_score >= 5:
    print("HIGH RISK - Refer to specialist")
elif risk_score >= 3:
    print("MODERATE RISK - Close monitoring")
else:
    print("LOW RISK - Routine follow-up")
```

### 2. Progression Monitoring

**Track changes over time:**
```python
# Compare baseline to follow-up
delta_VAD = followup['VAD'] - baseline['VAD']
delta_FD = followup['FD'] - baseline['FD']
delta_beta1 = followup['beta_1'] - baseline['beta_1']

if delta_VAD < -0.03 or delta_FD < -0.15:
    print("⚠️ PROGRESSION DETECTED")
```

### 3. Treatment Response

**Assess therapy effectiveness:**
```python
# After anti-VEGF treatment
if post['VAD'] > pre['VAD'] * 1.05:
    print("✓ Positive response - vessel recovery")
if post['void_fraction'] < pre['void_fraction'] * 0.9:
    print("✓ Ischemia improved")
```

### 4. Surgical Planning

**Identify critical structures:**
```python
# High-degree nodes are critical junctions
critical_nodes = degreedata[degreedata['degree'] >= 4]
print(f"Preserve these {len(critical_nodes)} critical junctions")

# Large loops provide collateral circulation
large_loops = components[components['size'] > 1000]
print(f"Important collateral regions: {len(large_loops)}")
```

### 5. Research Studies

**Quantitative biomarkers:**
```python
# Create feature vector for machine learning
features = [
    VAD,
    VLD,
    FD,
    beta_0,
    beta_1,
    void_fraction,
    mean_angle,
    mean_area
]

# Classify disease stage
stage = classifier.predict([features])
```

---

## Summary Table

| Feature | File | Key Metrics | Clinical Use |
|---------|------|-------------|--------------|
| **Edge Data** | alldata.xlsx | mean_area, std_area | Vessel caliber, stenosis detection |
| **Node Data** | degreedata.xlsx | degree distribution | Junction analysis, anastomoses |
| **Regional Area** | regional_area.xlsx | spatial variation | Asymmetry, ischemic zones |
| **Branching Angles** | angle_statistics.xlsx | mean_angle, distribution | Network optimization, quality |
| **Vascular Density** | vascular_density.xlsx | VAD, VLD, asymmetry | Coverage, perfusion capacity |
| **Fractal Dimension** | fractal_dimension.xlsx | FD, R² | Complexity, disease staging |
| **Betti Numbers** | betti_numbers.xlsx | β₀, β₁, β₂, voids | Topology, collaterals, ischemia |

---

## Quick Reference

### File Priority for Different Goals

**Goal: Disease Screening**
1. vascular_density.xlsx (VAD)
2. fractal_dimension.xlsx (FD)
3. betti_numbers.xlsx (β₀, void_fraction)

**Goal: Detailed Analysis**
1. All files
2. Compare to age-matched norms
3. Look for multi-feature patterns

**Goal: Longitudinal Tracking**
1. vascular_density.xlsx (VAD)
2. fractal_dimension.xlsx (FD)
3. betti_numbers.xlsx (β₁)

**Goal: Surgical Planning**
1. degreedata.xlsx (high-degree nodes)
2. betti_numbers.xlsx (large components)
3. angle_statistics.xlsx (junction locations)

---

## Technical Notes

### Quality Control

**Check these for each image:**

1. **Fractal R²:** Should be >0.90
2. **Betti β₀:** Should be <50 (if higher, check segmentation)
3. **VAD:** Should be 0.05-0.40 (if outside, check thresholding)
4. **Mean area:** Should be >100 pixels² (if lower, check resolution)

### Units

- **Length:** pixels (convert to mm using resolution)
- **Area:** pixels² (convert to mm² using resolution²)
- **Density:** pixels⁻¹ (convert to mm⁻¹)
- **Angles:** degrees
- **Fractal dimension:** dimensionless
- **Betti numbers:** counts

### Resolution Conversion

If image resolution is known:
```python
mm_per_pixel = 0.01  # Example: 10 μm/pixel

# Convert length
length_mm = length_pixels * mm_per_pixel

# Convert area
area_mm2 = area_pixels * (mm_per_pixel ** 2)

# Convert VLD
VLD_mm = VLD_pixels / mm_per_pixel
```

---

## Frequently Asked Questions

**Q: Which feature is most important?**
A: Depends on clinical question. For screening: VAD and FD. For detailed analysis: all features.

**Q: What's a "healthy" network?**
A: VAD 0.18-0.25, FD 1.7-1.8, β₀ <5, β₁/β₀ >80, void_fraction <0.12, mean_angle 30-40°.

**Q: How often should I monitor?**
A: Depends on disease. Diabetics: annually. High-risk: every 6 months.

**Q: Can I compare different patients?**
A: Yes, but account for age, image quality, and anatomical location.

**Q: What if results are borderline?**
A: Use clinical judgment. Consider multiple features together, not single metrics.

