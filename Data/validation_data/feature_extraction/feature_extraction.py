#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy, scipy.io
from PIL import Image
import matplotlib.colors as mcolors
import cv2
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import operator
from scipy.spatial.distance import euclidean
import pandas as pd
from connectivity_matrix_test import *
from _utility import *
from skimage.morphology import thin
import pdb
from PIL import UnidentifiedImageError
from scipy import ndimage
from skimage import measure

imgs_dir=os.getcwd() + "/img/p2-p7/"
save_dir=os.getcwd() + "/feature/p2-p7/"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Feature Extraction v3.0 - COMPLETE
Features: Area/Volume, Branching Angles, Vascular Density, Fractal Dimension, Betti Numbers
"""

# ============================================================================
# FEATURES 1-4: (Same as before - Area, Angles, Density, Fractal)
# ============================================================================

def calculate_regional_area(image, num_regions=(2, 2)):
    """Calculate vessel area in different regions."""
    image_array = np.array(image)
    h, w = image_array.shape
    regions_h, regions_w = num_regions
    
    region_height = h // regions_h
    region_width = w // regions_w
    
    regional_data = []
    
    for i in range(regions_h):
        for j in range(regions_w):
            y_start = i * region_height
            y_end = (i + 1) * region_height if i < regions_h - 1 else h
            x_start = j * region_width
            x_end = (j + 1) * region_width if j < regions_w - 1 else w
            
            region = image_array[y_start:y_end, x_start:x_end]
            region_area = np.sum(region > 0)
            region_total = region.size
            region_density = region_area / region_total
            
            regional_data.append({
                'region_row': i,
                'region_col': j,
                'y_start': y_start,
                'y_end': y_end,
                'x_start': x_start,
                'x_end': x_end,
                'area_pixels': region_area,
                'total_pixels': region_total,
                'density': region_density
            })
    
    return regional_data


def get_branch_directions(branch_point, skeleton, radius=15):
    """
    Get direction vectors of branches emanating from a junction point.
    
    Uses the actual neighbor pixels to determine branch directions,
    not connected components in a region.
    """
    y, x = branch_point
    
    # Get 8-connected neighbors
    neighbor_offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    directions = []
    
    for dy, dx in neighbor_offsets:
        ny, nx = y + dy, x + dx
        
        # Check if neighbor is within bounds and is a skeleton pixel
        if (0 <= ny < skeleton.shape[0] and 
            0 <= nx < skeleton.shape[1] and 
            skeleton[ny, nx]):
            
            # Trace along this branch to get direction
            # Start from the neighbor and move outward
            trace_y, trace_x = ny, nx
            path_points = [(ny, nx)]
            
            # Trace for a few steps to get direction
            for step in range(min(radius, 10)):
                # Look at neighbors of current point (excluding previous point)
                found_next = False
                for dy2, dx2 in neighbor_offsets:
                    next_y, next_x = trace_y + dy2, trace_x + dx2
                    
                    # Skip if out of bounds, not skeleton, or is the branch point
                    if (next_y == y and next_x == x):
                        continue
                    if not (0 <= next_y < skeleton.shape[0] and 
                           0 <= next_x < skeleton.shape[1]):
                        continue
                    if not skeleton[next_y, next_x]:
                        continue
                    
                    # Skip if we already visited this point
                    if len(path_points) > 1 and (next_y, next_x) == path_points[-2]:
                        continue
                    
                    # Move to this point
                    trace_y, trace_x = next_y, next_x
                    path_points.append((next_y, next_x))
                    found_next = True
                    break
                
                if not found_next:
                    break
            
            # Calculate direction from branch point to furthest traced point
            if len(path_points) > 0:
                furthest = path_points[-1]
                dir_y = furthest[0] - y
                dir_x = furthest[1] - x
                length = np.sqrt(dir_x**2 + dir_y**2)
                
                if length > 0:
                    directions.append((dir_x/length, dir_y/length))
    
    return directions


def calculate_branching_angles(directions):
    """Calculate angles between branch directions."""
    angles = []
    n = len(directions)
    
    for i in range(n):
        for j in range(i+1, n):
            dot = directions[i][0]*directions[j][0] + directions[i][1]*directions[j][1]
            dot = np.clip(dot, -1, 1)
            angle = np.arccos(dot)
            angle_deg = np.degrees(angle)
            angles.append(angle_deg)
    
    return angles


def extract_branching_angle_features(skeleton, sample_size=None):
    """Extract branching angle features."""
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    neighbors = ndimage.convolve(skeleton.astype(int), kernel, mode='constant')
    neighbors = neighbors * skeleton
    
    # Find all branch points (3 or more neighbors)
    branch_points = np.argwhere(neighbors >= 3)
    
    if len(branch_points) == 0:
        return {'num_branch_points': 0}, []
    
    if sample_size is not None and sample_size < len(branch_points):
        sample_indices = np.random.choice(len(branch_points), sample_size, replace=False)
        sampled_points = branch_points[sample_indices]
    else:
        sampled_points = branch_points
    
    all_angles = []
    detailed_data = []
    
    for bp in sampled_points:
        directions = get_branch_directions(tuple(bp), skeleton, radius=15)
        if len(directions) >= 2:
            angles = calculate_branching_angles(directions)
            all_angles.extend(angles)
            # Only store x, y, and angles for output
            detailed_data.append({
                'y': int(bp[0]),
                'x': int(bp[1]),
                'angles': angles
            })
    
    if len(all_angles) > 0:
        stats = {
            'num_branch_points': len(branch_points),
            'num_sampled': len(sampled_points),
            'num_angles': len(all_angles),
            'mean_angle': np.mean(all_angles),
            'median_angle': np.median(all_angles),
            'std_angle': np.std(all_angles),
            'min_angle': np.min(all_angles),
            'max_angle': np.max(all_angles),
            'acute_count': np.sum(np.array(all_angles) < 60),
            'medium_count': np.sum((np.array(all_angles) >= 60) & (np.array(all_angles) < 120)),
            'obtuse_count': np.sum(np.array(all_angles) >= 120),
            'acute_percent': np.sum(np.array(all_angles) < 60) / len(all_angles) * 100,
            'medium_percent': np.sum((np.array(all_angles) >= 60) & (np.array(all_angles) < 120)) / len(all_angles) * 100,
            'obtuse_percent': np.sum(np.array(all_angles) >= 120) / len(all_angles) * 100
        }
    else:
        stats = {'num_branch_points': len(branch_points), 'num_angles': 0}
    
    return stats, detailed_data


def add_area_features_to_edges(graph, skeleton, distance_transform):
    """Add area features to edges."""
    for n1, n2, data in graph.edges(data=True):
        y1, x1 = n1
        y2, x2 = n2
        
        num_samples = max(int(euclidean(n1, n2)), 2)
        y_samples = np.linspace(y1, y2, num_samples).astype(int)
        x_samples = np.linspace(x1, x2, num_samples).astype(int)
        
        areas = []
        for y, x in zip(y_samples, x_samples):
            if 0 <= y < distance_transform.shape[0] and 0 <= x < distance_transform.shape[1]:
                radius = distance_transform[y, x]
                area = np.pi * (radius ** 2)
                areas.append(area)
        
        if len(areas) > 0:
            graph[n1][n2]['mean_area'] = np.mean(areas)
            graph[n1][n2]['min_area'] = np.min(areas)
            graph[n1][n2]['max_area'] = np.max(areas)
            graph[n1][n2]['area_variation'] = np.std(areas)
        else:
            graph[n1][n2]['mean_area'] = 0
            graph[n1][n2]['min_area'] = 0
            graph[n1][n2]['max_area'] = 0
            graph[n1][n2]['area_variation'] = 0


def add_branching_angles_to_nodes(graph, skeleton):
    """Add angle features to nodes."""
    for node in graph.nodes():
        degree = graph.degree(node)
        
        if degree >= 3:
            directions = get_branch_directions(node, skeleton, radius=15)
            
            if len(directions) >= 2:
                angles = calculate_branching_angles(directions)
                graph.nodes[node]['min_angle'] = min(angles)
                graph.nodes[node]['max_angle'] = max(angles)
                graph.nodes[node]['mean_angle'] = np.mean(angles)
                graph.nodes[node]['angles'] = angles
            else:
                graph.nodes[node]['min_angle'] = None
                graph.nodes[node]['max_angle'] = None
                graph.nodes[node]['mean_angle'] = None
                graph.nodes[node]['angles'] = []
        else:
            graph.nodes[node]['min_angle'] = None
            graph.nodes[node]['max_angle'] = None
            graph.nodes[node]['mean_angle'] = None
            graph.nodes[node]['angles'] = []


def calculate_vascular_density(vessel_binary, skeleton, regions=None):
    """Calculate vascular density metrics."""
    h, w = vessel_binary.shape
    total_area = h * w
    
    vessel_area = np.sum(vessel_binary)
    skeleton_length = np.sum(skeleton)
    
    vessel_area_density = vessel_area / total_area
    vessel_length_density = skeleton_length / total_area
    
    results = {
        'total_area': total_area,
        'vessel_area': vessel_area,
        'skeleton_length': skeleton_length,
        'vessel_area_density': vessel_area_density,
        'vessel_length_density': vessel_length_density,
        'vessel_area_fraction': vessel_area_density,
        'vessel_coverage_percent': vessel_area_density * 100
    }
    
    if regions is not None:
        regions_h, regions_w = regions
        regional_densities = []
        
        for i in range(regions_h):
            for j in range(regions_w):
                y_start = i * (h // regions_h)
                y_end = (i + 1) * (h // regions_h) if i < regions_h - 1 else h
                x_start = j * (w // regions_w)
                x_end = (j + 1) * (w // regions_w) if j < regions_w - 1 else w
                
                region_vessel = vessel_binary[y_start:y_end, x_start:x_end]
                region_skeleton = skeleton[y_start:y_end, x_start:x_end]
                region_total = region_vessel.size
                
                region_vad = np.sum(region_vessel) / region_total
                region_vld = np.sum(region_skeleton) / region_total
                
                regional_densities.append({
                    'region_row': i,
                    'region_col': j,
                    'vessel_area_density': region_vad,
                    'vessel_length_density': region_vld,
                    'vad_percent': region_vad * 100,
                    'y_start': y_start,
                    'y_end': y_end,
                    'x_start': x_start,
                    'x_end': x_end
                })
        
        results['regional_densities'] = regional_densities
    
    return results


def calculate_fractal_dimension_boxcount(binary_img, min_box_size=4, max_box_size=None):
    """Calculate fractal dimension."""
    if max_box_size is None:
        max_box_size = min(binary_img.shape) // 4
    
    sizes = []
    size = min_box_size
    while size <= max_box_size:
        sizes.append(size)
        size *= 2
    
    counts = []
    
    for size in sizes:
        h_boxes = binary_img.shape[0] // size
        w_boxes = binary_img.shape[1] // size
        
        h_crop = h_boxes * size
        w_crop = w_boxes * size
        cropped = binary_img[:h_crop, :w_crop]
        
        reshaped = cropped.reshape(h_boxes, size, w_boxes, size)
        boxes_filled = np.sum(reshaped.max(axis=(1, 3)) > 0)
        counts.append(boxes_filled)
    
    sizes_array = np.array(sizes, dtype=float)
    counts_array = np.array(counts, dtype=float)
    
    valid = counts_array > 0
    sizes_array = sizes_array[valid]
    counts_array = counts_array[valid]
    
    if len(sizes_array) > 1:
        log_sizes = np.log(sizes_array)
        log_counts = np.log(counts_array)
        
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        fractal_dim = -coeffs[0]
        
        log_counts_pred = coeffs[0] * log_sizes + coeffs[1]
        ss_res = np.sum((log_counts - log_counts_pred) ** 2)
        ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    else:
        fractal_dim = None
        r_squared = None
        coeffs = None
        sizes = []
        counts = []
    
    return {
        'fractal_dimension': fractal_dim,
        'r_squared': r_squared,
        'box_sizes': sizes,
        'box_counts': counts,
        'regression_coeffs': coeffs.tolist() if coeffs is not None else None
    }


# ============================================================================
# FEATURE 5: BETTI NUMBERS (NEW)
# ============================================================================

def calculate_betti_numbers(vessel_binary, skeleton, graph=None):
    """
    Calculate Betti numbers - topological invariants of the vascular network.
    
    Betti numbers quantify topological features:
    - β0: Number of connected components (network fragmentation)
    - β1: Number of independent loops (anastomoses, collateral circulation)
    - β2: Number of enclosed voids (avascular regions)
    
    Parameters:
    -----------
    vessel_binary : numpy.ndarray (bool)
        Binary vessel image
    skeleton : numpy.ndarray (bool)
        Binary skeleton image
    graph : networkx.Graph or None
        Pre-computed graph (optional, for more accurate β1)
    
    Returns:
    --------
    dict : Betti numbers and related topological metrics
    """
    
    # ========== β0: Connected Components ==========
    labeled_vessels, beta_0 = ndimage.label(vessel_binary)
    labeled_skeleton, beta_0_skeleton = ndimage.label(skeleton)
    
    # Component sizes for analysis
    component_sizes = []
    for comp_id in range(1, beta_0 + 1):
        comp_size = np.sum(labeled_vessels == comp_id)
        component_sizes.append(comp_size)
    
    # ========== β1: Independent Loops ==========
    # Using Euler characteristic: β1 = β0 - χ (for 2D)
    euler_number = measure.euler_number(vessel_binary)
    beta_1_euler = beta_0 - euler_number
    
    # If graph provided, can also compute from cycle basis
    if graph is not None and graph.number_of_nodes() > 0:
        try:
            largest_cc = max(nx.connected_components(graph), key=len)
            subgraph = graph.subgraph(largest_cc)
            cycle_basis = nx.cycle_basis(subgraph)
            beta_1_graph = len(cycle_basis)
        except:
            beta_1_graph = beta_1_euler
    else:
        beta_1_graph = beta_1_euler
    
    # Use the more accurate estimate
    beta_1 = beta_1_graph if graph is not None else beta_1_euler
    
    # ========== β2: Enclosed Voids ==========
    # Find completely enclosed empty regions (avascular areas)
    background = ~vessel_binary
    labeled_bg, num_bg_regions = ndimage.label(background)
    
    # Identify exterior region (touches border)
    h, w = vessel_binary.shape
    border_mask = np.zeros_like(labeled_bg, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    
    border_labels = np.unique(labeled_bg[border_mask])
    border_labels = border_labels[border_labels > 0]
    
    # Voids are background regions not touching border
    all_labels = set(range(1, num_bg_regions + 1))
    border_labels_set = set(border_labels)
    void_labels = all_labels - border_labels_set
    
    beta_2 = len(void_labels)
    
    # Analyze void properties
    void_sizes = []
    void_locations = []
    for void_label in void_labels:
        void_mask = (labeled_bg == void_label)
        void_size = np.sum(void_mask)
        void_sizes.append(void_size)
        
        # Find centroid
        coords = np.argwhere(void_mask)
        if len(coords) > 0:
            centroid = coords.mean(axis=0)
            void_locations.append(tuple(centroid))
    
    # ========== Topological Summary ==========
    # Euler characteristic: χ = β0 - β1 + β2
    euler_calc = beta_0 - beta_1 + beta_2
    
    # Total topological features
    total_features = beta_0 + beta_1 + beta_2
    
    # Redundancy index (loops per component)
    redundancy = beta_1 / beta_0 if beta_0 > 0 else 0
    
    results = {
        # Betti numbers
        'beta_0': beta_0,
        'beta_0_skeleton': beta_0_skeleton,
        'beta_1': beta_1,
        'beta_1_euler': beta_1_euler,
        'beta_2': beta_2,
        
        # Euler characteristic
        'euler_number': euler_number,
        'euler_calculated': euler_calc,
        
        # Component analysis
        'num_components': beta_0,
        'component_sizes': component_sizes,
        'largest_component': max(component_sizes) if component_sizes else 0,
        'smallest_component': min(component_sizes) if component_sizes else 0,
        'mean_component_size': np.mean(component_sizes) if component_sizes else 0,
        
        # Loop analysis
        'num_loops': beta_1,
        'loops_per_component': redundancy,
        
        # Void analysis
        'num_voids': beta_2,
        'void_sizes': void_sizes,
        'void_locations': void_locations,
        'total_void_area': sum(void_sizes) if void_sizes else 0,
        'mean_void_size': np.mean(void_sizes) if void_sizes else 0,
        'median_void_size': np.median(void_sizes) if void_sizes else 0,
        'max_void_size': max(void_sizes) if void_sizes else 0,
        'void_area_fraction': sum(void_sizes) / vessel_binary.size if void_sizes else 0,
        
        # Summary metrics
        'total_topological_features': total_features,
        'topological_complexity': total_features / beta_0 if beta_0 > 0 else 0
    }
    
    return results


# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

for path, subdirs, files in os.walk(imgs_dir):
    for i in range(len(files)):
        print("=" * 70)
        print("Processing: " + files[i])
        print("=" * 70)
        try:
          image = Image.open(imgs_dir + files[i])
        except UnidentifiedImageError:
          print(f"  ✗ Skipped {files[i]}: Unidentified or corrupted image file.")
          print("=" * 70)
          continue
        # Skeletonize
        skeleton = thin(image)
        
        # Distance transform
        image_array = np.array(image)
        vessel_binary = (image_array > 0).astype(np.uint8)
        distance_transform = ndimage.distance_transform_edt(vessel_binary)
        
        # Extract graph
        graph = extract_graph(skeleton, image)
        print(f"Initial nodes: {graph.number_of_nodes()}")
        
        # Merge nodes
        k = True
        while k:
            tmp = graph.number_of_nodes()
            attribute = "line"
            attribute_threshold_value = 5
            to_be_removed = [(u, v) for u, v, data in
                            graph.edges(data=True)
                            if operator.le(data[attribute], attribute_threshold_value)]
            length = len(to_be_removed)
            for n in range(length):
                nodes = to_be_removed[n]
                merge_nodes_2(graph, nodes)
            
            for n1, n2, data in graph.edges(data=True):
                line = euclidean(n1, n2)
                graph[n1][n2]['line'] = line
            
            number_of_nodes = graph.number_of_nodes()
            k = tmp != number_of_nodes
        
        print(f"After merging: {graph.number_of_nodes()}")
        
        # Check connected components
        compnt_size = 1
        operators = operator.le
        
        connected_components = sorted(
            list((graph.subgraph(c) for c in nx.connected_components(graph))),
            key=lambda graph: graph.number_of_nodes()
        )
        
        to_be_removed = [subgraph for subgraph in connected_components
                        if operators(subgraph.number_of_nodes(), compnt_size)]
        
        for subgraph in to_be_removed:
            graph.remove_nodes_from(subgraph)
        
        print(f'Discarded {len(to_be_removed)} connected components')
        
        # Calculate center
        nodes = [n for n in graph.nodes()]
        x = [x for (x, y) in nodes]
        y = [y for (x, y) in nodes]
        x1 = int(np.min(x) + (np.max(x) - np.min(x)) / 2)
        y1 = int(np.min(y) + (np.max(y) - np.min(y)) / 2)
        
        print(f"Network center: ({x1}, {y1})")
        
        # Add original features
        for n1, n2, data in graph.edges(data=True):
            centerdis1 = euclidean((x1, y1), n2)
            centerdis2 = euclidean((x1, y1), n1)
            
            if centerdis1 >= centerdis2:
                centerdislow = centerdis2
                centerdishigh = centerdis1
            else:
                centerdislow = centerdis1
                centerdishigh = centerdis2
            
            graph[n1][n2]['centerdislow'] = centerdislow
            graph[n1][n2]['centerdishigh'] = centerdishigh
        
        # Add new features
        print("\n[NEW] Adding cross-sectional area features...")
        add_area_features_to_edges(graph, skeleton, distance_transform)
        
        print("[NEW] Adding branching angle features...")
        add_branching_angles_to_nodes(graph, skeleton)
        
        print("[NEW] Calculating regional areas...")
        regional_data = calculate_regional_area(image, num_regions=(2, 2))
        regional_df = pd.DataFrame(regional_data)
        #regional_name = files[i][0:6] + "_regional_area.xlsx"
        regional_name = files[i].split(".")[0] + "_regional_area.xlsx"
        regional_writer = pd.ExcelWriter(save_dir + regional_name, engine='xlsxwriter')
        regional_df.to_excel(regional_writer, index=False)
        regional_writer.close()
        print(f"Saved: {regional_name}")
        
        print("[NEW] Calculating branching angles...")
        angle_stats, angle_detailed = extract_branching_angle_features(skeleton, sample_size=500)
        angle_stats_df = pd.DataFrame([angle_stats])
        # angle_stats_name = files[i][0:6] + "_angle_statistics.xlsx"
        angle_stats_name = files[i].split(".")[0] + "_angle_statistics.xlsx"
        angle_stats_writer = pd.ExcelWriter(save_dir + angle_stats_name, engine='xlsxwriter')
        angle_stats_df.to_excel(angle_stats_writer, sheet_name='Statistics', index=False)
        
        if len(angle_detailed) > 0:
            angle_detail_rows = []
            for detail in angle_detailed:
                # Only keep x, y
                if detail['angles']:
                    for idx, angle in enumerate(detail['angles']):
                        row = {
                            'x': detail['x'],
                            'y': detail['y'],
                            'angle_index': idx,
                            'angle': angle
                        }
                        angle_detail_rows.append(row)
            
            if angle_detail_rows:
                angle_detail_df = pd.DataFrame(angle_detail_rows)
                angle_detail_df.to_excel(angle_stats_writer, sheet_name='Detailed', index=False)
        
        angle_stats_writer.close()
        print(f"Saved: {angle_stats_name}")
        
        print("[NEW] Calculating vascular density...")
        density_results = calculate_vascular_density(
            vessel_binary.astype(bool), 
            skeleton.astype(bool),
            regions=(2, 2)
        )
        
        density_global = {k: v for k, v in density_results.items() if k != 'regional_densities'}
        density_global_df = pd.DataFrame([density_global])
        
        # density_name = files[i][0:6] + "_vascular_density.xlsx"
        density_name = files[i].split(".")[0] + "_vascular_density.xlsx"
        density_writer = pd.ExcelWriter(save_dir + density_name, engine='xlsxwriter')
        density_global_df.to_excel(density_writer, sheet_name='Global', index=False)
        
        if 'regional_densities' in density_results:
            density_regional_df = pd.DataFrame(density_results['regional_densities'])
            density_regional_df.to_excel(density_writer, sheet_name='Regional', index=False)
        
        density_writer.close()
        print(f"Saved: {density_name}")
        print(f"  VAD: {density_results['vessel_area_density']:.4f}")
        print(f"  VLD: {density_results['vessel_length_density']:.6f}")
        
        print("[NEW] Calculating fractal dimension...")
        fractal_vessel = calculate_fractal_dimension_boxcount(
            vessel_binary.astype(bool), min_box_size=4, max_box_size=256
        )
        
        fractal_skeleton = calculate_fractal_dimension_boxcount(
            skeleton.astype(bool), min_box_size=4, max_box_size=256
        )
        
        fractal_data = {
            'vessel_fractal_dimension': fractal_vessel['fractal_dimension'],
            'vessel_r_squared': fractal_vessel['r_squared'],
            'skeleton_fractal_dimension': fractal_skeleton['fractal_dimension'],
            'skeleton_r_squared': fractal_skeleton['r_squared'],
            'vessel_box_sizes': str(fractal_vessel['box_sizes']),
            'vessel_box_counts': str(fractal_vessel['box_counts']),
            'skeleton_box_sizes': str(fractal_skeleton['box_sizes']),
            'skeleton_box_counts': str(fractal_skeleton['box_counts'])
        }
        
        fractal_df = pd.DataFrame([fractal_data])
        # fractal_name = files[i][0:6] + "_fractal_dimension.xlsx"
        fractal_name = files[i].split(".")[0] + "_fractal_dimension.xlsx"
        fractal_writer = pd.ExcelWriter(save_dir + fractal_name, engine='xlsxwriter')
        fractal_df.to_excel(fractal_writer, index=False)
        fractal_writer.close()
        print(f"Saved: {fractal_name}")
        print(f"  Vessel FD: {fractal_vessel['fractal_dimension']:.4f}")
        print(f"  Skeleton FD: {fractal_skeleton['fractal_dimension']:.4f}")
        
        # ========== NEW FEATURE 5: Betti Numbers ==========
        print("[NEW] Calculating Betti numbers...")
        betti_results = calculate_betti_numbers(
            vessel_binary.astype(bool),
            skeleton.astype(bool),
            graph=graph
        )
        
        # Prepare data for Excel (remove list fields for summary)
        betti_summary = {k: v for k, v in betti_results.items() 
                        if not isinstance(v, list)}
        betti_summary_df = pd.DataFrame([betti_summary])
        
        # betti_name = files[i][0:6] + "_betti_numbers.xlsx"
        betti_name = files[i].split(".")[0] + "_betti_numbers.xlsx"
        betti_writer = pd.ExcelWriter(save_dir + betti_name, engine='xlsxwriter')
        betti_summary_df.to_excel(betti_writer, sheet_name='Summary', index=False)
        
        # Add void details sheet
        if betti_results['void_sizes']:
            void_details = pd.DataFrame({
                'void_id': range(1, len(betti_results['void_sizes']) + 1),
                'size_pixels': betti_results['void_sizes'],
                'centroid_y': [loc[0] for loc in betti_results['void_locations']],
                'centroid_x': [loc[1] for loc in betti_results['void_locations']]
            })
            void_details.to_excel(betti_writer, sheet_name='Voids', index=False)
        
        betti_writer.close()
        print(f"Saved: {betti_name}")
        print(f"  β0 (Components): {betti_results['beta_0']}")
        print(f"  β1 (Loops): {betti_results['beta_1']}")
        print(f"  β2 (Voids): {betti_results['beta_2']}")
        
        # Save original data
        alldata = save_data(graph, center=False)
        # data_name = files[i][0:6] + "_alldata.xlsx"
        data_name = files[i].split(".")[0] + "_alldata.xlsx"
        writer = pd.ExcelWriter(save_dir + data_name, engine='xlsxwriter')
        alldata.to_excel(writer, index=False)
        writer.close()
        
        degreedata = save_degree(graph, x1, y1)
        # degree_name = files[i][0:6] + "_degreedata.xlsx"
        degree_name = files[i].split(".")[0] + "_degreedata.xlsx"
        degreewriter = pd.ExcelWriter(save_dir + degree_name, engine='xlsxwriter')
        degreedata.to_excel(degreewriter, index=False)
        degreewriter.close()
        
        # Draw network
        pic = draw_graph2(np.asarray(image.convert("RGB")), graph, center=False)
        # pic_name = files[i][0:6] + "_network.png"
        pic_name = files[i].split(".")[0] + "_network.png"
        plt.imsave(save_dir + pic_name, pic)
        
        print(f"\n✓ Completed {files[i]}")
        print(f"  Total output files: 8")
        print("=" * 70)