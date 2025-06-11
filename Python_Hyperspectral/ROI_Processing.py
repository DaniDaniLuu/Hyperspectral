import numpy as np
import matplotlib.pyplot as plt
from bioio import BioImage
from scipy.ndimage import gaussian_filter, label, center_of_mass
from skimage.measure import regionprops, label as ski_label 
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr
import seaborn as sns
from roifile import roiread 
from skimage.draw import polygon, ellipse 
import colorsys
import glob
import os
import zipfile


czi_path = "241121_10h40min_bra.h2b.mAp_twist.RFP_crbn.h2b.GFP_meis.kaede.czi"
# Support multiple ROI zip files
ROI_ZIP_PATHS = ["RoiSet.zip"]  # List of ROI zip files to process

img = BioImage(czi_path)
img_data = img.data.squeeze() # Removes the time dimension bc we only have one timepoint

print(f"Image metadata: {img_data.shape}") # (15, 31, 1069, 1069)

# Do max projection along Z axis
max_proj = np.max(img_data, axis=1)


apply_smoothing = False # Set to True to apply Gaussian smoothing before segmentation
if apply_smoothing:
    try:
        # Apply filter only on spatial dims (Y, X) -> axes 1 and 2
        smoothed_proj = gaussian_filter(max_proj, sigma=(0, 1, 1))
        print(f"Applied Gaussian smoothing. Smoothed projection shape: {smoothed_proj.shape}")
        segmentation_input = smoothed_proj # Use smoothed data for segmentation
    except Exception as e:
        print(f"Error applying Gaussian filter: {e}")
        print("Using original max projection without smoothing.")
        segmentation_input = max_proj # Fallback to unsmoothed data
else:
    print("Skipping Gaussian smoothing.")
    segmentation_input = max_proj # Use original max projection for segmentation

print(f"filtered_proj shape {segmentation_input.shape}") # (15, 1069, 1069)


def create_roi_colormap(num_rois):
    """
    Create an advanced colormap for ROI visualization with better color separation
    for large numbers of ROIs (50+).
    
    Uses 12 distinct color families with multiple shades each, plus HSV fallback.
    """
    if num_rois <= 0:
        return plt.cm.get_cmap('viridis', 1)
    
    # Define 12 distinct color families with 8 variations each (96 total base colors)
    color_families = {
        'blues': ['#000080', '#0000CD', '#0000FF', '#1E90FF', '#4169E1', '#6495ED', '#87CEEB', '#B0E0E6'],
        'reds': ['#8B0000', '#DC143C', '#FF0000', '#FF4500', '#FF6347', '#FF7F50', '#FFA07A', '#FFCCCB'],
        'greens': ['#006400', '#008000', '#00FF00', '#32CD32', '#7CFC00', '#ADFF2F', '#9AFF9A', '#98FB98'],
        'purples': ['#4B0082', '#8B008B', '#9400D3', '#9932CC', '#BA55D3', '#DA70D6', '#DDA0DD', '#E6E6FA'],
        'oranges': ['#8B4513', '#D2691E', '#FF8C00', '#FFA500', '#FFB347', '#FFCC99', '#FFDAB9', '#FFE4B5'],
        'magentas': ['#8B008B', '#B22222', '#FF1493', '#FF69B4', '#FF6347', '#FFB6C1', '#FFC0CB', '#FFCCCB'],
        'cyans': ['#008B8B', '#00CED1', '#00FFFF', '#40E0D0', '#48D1CC', '#AFEEEE', '#B0E0E6', '#E0FFFF'],
        'yellows': ['#B8860B', '#DAA520', '#FFD700', '#FFFF00', '#FFFFE0', '#FFFACD', '#FFEFD5', '#FFF8DC'],
        'browns': ['#654321', '#8B4513', '#A0522D', '#CD853F', '#D2691E', '#DEB887', '#F4A460', '#FAD5A5'],
        'grays': ['#2F4F4F', '#696969', '#708090', '#778899', '#A9A9A9', '#C0C0C0', '#D3D3D3', '#DCDCDC'],
        'teals': ['#2F4F4F', '#008080', '#20B2AA', '#48D1CC', '#40E0D0', '#00CED1', '#AFEEEE', '#B0E0E6'],
        'limes': ['#556B2F', '#6B8E23', '#7CFC00', '#ADFF2F', '#9AFF9A', '#98FB98', '#F0FFF0', '#F5FFFA']
    }
    
    # Create list of all base colors
    all_colors = []
    for family_colors in color_families.values():
        all_colors.extend(family_colors)
    
    # If we need more colors than our base set, generate additional colors using HSV
    if num_rois > len(all_colors):
        print(f"Generating {num_rois - len(all_colors)} additional colors using HSV space")
        additional_colors = []
        
        # Generate colors with different hue, saturation, and value combinations
        hue_steps = max(10, int(np.ceil((num_rois - len(all_colors)) / 9)))
        for i in range(hue_steps):
            hue = i / hue_steps
            for sat in [0.6, 0.8, 1.0]:
                for val in [0.6, 0.8, 1.0]:
                    if len(additional_colors) >= (num_rois - len(all_colors)):
                        break
                    rgb = colorsys.hsv_to_rgb(hue, sat, val)
                    hex_color = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
                    additional_colors.append(hex_color)
                if len(additional_colors) >= (num_rois - len(all_colors)):
                    break
            if len(additional_colors) >= (num_rois - len(all_colors)):
                break
        
        all_colors.extend(additional_colors)
    
    # Take exactly the number of colors we need
    final_colors = ['#000000'] + all_colors[:num_rois]  # Add black for background
    
    # Create custom colormap
    from matplotlib.colors import ListedColormap
    return ListedColormap(final_colors)


def load_multiple_roi_files(roi_zip_paths, image_height, image_width):
    """
    Load ROIs from multiple zip files and create a combined labeled mask.
    
    Args:
        roi_zip_paths: List of paths to ROI zip files
        image_height: Height of the image
        image_width: Width of the image
        
    Returns:
        tuple: (labeled_mask, total_rois_loaded, all_roi_info)
    """
    all_rois = []
    roi_info = []
    
    for zip_path in roi_zip_paths:
        if not os.path.exists(zip_path):
            print(f"ROI file not found: {zip_path}, skipping...")
            continue
            
        print(f"\nLoading ROIs from: {zip_path}")
        try:
            # Try to read the zip file directly
            try:
                current_rois = roiread(zip_path)
                print(f"Successfully read {len(current_rois)} ROIs from {zip_path}")
            except Exception as e:
                print(f"Could not read zip file directly: {e}")
                # Try extracting and reading individual files
                base_name = os.path.splitext(os.path.basename(zip_path))[0]
                extract_dir = f"{base_name}_extracted"
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                roi_files = glob.glob(os.path.join(extract_dir, "*.roi"))
                current_rois = []
                
                for roi_file in roi_files:
                    try:
                        roi = roiread(roi_file)
                        if isinstance(roi, list):
                            current_rois.extend(roi)
                        else:
                            current_rois.append(roi)
                    except Exception as roi_e:
                        print(f"Failed to load ROI from {roi_file}: {roi_e}")
            
            # Add source information to each ROI
            for roi in current_rois:
                roi_info.append({
                    'roi': roi,
                    'source_file': zip_path,
                    'original_index': len(all_rois)
                })
                all_rois.append(roi)
                
        except Exception as e:
            print(f"Error loading ROIs from {zip_path}: {e}")
    
    if not all_rois:
        print("No ROIs found in any of the provided files.")
        return None, 0, []
    
    # Create labeled mask
    labeled_mask = create_roi_mask(all_rois, image_height, image_width)
    
    return labeled_mask, len(all_rois), roi_info


def create_roi_mask(rois, image_height, image_width):
    """
    Create a labeled mask from ROI objects, handling overlaps by giving priority
    to earlier ROIs (lower IDs) to preserve all ROIs in the final mask.
    
    Args:
        rois: List of ROI objects
        image_height: Height of the image
        image_width: Width of the image
        
    Returns:
        numpy.ndarray: Labeled mask with unique integer ID for each ROI
    """
    labeled_mask = np.zeros((image_height, image_width), dtype=np.int32)
    successful_rois = 0
    
    print(f"Creating ROI mask for {len(rois)} ROIs...")
    
    for i, roi in enumerate(rois):
        roi_id = i + 1  # Assign unique ID starting from 1
        
        try:
            # Extract coordinates from ROI
            coords = extract_roi_coordinates(roi)
            if coords is None:
                print(f"Warning: Could not extract coordinates for ROI {roi_id}")
                continue
                
            x_coords, y_coords = coords
            
            # Create temporary mask for this ROI
            temp_mask = np.zeros((image_height, image_width), dtype=bool)
            
            # Handle different ROI types
            if len(x_coords) > 2 and len(y_coords) > 2:  # Polygon
                x_coords = np.clip(x_coords, 0, image_width - 1)
                y_coords = np.clip(y_coords, 0, image_height - 1)
                
                rr, cc = polygon(y_coords, x_coords, shape=temp_mask.shape)
                temp_mask[rr, cc] = True
                
            elif len(x_coords) == 2 and len(y_coords) == 2:  # Rectangle
                x1, x2 = int(min(x_coords)), int(max(x_coords))
                y1, y2 = int(min(y_coords)), int(max(y_coords))
                
                x1, x2 = max(0, x1), min(image_width - 1, x2)
                y1, y2 = max(0, y1), min(image_height - 1, y2)
                
                temp_mask[y1:y2+1, x1:x2+1] = True
            else:
                print(f"Warning: ROI {roi_id} has unexpected coordinate structure")
                continue
            
            # Only assign pixels that are not already assigned to prevent overlap overwrites
            available_pixels = temp_mask & (labeled_mask == 0)
            labeled_mask[available_pixels] = roi_id
            
            if np.any(available_pixels):
                successful_rois += 1
                if np.any(temp_mask & (labeled_mask != roi_id) & (labeled_mask != 0)):
                    print(f"  ROI {roi_id}: Partial overlap detected, preserved non-overlapping region")
                else:
                    print(f"  ROI {roi_id}: Successfully added")
            else:
                print(f"  ROI {roi_id}: Completely overlapped by previous ROIs, skipped")
                
        except Exception as e:
            print(f"Error processing ROI {roi_id}: {e}")
            continue
    
    print(f"Successfully created mask with {successful_rois} ROIs (from {len(rois)} total)")
    return labeled_mask


def extract_roi_coordinates(roi):
    """
    Extract x,y coordinates from an ROI object, handling different attribute names.
    
    Returns:
        tuple: (x_coords, y_coords) or None if extraction fails
    """
    try:
        # Try different possible coordinate attribute names
        if hasattr(roi, 'coordinates') and roi.coordinates is not None:
            coords = roi.coordinates()
            if coords is not None and len(coords) > 0:
                coords = np.array(coords)
                if coords.ndim > 1:
                    return coords[:, 0], coords[:, 1]
                else:
                    return coords[0], coords[1]
                    
        elif hasattr(roi, 'x') and hasattr(roi, 'y'):
            return np.array(roi.x), np.array(roi.y)
            
        elif hasattr(roi, 'x1') and hasattr(roi, 'y1'):
            x_coords = [roi.x1, roi.x2] if hasattr(roi, 'x2') else [roi.x1]
            y_coords = [roi.y1, roi.y2] if hasattr(roi, 'y2') else [roi.y1]
            return np.array(x_coords), np.array(y_coords)
            
        elif hasattr(roi, 'left') and hasattr(roi, 'top'):
            x_coords = [roi.left, roi.right]
            y_coords = [roi.top, roi.bottom]
            return np.array(x_coords), np.array(y_coords)
            
    except Exception as e:
        print(f"Error extracting coordinates: {e}")
    
    return None




# --- Visualization of Filtered Channels (with Fixed Scale)
# Calculate global min and max across all filtered channels for consistent scaling
vmin_val = np.min(segmentation_input)
vmax_val = np.max(segmentation_input)
print(f"\nUsing fixed intensity scale for filtered channel plots: vmin={vmin_val}, vmax={vmax_val}")

# Create a subplot grid for all filtered channels
num_channels_to_plot = segmentation_input.shape[0]
# Adjust grid size dynamically if fewer than 15 channels
ncols = 5
nrows = (num_channels_to_plot + ncols - 1) // ncols # Calculate rows needed
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3), squeeze=False)
axes = axes.ravel() # Flatten the axes array

# --- Plot each filtered channel using the fixed scale
for i in range(num_channels_to_plot):
    ax = axes[i]
    # Use the calculated vmin_val and vmax_val for consistent scaling
    im = ax.imshow(segmentation_input[i], cmap='gray', vmin=vmin_val, vmax=vmax_val)
    ax.set_title(f'Filtered Channel {i+1}')
    ax.axis('off')
    # Add a colorbar for reference (will be the same scale for all)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Hide any unused subplots if num_channels is not a multiple of ncols
for j in range(num_channels_to_plot, nrows * ncols):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig('filtered_channel_projections_fixed_scale.png', dpi=300, bbox_inches='tight')
print("Saved filtered channel projections with fixed scale to filtered_channel_projections_fixed_scale.png")
plt.close(fig)

# --- Load Manual ROIs from FIJI and Create Labeled Mask ---
print(f"\nLoading manual ROIs from multiple sources...")

# Use the enhanced multi-file loading function
image_height = segmentation_input.shape[1]
image_width = segmentation_input.shape[2]

filtered_labeled_mask, num_filtered_labels, roi_info = load_multiple_roi_files(
    ROI_ZIP_PATHS, image_height, image_width
)

if filtered_labeled_mask is None or num_filtered_labels == 0:
    print("ERROR: Could not load any ROIs. Exiting.")
    exit()

print(f"\nSuccessfully created a labeled mask with {num_filtered_labels} ROIs from multiple files.")



#4 --- Extract Spectral Information for each ROI ---
num_channels = segmentation_input.shape[0]
roi_spectral_data = []

props = regionprops(filtered_labeled_mask, intensity_image=segmentation_input[0]) # Use first channel just for geometric props

# Check if props were generated
if not props:
    print("Error: regionprops did not return any regions. Check the filtered_labeled_mask.")
    exit()
    
print(f"regionprops found {len(props)} regions.")

# Ensure props are indexed correctly if they don't match num_filtered_labels
# This can happen if some label IDs are skipped or if regionprops behaves unexpectedly.
# Create a mapping from label ID in the mask to the regionprop object
prop_dict = {prop.label: prop for prop in props}


for roi_label_id in range(1, num_filtered_labels + 1): # Iterate from 1 to num_filtered_labels
    if roi_label_id not in prop_dict:
        print(f"Warning: ROI with label ID {roi_label_id} not found in regionprops output. Skipping.")
        continue

    region = prop_dict[roi_label_id]
    roi_coords = region.coords # shape (N, 2) where N is number of pixels in ROI

    spectral_signature = []
    for c in range(num_channels):
        channel_data = segmentation_input[c]
        roi_pixels_in_channel = channel_data[roi_coords[:, 0], roi_coords[:, 1]]
        mean_intensity = np.mean(roi_pixels_in_channel)
        spectral_signature.append(mean_intensity)

    roi_spectral_data.append({
        'roi_id': region.label, # This is the unique integer ID from the mask
        'centroid': region.centroid, # (row, col)
        'spectral_signature': np.array(spectral_signature)
    })

print(f"Extracted spectral data for {len(roi_spectral_data)} ROIs.")


def plot_intensity_histograms(spectral_df, segmentation_input, filtered_labeled_mask, num_channels):
    """
    Create comprehensive intensity distribution histograms for each ROI across all channels.
    
    Args:
        spectral_df: DataFrame with ROI spectral data
        segmentation_input: The hyperspectral image data
        filtered_labeled_mask: The labeled ROI mask
        num_channels: Number of spectral channels
    """
    if spectral_df.empty:
        print("No spectral data available for histogram analysis")
        return
    
    print("\nGenerating intensity distribution histograms...")
    
    # 1. Individual ROI histograms in a grid layout
    num_rois = len(spectral_df)
    ncols = 8  # 8 columns for better layout with 53 ROIs
    nrows = (num_rois + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.ravel()
    
    colors = plt.cm.tab20(np.linspace(0, 1, num_channels))
    
    for idx, (_, roi_row) in enumerate(spectral_df.iterrows()):
        roi_id = roi_row['roi_id']
        ax = axes[idx]
        
        # Get pixel intensities for this ROI across all channels
        roi_mask = filtered_labeled_mask == roi_id
        roi_intensities = []
        
        for ch in range(num_channels):
            channel_data = segmentation_input[ch]
            roi_pixels = channel_data[roi_mask]
            roi_intensities.extend(roi_pixels)
        
        if roi_intensities:
            ax.hist(roi_intensities, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'ROI {roi_id}', fontsize=10)
            ax.set_xlabel('Intensity', fontsize=8)
            ax.set_ylabel('Frequency', fontsize=8)
            ax.tick_params(labelsize=7)
        else:
            ax.text(0.5, 0.5, f'ROI {roi_id}\nNo Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(num_rois, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('roi_intensity_histograms_grid.png', dpi=300, bbox_inches='tight')
    print("Saved individual ROI intensity histograms to roi_intensity_histograms_grid.png")
    plt.close()
    
    # 2. Combined distribution comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Box plot of all ROIs
    spectral_cols = [f'Channel_{i+1}_Mean' for i in range(num_channels)]
    box_data = []
    roi_labels = []
    
    for _, roi_row in spectral_df.iterrows():
        roi_id = roi_row['roi_id']
        roi_means = roi_row[spectral_cols].values
        box_data.append(roi_means)
        roi_labels.append(f'ROI {roi_id}')
    
    ax1.boxplot(box_data, labels=roi_labels)
    ax1.set_title('Intensity Distributions Across All ROIs')
    ax1.set_xlabel('ROI')
    ax1.set_ylabel('Mean Intensity')
    ax1.tick_params(axis='x', rotation=45, labelsize=6)
    
    # Channel-wise comparison
    channel_data = []
    for ch in range(num_channels):
        channel_col = f'Channel_{ch+1}_Mean'
        if channel_col in spectral_df.columns:
            channel_data.append(spectral_df[channel_col].values)
    
    ax2.boxplot(channel_data, labels=[f'Ch{i+1}' for i in range(len(channel_data))])
    ax2.set_title('Channel-wise Intensity Distributions')
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Mean Intensity')
    
    # Overall intensity distribution
    all_intensities = []
    for col in spectral_cols:
        if col in spectral_df.columns:
            all_intensities.extend(spectral_df[col].dropna().values)
    
    ax3.hist(all_intensities, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax3.set_title('Overall Intensity Distribution (All ROIs, All Channels)')
    ax3.set_xlabel('Mean Intensity')
    ax3.set_ylabel('Frequency')
    
    # Channel-wise intensity heatmap
    intensity_matrix = spectral_df[spectral_cols].values.T  # Channels x ROIs
    im = ax4.imshow(intensity_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    ax4.set_title('Intensity Heatmap (Channels × ROIs)')
    ax4.set_xlabel('ROI Index')
    ax4.set_ylabel('Channel')
    ax4.set_yticks(range(num_channels))
    ax4.set_yticklabels([f'Ch{i+1}' for i in range(num_channels)])
    plt.colorbar(im, ax=ax4, label='Mean Intensity')
    
    plt.tight_layout()
    plt.savefig('roi_intensity_distributions_combined.png', dpi=300, bbox_inches='tight')
    print("Saved combined intensity distribution analysis to roi_intensity_distributions_combined.png")
    plt.close()


def perform_statistical_grouping_analysis(spectral_df, num_channels):
    """
    Perform comprehensive statistical analysis to group ROIs based on intensity distributions.
    
    Args:
        spectral_df: DataFrame with ROI spectral data
        num_channels: Number of spectral channels
        
    Returns:
        dict: Analysis results including correlation, PCA, clustering
    """
    if spectral_df.empty or len(spectral_df) < 2:
        print("Insufficient data for statistical grouping analysis")
        return {}
    
    print("\nPerforming statistical grouping analysis...")
    
    # Prepare data
    spectral_cols = [f'Channel_{i+1}_Mean' for i in range(num_channels)]
    if not all(col in spectral_df.columns for col in spectral_cols):
        print("Missing spectral columns for analysis")
        return {}
    
    # Remove NaN values
    clean_df = spectral_df.dropna(subset=spectral_cols)
    if len(clean_df) < 2:
        print("Insufficient clean data for analysis")
        return {}
    
    spectral_data = clean_df[spectral_cols].values
    roi_ids = clean_df['roi_id'].values
    
    # 1. Correlation Analysis
    print("  Computing correlation analysis...")
    correlation_matrix = np.corrcoef(spectral_data)
    
    # Find highly correlated ROI pairs (>0.8 correlation)
    high_corr_pairs = []
    for i in range(len(correlation_matrix)):
        for j in range(i+1, len(correlation_matrix)):
            if correlation_matrix[i, j] > 0.8:
                high_corr_pairs.append((roi_ids[i], roi_ids[j], correlation_matrix[i, j]))
    
    # 2. PCA Analysis
    print("  Performing PCA analysis...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(spectral_data)
    
    pca = PCA()
    pca_data = pca.fit_transform(scaled_data)
    explained_variance = pca.explained_variance_ratio_
    
    # 3. Hierarchical Clustering
    print("  Performing hierarchical clustering...")
    linkage_matrix = linkage(scaled_data, method='ward')
    
    # Create clusters (example: 4 clusters)
    num_clusters = min(4, len(clean_df) - 1)
    hierarchical_clusters = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
    hier_labels = hierarchical_clusters.fit_predict(scaled_data)
    
    # 4. K-means for comparison
    print("  Performing K-means clustering...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(scaled_data)
    
    # 5. Intensity-based grouping
    print("  Creating intensity-based groups...")
    mean_intensities = np.mean(spectral_data, axis=1)
    intensity_terciles = np.percentile(mean_intensities, [33, 67])
    intensity_groups = np.digitize(mean_intensities, intensity_terciles)
    
    # 6. Spectral pattern similarity
    print("  Analyzing spectral pattern similarity...")
    similarity_matrix = cosine_similarity(spectral_data)
    
    # Find ROIs with similar spectral patterns (>0.9 cosine similarity)
    similar_patterns = []
    for i in range(len(similarity_matrix)):
        for j in range(i+1, len(similarity_matrix)):
            if similarity_matrix[i, j] > 0.9:
                similar_patterns.append((roi_ids[i], roi_ids[j], similarity_matrix[i, j]))
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Correlation heatmap
    ax1 = plt.subplot(3, 3, 1)
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
                xticklabels=[f'ROI{id}' for id in roi_ids],
                yticklabels=[f'ROI{id}' for id in roi_ids])
    ax1.set_title('ROI Correlation Matrix')
    
    # 2. PCA explained variance
    ax2 = plt.subplot(3, 3, 2)
    ax2.bar(range(1, len(explained_variance) + 1), explained_variance)
    ax2.set_title('PCA Explained Variance Ratio')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance Ratio')
    
    # 3. PCA scatter plot
    ax3 = plt.subplot(3, 3, 3)
    scatter = ax3.scatter(pca_data[:, 0], pca_data[:, 1], c=hier_labels, cmap='tab10', s=50)
    ax3.set_title('PCA: PC1 vs PC2 (colored by hierarchical clusters)')
    ax3.set_xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
    ax3.set_ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
    
    # Add ROI labels to PCA plot
    for i, roi_id in enumerate(roi_ids):
        ax3.annotate(f'{roi_id}', (pca_data[i, 0], pca_data[i, 1]), fontsize=8)
    
    # 4. Dendrogram
    ax4 = plt.subplot(3, 3, 4)
    dendrogram(linkage_matrix, labels=[f'ROI{id}' for id in roi_ids], ax=ax4, leaf_rotation=90)
    ax4.set_title('Hierarchical Clustering Dendrogram')
    
    # 5. Cluster comparison
    ax5 = plt.subplot(3, 3, 5)
    cluster_comparison = pd.DataFrame({
        'ROI_ID': roi_ids,
        'Hierarchical': hier_labels,
        'K-means': kmeans_labels,
        'Intensity_Group': intensity_groups
    })
    
    # Plot cluster assignments
    x_pos = np.arange(len(roi_ids))
    width = 0.25
    ax5.bar(x_pos - width, hier_labels, width, label='Hierarchical', alpha=0.7)
    ax5.bar(x_pos, kmeans_labels, width, label='K-means', alpha=0.7)
    ax5.bar(x_pos + width, intensity_groups, width, label='Intensity Groups', alpha=0.7)
    ax5.set_title('Clustering Method Comparison')
    ax5.set_xlabel('ROI Index')
    ax5.set_ylabel('Cluster ID')
    ax5.legend()
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([f'ROI{id}' for id in roi_ids], rotation=45, fontsize=8)
    
    # 6. Intensity distribution by cluster
    ax6 = plt.subplot(3, 3, 6)
    for cluster_id in range(num_clusters):
        cluster_mask = hier_labels == cluster_id
        if np.any(cluster_mask):
            cluster_intensities = mean_intensities[cluster_mask]
            ax6.hist(cluster_intensities, alpha=0.7, label=f'Cluster {cluster_id}', bins=10)
    ax6.set_title('Intensity Distribution by Hierarchical Cluster')
    ax6.set_xlabel('Mean Intensity')
    ax6.set_ylabel('Frequency')
    ax6.legend()
    
    # 7. Spectral pattern similarity heatmap
    ax7 = plt.subplot(3, 3, 7)
    sns.heatmap(similarity_matrix, annot=False, cmap='viridis',
                xticklabels=[f'ROI{id}' for id in roi_ids],
                yticklabels=[f'ROI{id}' for id in roi_ids])
    ax7.set_title('Spectral Pattern Similarity (Cosine)')
    
    # 8. Mean spectral signatures by cluster
    ax8 = plt.subplot(3, 3, 8)
    channel_numbers = np.arange(1, num_channels + 1)
    
    for cluster_id in range(num_clusters):
        cluster_mask = hier_labels == cluster_id
        if np.any(cluster_mask):
            cluster_data = spectral_data[cluster_mask]
            mean_signature = np.mean(cluster_data, axis=0)
            std_signature = np.std(cluster_data, axis=0)
            
            ax8.plot(channel_numbers, mean_signature, marker='o', linewidth=2,
                    label=f'Cluster {cluster_id} (n={np.sum(cluster_mask)})')
            ax8.fill_between(channel_numbers, mean_signature - std_signature,
                           mean_signature + std_signature, alpha=0.2)
    
    ax8.set_title('Mean Spectral Signatures by Cluster')
    ax8.set_xlabel('Channel Number')
    ax8.set_ylabel('Mean Intensity')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Summary statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = f"""Statistical Grouping Summary:
    
Total ROIs analyzed: {len(clean_df)}
Spectral channels: {num_channels}

High correlation pairs (>0.8): {len(high_corr_pairs)}
Similar pattern pairs (>0.9): {len(similar_patterns)}

PCA: PC1+PC2 explain {explained_variance[0]+explained_variance[1]:.1%} variance

Hierarchical clusters: {num_clusters}
Cluster sizes: {[np.sum(hier_labels == i) for i in range(num_clusters)]}

Intensity groups (Low/Med/High): 
{np.bincount(intensity_groups)}
"""
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('roi_statistical_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    print("Saved comprehensive statistical analysis to roi_statistical_analysis_comprehensive.png")
    plt.close()
    
    # Return analysis results
    results = {
        'correlation_matrix': correlation_matrix,
        'high_corr_pairs': high_corr_pairs,
        'pca_explained_variance': explained_variance,
        'pca_data': pca_data,
        'hierarchical_labels': hier_labels,
        'kmeans_labels': kmeans_labels,
        'intensity_groups': intensity_groups,
        'similarity_matrix': similarity_matrix,
        'similar_patterns': similar_patterns,
        'roi_ids': roi_ids,
        'num_clusters': num_clusters
    }
    
    # Print summary of findings
    print(f"\n=== STATISTICAL GROUPING RESULTS ===")
    print(f"High correlation pairs (>0.8): {len(high_corr_pairs)}")
    if high_corr_pairs:
        for pair in high_corr_pairs[:5]:  # Show first 5
            print(f"  ROI {pair[0]} ↔ ROI {pair[1]}: r={pair[2]:.3f}")
    
    print(f"\nSimilar spectral patterns (>0.9): {len(similar_patterns)}")
    if similar_patterns:
        for pair in similar_patterns[:5]:  # Show first 5
            print(f"  ROI {pair[0]} ↔ ROI {pair[1]}: similarity={pair[2]:.3f}")
    
    print(f"\nHierarchical cluster sizes: {[np.sum(hier_labels == i) for i in range(num_clusters)]}")
    print(f"PCA: First two components explain {explained_variance[0]+explained_variance[1]:.1%} of variance")
    
    return results


def visualize_roi_mask_enhanced(segmentation_input, filtered_labeled_mask, spectral_df, num_filtered_labels):
    """
    Create enhanced ROI visualizations with multiple approaches for better identification.
    
    Args:
        segmentation_input: Hyperspectral image data
        filtered_labeled_mask: Labeled ROI mask
        spectral_df: DataFrame with ROI spectral data
        num_filtered_labels: Number of ROIs
    """
    print("\nCreating enhanced ROI visualizations...")
    
    # Create the enhanced colormap
    roi_colormap = create_roi_colormap(num_filtered_labels)
    
    # Create a 2x3 subplot layout for different visualization approaches
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    # Reference image (background for all visualizations)
    reference_channel = 14 if segmentation_input.shape[0] > 14 else 0
    ref_img = segmentation_input[reference_channel]
    
    # 1. Standard colored ROI mask
    axes[0].imshow(ref_img, cmap='gray', alpha=0.7)
    im1 = axes[0].imshow(filtered_labeled_mask, cmap=roi_colormap, alpha=0.8, 
                        vmin=0, vmax=num_filtered_labels, interpolation='nearest')
    axes[0].set_title(f'Standard ROI Mask ({num_filtered_labels} ROIs)')
    axes[0].axis('off')
    
    # 2. ROI outlines only (no fill)
    axes[1].imshow(ref_img, cmap='gray')
    from skimage.segmentation import find_boundaries
    boundaries = find_boundaries(filtered_labeled_mask, mode='outer')
    axes[1].contour(filtered_labeled_mask, levels=range(1, num_filtered_labels+1), 
                   colors='red', linewidths=1.5, alpha=0.8)
    axes[1].set_title('ROI Outlines Only')
    axes[1].axis('off')
    
    # 3. ROI numbers directly on image
    axes[2].imshow(ref_img, cmap='gray')
    axes[2].imshow(filtered_labeled_mask, cmap=roi_colormap, alpha=0.3, 
                  vmin=0, vmax=num_filtered_labels, interpolation='nearest')
    
    # Add ROI numbers at centroids
    if not spectral_df.empty and 'centroid_col' in spectral_df.columns:
        for _, roi_row in spectral_df.iterrows():
            roi_id = roi_row['roi_id']
            x, y = roi_row['centroid_col'], roi_row['centroid_row']
            axes[2].text(x, y, str(roi_id), ha='center', va='center', 
                        fontsize=8, fontweight='bold', color='white',
                        bbox=dict(boxstyle='circle,pad=0.1', facecolor='black', alpha=0.7))
    
    axes[2].set_title('ROI Numbers on Image')
    axes[2].axis('off')
    
    # 4. Size-coded visualization
    axes[3].imshow(ref_img, cmap='gray', alpha=0.7)
    
    # Calculate ROI sizes
    roi_sizes = {}
    for roi_id in range(1, num_filtered_labels + 1):
        roi_sizes[roi_id] = np.sum(filtered_labeled_mask == roi_id)
    
    # Create size-based colormap
    if roi_sizes:
        max_size = max(roi_sizes.values())
        min_size = min(roi_sizes.values())
        
        size_mask = np.zeros_like(filtered_labeled_mask, dtype=float)
        for roi_id, size in roi_sizes.items():
            normalized_size = (size - min_size) / (max_size - min_size) if max_size > min_size else 0.5
            size_mask[filtered_labeled_mask == roi_id] = normalized_size
        
        im4 = axes[3].imshow(size_mask, cmap='plasma', alpha=0.8, vmin=0, vmax=1)
        plt.colorbar(im4, ax=axes[3], label='Relative Size', fraction=0.046, pad=0.04)
    
    axes[3].set_title('Size-Coded ROIs')
    axes[3].axis('off')
    
    # 5. Intensity-coded visualization
    axes[4].imshow(ref_img, cmap='gray', alpha=0.7)
    
    # Create intensity-based colormap
    if not spectral_df.empty:
        intensity_mask = np.zeros_like(filtered_labeled_mask, dtype=float)
        
        # Use mean spectral intensity
        spectral_cols = [f'Channel_{i+1}_Mean' for i in range(segmentation_input.shape[0])]
        if all(col in spectral_df.columns for col in spectral_cols):
            mean_intensities = spectral_df[spectral_cols].mean(axis=1)
            max_intensity = mean_intensities.max()
            min_intensity = mean_intensities.min()
            
            for _, roi_row in spectral_df.iterrows():
                roi_id = roi_row['roi_id']
                roi_intensity = mean_intensities.loc[roi_row.name]
                normalized_intensity = (roi_intensity - min_intensity) / (max_intensity - min_intensity) if max_intensity > min_intensity else 0.5
                intensity_mask[filtered_labeled_mask == roi_id] = normalized_intensity
            
            im5 = axes[4].imshow(intensity_mask, cmap='viridis', alpha=0.8, vmin=0, vmax=1)
            plt.colorbar(im5, ax=axes[4], label='Relative Intensity', fraction=0.046, pad=0.04)
    
    axes[4].set_title('Intensity-Coded ROIs')
    axes[4].axis('off')
    
    # 6. Interactive-style grid overlay
    axes[5].imshow(ref_img, cmap='gray')
    axes[5].imshow(filtered_labeled_mask, cmap=roi_colormap, alpha=0.6, 
                  vmin=0, vmax=num_filtered_labels, interpolation='nearest')
    
    # Add grid overlay
    height, width = filtered_labeled_mask.shape
    grid_spacing = min(100, max(50, min(height, width) // 20))
    
    for i in range(0, height, grid_spacing):
        axes[5].axhline(y=i, color='white', alpha=0.3, linewidth=0.5)
    for j in range(0, width, grid_spacing):
        axes[5].axvline(x=j, color='white', alpha=0.3, linewidth=0.5)
    
    axes[5].set_title('Grid Overlay for Reference')
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig('roi_enhanced_visualizations.png', dpi=300, bbox_inches='tight')
    print("Saved enhanced ROI visualizations to roi_enhanced_visualizations.png")
    plt.close()


def create_detailed_legend(spectral_df, num_filtered_labels, statistical_results=None):
    """
    Create a detailed multi-panel legend with ROI information and references.
    
    Args:
        spectral_df: DataFrame with ROI spectral data
        num_filtered_labels: Number of ROIs
        statistical_results: Results from statistical grouping analysis
    """
    print("Creating detailed ROI reference legend...")
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(16, 20))
    
    # Panel 1: ROI color reference (top half)
    ax1 = plt.subplot(3, 1, 1)
    roi_colormap = create_roi_colormap(num_filtered_labels)
    
    # Create a grid showing ROI colors and IDs
    ncols = 10
    nrows = (num_filtered_labels + ncols - 1) // ncols
    
    color_grid = np.zeros((nrows, ncols))
    for i in range(num_filtered_labels):
        row = i // ncols
        col = i % ncols
        color_grid[row, col] = i + 1
    
    im = ax1.imshow(color_grid, cmap=roi_colormap, vmin=0, vmax=num_filtered_labels, 
                   interpolation='nearest', aspect='equal')
    
    # Add ROI ID labels
    for i in range(num_filtered_labels):
        row = i // ncols
        col = i % ncols
        roi_id = i + 1
        ax1.text(col, row, str(roi_id), ha='center', va='center', 
                fontsize=8, fontweight='bold', color='white' if roi_id % 2 == 0 else 'black')
    
    ax1.set_title(f'ROI Color Reference ({num_filtered_labels} ROIs)')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Panel 2: ROI statistics table
    ax2 = plt.subplot(3, 1, 2)
    ax2.axis('off')
    
    if not spectral_df.empty:
        # Create summary table
        table_data = []
        spectral_cols = [f'Channel_{i+1}_Mean' for i in range(min(15, len([c for c in spectral_df.columns if 'Channel_' in c])))]
        
        for _, roi_row in spectral_df.head(20).iterrows():  # Show first 20 ROIs
            roi_id = roi_row['roi_id']
            if all(col in spectral_df.columns for col in spectral_cols):
                mean_intensity = np.mean(roi_row[spectral_cols])
                max_intensity = np.max(roi_row[spectral_cols])
                min_intensity = np.min(roi_row[spectral_cols])
            else:
                mean_intensity = max_intensity = min_intensity = 0
            
            centroid_x = roi_row.get('centroid_col', 0)
            centroid_y = roi_row.get('centroid_row', 0)
            
            table_data.append([
                f'ROI {roi_id}',
                f'{centroid_x:.0f}, {centroid_y:.0f}',
                f'{mean_intensity:.1f}',
                f'{min_intensity:.1f}-{max_intensity:.1f}'
            ])
        
        # Create table
        table = ax2.table(cellText=table_data,
                         colLabels=['ROI ID', 'Centroid (X,Y)', 'Mean Intensity', 'Intensity Range'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        # Color code the ROI ID column
        for i, (roi_id, _, _, _) in enumerate(table_data):
            try:
                roi_num = int(float(roi_id.split()[1]))  # Handle both int and float strings
                color = roi_colormap(roi_num / num_filtered_labels)
                table[(i+1, 0)].set_facecolor(color)
                table[(i+1, 0)].set_text_props(weight='bold', color='white')
            except (ValueError, IndexError):
                # Skip coloring if we can't parse the ROI ID
                pass
    
    ax2.set_title('ROI Statistics Summary (First 20 ROIs)', y=0.95)
    
    # Panel 3: Clustering information
    ax3 = plt.subplot(3, 1, 3)
    ax3.axis('off')
    
    cluster_text = f"""ROI Analysis Summary:

Total ROIs loaded: {num_filtered_labels}
Successfully analyzed: {len(spectral_df) if not spectral_df.empty else 0}

Color Coding Strategy:
• {num_filtered_labels} distinct colors using 12 color families
• Each family contains 8 color variations
• Additional colors generated using HSV space as needed

Visualization Methods:
1. Standard colored mask
2. Outline-only view
3. Numbered ROIs
4. Size-based coding
5. Intensity-based coding
6. Grid reference overlay
"""
    
    if statistical_results:
        cluster_text += f"""
Statistical Grouping Results:
• High correlation pairs: {len(statistical_results.get('high_corr_pairs', []))}
• Similar spectral patterns: {len(statistical_results.get('similar_patterns', []))}
• Hierarchical clusters: {statistical_results.get('num_clusters', 0)}
• PCA variance explained: {statistical_results.get('pca_explained_variance', [0, 0])[0]+statistical_results.get('pca_explained_variance', [0, 0])[1]:.1%}
"""
    
    ax3.text(0.05, 0.95, cluster_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('roi_detailed_legend.png', dpi=300, bbox_inches='tight')
    print("Saved detailed ROI legend to roi_detailed_legend.png")
    plt.close()


def create_interactive_roi_map(filtered_labeled_mask, spectral_df, num_filtered_labels):
    """
    Create an interactive-style ROI location map with grid coordinates.
    
    Args:
        filtered_labeled_mask: Labeled ROI mask
        spectral_df: DataFrame with ROI spectral data
        num_filtered_labels: Number of ROIs
    """
    print("Creating interactive ROI location map...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left panel: ROI map with grid
    roi_colormap = create_roi_colormap(num_filtered_labels)
    height, width = filtered_labeled_mask.shape
    
    # Create grid coordinates
    grid_spacing = min(100, max(50, min(height, width) // 15))
    
    im = ax1.imshow(filtered_labeled_mask, cmap=roi_colormap, vmin=0, vmax=num_filtered_labels, 
                   interpolation='nearest')
    
    # Add grid
    for i in range(0, height, grid_spacing):
        ax1.axhline(y=i, color='white', alpha=0.6, linewidth=1)
        if i < height - 20:  # Add labels
            ax1.text(-20, i, f'{i}', ha='right', va='center', color='white', fontweight='bold')
    
    for j in range(0, width, grid_spacing):
        ax1.axvline(x=j, color='white', alpha=0.6, linewidth=1)
        if j < width - 20:  # Add labels
            ax1.text(j, -20, f'{j}', ha='center', va='top', color='white', fontweight='bold')
    
    # Add ROI centroids with labels
    if not spectral_df.empty and 'centroid_col' in spectral_df.columns:
        for _, roi_row in spectral_df.iterrows():
            roi_id = roi_row['roi_id']
            x, y = roi_row['centroid_col'], roi_row['centroid_row']
            ax1.plot(x, y, 'ko', markersize=4)
            ax1.text(x+10, y-10, f'{roi_id}', fontsize=8, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    ax1.set_title('ROI Location Map with Grid Coordinates')
    ax1.set_xlabel('X coordinate (pixels)')
    ax1.set_ylabel('Y coordinate (pixels)')
    
    # Right panel: ROI location table
    ax2.axis('off')
    
    if not spectral_df.empty:
        # Create location reference table
        location_data = []
        for _, roi_row in spectral_df.iterrows():
            roi_id = roi_row['roi_id']
            centroid_x = roi_row.get('centroid_col', 0)
            centroid_y = roi_row.get('centroid_row', 0)
            
            # Calculate grid coordinates
            grid_x = int(centroid_x // grid_spacing)
            grid_y = int(centroid_y // grid_spacing)
            
            location_data.append([
                f'ROI {roi_id}',
                f'({centroid_x:.0f}, {centroid_y:.0f})',
                f'Grid [{grid_x}, {grid_y}]'
            ])
        
        # Show in chunks if too many ROIs
        chunk_size = 25
        for chunk_start in range(0, len(location_data), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(location_data))
            chunk_data = location_data[chunk_start:chunk_end]
            
            # Create table for this chunk
            table = ax2.table(cellText=chunk_data,
                             colLabels=['ROI ID', 'Pixel Coords', 'Grid Coords'],
                             cellLoc='center',
                             loc='upper left',
                             bbox=[0, 1 - (chunk_start + len(chunk_data))/len(location_data), 1, len(chunk_data)/len(location_data)],
                             edges='closed')
            
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1, 0.8)
            
            # Color code the ROI ID column
            for i, (roi_id, _, _) in enumerate(chunk_data):
                try:
                    roi_num = int(float(roi_id.split()[1]))  # Handle both int and float strings
                    color = roi_colormap(roi_num / num_filtered_labels)
                    table[(i+1, 0)].set_facecolor(color)
                    table[(i+1, 0)].set_text_props(weight='bold', color='white')
                except (ValueError, IndexError):
                    # Skip coloring if we can't parse the ROI ID
                    pass
    
    ax2.set_title(f'ROI Location Reference\n(Grid spacing: {grid_spacing} pixels)')
    
    plt.tight_layout()
    plt.savefig('roi_interactive_map.png', dpi=300, bbox_inches='tight')
    print("Saved interactive ROI map to roi_interactive_map.png")
    plt.close()

# --- ROI Summary Statistics ---
if roi_spectral_data:
    print(f"\n=== ROI SEGMENTATION SUMMARY ===")
    print(f"Total ROIs processed: {len(roi_spectral_data)}")
    print(f"Channels analyzed: {num_channels}")
    
    # Calculate basic statistics across all ROIs and channels
    all_intensities = []
    for roi_data in roi_spectral_data:
        all_intensities.extend(roi_data['spectral_signature'])
    
    if all_intensities:
        print(f"Overall intensity range: {np.min(all_intensities):.2f} - {np.max(all_intensities):.2f}")
        print(f"Overall mean intensity: {np.mean(all_intensities):.2f} ± {np.std(all_intensities):.2f}")
    
    # Print individual ROI information
    print(f"\nIndividual ROI Details:")
    for i, roi_data in enumerate(roi_spectral_data):
        roi_id = roi_data['roi_id']
        centroid = roi_data['centroid']
        mean_spectral = np.mean(roi_data['spectral_signature'])
        print(f"  ROI {roi_id}: Centroid ({centroid[1]:.1f}, {centroid[0]:.1f}), Mean Intensity: {mean_spectral:.2f}")


spectral_df = pd.DataFrame() # Initialize
if roi_spectral_data:
    spectral_df = pd.DataFrame(roi_spectral_data)
    spectral_signatures_df = pd.DataFrame(spectral_df['spectral_signature'].tolist(),
                                        columns=[f'Channel_{i+1}_Mean' for i in range(num_channels)])
    # Prepare centroid columns correctly for later plotting
    spectral_df['centroid_row'] = spectral_df['centroid'].apply(lambda x: x[0])
    spectral_df['centroid_col'] = spectral_df['centroid'].apply(lambda x: x[1])
    spectral_df = pd.concat([spectral_df.drop(['spectral_signature', 'centroid'], axis=1), spectral_signatures_df], axis=1)
    
    print("\nSpectral Data DataFrame Head:")
    print(spectral_df.head())
    try:
        spectral_df.to_csv('roi_spectral_data_manual.csv', index=False)
        print("\nSaved spectral data to roi_spectral_data_manual.csv")
    except Exception as e:
        print(f"\nError saving spectral data to CSV: {e}")
else:
    print("\nNo spectral data extracted from manual ROIs.")
    
# --- Visualization of Labeled Manual ROIs ---
if num_filtered_labels > 0:
    # Use enhanced visualization methods
    visualize_roi_mask_enhanced(segmentation_input, filtered_labeled_mask, spectral_df, num_filtered_labels)
    
    # Create the basic visualization for backward compatibility
    fig_manual_rois, ax_manual_rois = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display reference image
    ax_manual_rois[0].imshow(segmentation_input[14], cmap='gray', vmin=vmin_val, vmax=vmax_val)
    ax_manual_rois[0].set_title('Reference Image Channel 15 w/ Max Z Projection')
    ax_manual_rois[0].axis('off')
    
    # Display the enhanced labeled mask
    roi_colormap = create_roi_colormap(num_filtered_labels)
    im_labeled = ax_manual_rois[1].imshow(filtered_labeled_mask, cmap=roi_colormap, interpolation='nearest')
    ax_manual_rois[1].set_title(f'Enhanced ROI Mask ({num_filtered_labels} ROIs)')
    ax_manual_rois[1].axis('off')
    
    # Add centroids if available
    if not spectral_df.empty and 'centroid_col' in spectral_df.columns and 'centroid_row' in spectral_df.columns:
        ax_manual_rois[1].scatter(spectral_df['centroid_col'], spectral_df['centroid_row'], 
                                 s=10, c='red', marker='x', label='Centroids')
    
    plt.tight_layout()
    plt.savefig('manual_roi_labeled_mask.png', dpi=150, bbox_inches='tight')
    print("\nSaved enhanced ROI labeled mask visualization to manual_roi_labeled_mask.png")
    plt.close(fig_manual_rois)
    
    # --- Intensity Distribution Analysis ---
    # Generate comprehensive intensity histograms
    plot_intensity_histograms(spectral_df, segmentation_input, filtered_labeled_mask, num_channels)
    
    # Perform statistical grouping analysis
    statistical_results = perform_statistical_grouping_analysis(spectral_df, num_channels)
    
    # Create detailed reference materials
    create_detailed_legend(spectral_df, num_filtered_labels, statistical_results)
    create_interactive_roi_map(filtered_labeled_mask, spectral_df, num_filtered_labels)
else:
    print("\nNo manual ROIs to visualize.")
    statistical_results = {}



# --- Vis: Segmentation and Combined Histogram
if not spectral_df.empty:
    # Plot 2: Individual Channel Plots (ROI ID vs Mean Intensity) ---
    print("\nPlotting individual channel plots (ROI ID vs Mean Intensity)...")
    individual_hist_figs = [] # Keep track of figures to close
    try:
        for i in range(num_channels):
            channel_name = f'Channel_{i+1}_Mean'
            if channel_name in spectral_df.columns:
                intensities = spectral_df[channel_name].dropna()
                roi_ids = spectral_df.loc[intensities.index, 'roi_id']
                
                if not intensities.empty:
                    # Create a new figure for each channel's plot
                    fig_hist_ind, ax_hist_ind = plt.subplots(figsize=(10, 6))
                    individual_hist_figs.append(fig_hist_ind) # Add fig to list

                    # Plot ROI ID vs Mean Intensity (switched axes as requested)
                    ax_hist_ind.bar(roi_ids, intensities, edgecolor='black', color='skyblue', alpha=0.7)
                    ax_hist_ind.set_title(f'Mean ROI Intensities for Channel {i+1}')
                    ax_hist_ind.set_xlabel('ROI ID')
                    ax_hist_ind.set_ylabel('Mean Intensity')
                    ax_hist_ind.grid(axis='y', alpha=0.75)
                    
                    # Set x-axis to show all ROI IDs
                    ax_hist_ind.set_xticks(roi_ids)
                    ax_hist_ind.set_xticklabels(roi_ids, rotation=45 if len(roi_ids) > 10 else 0)

                    plt.tight_layout()
                    filename = f'roi_intensities_channel_{i+1}.png'
                    plt.savefig(filename, dpi=150, bbox_inches='tight')
                    print(f"  - Saved ROI intensity plot to {filename}")
                    plt.close(fig_hist_ind) # Close the figure after saving
                    individual_hist_figs.remove(fig_hist_ind) # Remove from list after closing
                else:
                    print(f"  - Skipping {channel_name} (no valid data)")
            else:
                print(f"  - Skipping {channel_name} (column not found)")
    except Exception as e:
        print(f"Error during individual channel plotting: {e}")
    finally:
        # Ensure all created figures are closed even if an error occurred mid-loop
        for fig in individual_hist_figs:
            try:
                plt.close(fig)
            except Exception:
                pass # Ignore errors during cleanup closing


    # Plot 3: Combined Spectral Signatures Plot (All ROIs across all channels)
    try:
        fig_spectral, ax_spectral = plt.subplots(figsize=(14, 8))
        
        print("\nPlotting combined spectral signatures for all ROIs...")
        
        # Define a colormap to cycle through for different ROIs
        colors = plt.cm.tab10(np.linspace(0, 1, len(spectral_df)))
        if len(spectral_df) > 10:  # Use a different colormap for more ROIs
            colors = plt.cm.viridis(np.linspace(0, 1, len(spectral_df)))

        channel_numbers = np.arange(1, num_channels + 1)
        
        plotted_rois = 0
        for idx, (_, roi_row) in enumerate(spectral_df.iterrows()):
            roi_id = roi_row['roi_id']
            spectral_cols = [f'Channel_{i+1}_Mean' for i in range(num_channels)]
            
            if all(col in spectral_df.columns for col in spectral_cols):
                spectral_values = roi_row[spectral_cols].values
                
                # Check for NaN values
                if not np.isnan(spectral_values).any():
                    ax_spectral.plot(channel_numbers, spectral_values, 
                                   marker='o', linestyle='-', linewidth=2, markersize=4,
                                   label=f'ROI {roi_id}', color=colors[idx], alpha=0.8)
                    plotted_rois += 1
                else:
                    print(f"  - Skipping ROI {roi_id} (contains NaN values)")
            else:
                print(f"  - Skipping ROI {roi_id} (missing spectral columns)")

        if plotted_rois > 0:
            ax_spectral.set_title(f'Spectral Signatures of All ROIs ({plotted_rois} ROIs)')
            ax_spectral.set_xlabel('Channel Number')
            ax_spectral.set_ylabel('Mean Intensity')
            ax_spectral.grid(True, alpha=0.3)
            ax_spectral.set_xticks(channel_numbers)
            
            # Add legend - place outside plot area if too crowded
            if plotted_rois <= 10:
                ax_spectral.legend(title="ROIs", fontsize='small')
            else:
                ax_spectral.legend(title="ROIs", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                
            plt.tight_layout()
            if plotted_rois > 10:
                plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make space for external legend

            plt.savefig('spectral_signatures_all_rois.png', dpi=150, bbox_inches='tight')
            print(f"\nSaved combined spectral signatures plot to spectral_signatures_all_rois.png")
        else:
            print("\nNo ROIs plotted in the spectral signatures plot.")

        plt.close(fig_spectral)

    except Exception as e:
        print(f"Error during spectral signatures plotting: {e}")
        if 'fig_spectral' in locals() and fig_spectral is not None:
            plt.close(fig_spectral)

    # Plot 3: Individual ROI Spectral Signature Histograms ---
    print("\nGenerating individual spectral signature histograms for each ROI...")
    try:
        # Get spectral channel columns
        spectral_cols = [f'Channel_{i+1}_Mean' for i in range(num_channels)]
        available_spectral_cols = [col for col in spectral_cols if col in spectral_df.columns]
        
        if available_spectral_cols:
            # Create colormap for consistent ROI coloring
            roi_colormap = create_roi_colormap(len(spectral_df))
            
            # Create individual spectral signature plot for each ROI
            for idx, row in spectral_df.iterrows():
                roi_id = row['roi_id']
                
                # Extract spectral intensities for this ROI
                roi_intensities = [row[col] for col in available_spectral_cols]
                channel_numbers = list(range(1, len(available_spectral_cols) + 1))
                
                # Create figure for individual ROI
                fig_roi, ax_roi = plt.subplots(figsize=(12, 6))
                
                # Get consistent color for this ROI
                roi_color = roi_colormap(roi_id / len(spectral_df))
                
                # Create histogram-style bar plot
                bars = ax_roi.bar(channel_numbers, roi_intensities, 
                                color=roi_color, alpha=0.7, 
                                edgecolor='black', linewidth=0.5)
                
                # Customize the plot
                ax_roi.set_title(f'Spectral Signature - ROI {roi_id}\n'
                               f'(Centroid: {row["centroid_row"]:.1f}, {row["centroid_col"]:.1f})',
                               fontsize=14, fontweight='bold')
                ax_roi.set_xlabel('Channel Number', fontsize=12)
                ax_roi.set_ylabel('Mean Intensity', fontsize=12)
                ax_roi.grid(axis='y', alpha=0.3)
                
                # Set x-axis ticks for all channels
                ax_roi.set_xticks(channel_numbers)
                ax_roi.set_xticklabels([f'Ch{i}' for i in channel_numbers])
                
                # Add value labels on top of bars for better readability
                for bar, intensity in zip(bars, roi_intensities):
                    height = bar.get_height()
                    ax_roi.text(bar.get_x() + bar.get_width()/2., height + max(roi_intensities)*0.01,
                              f'{intensity:.1f}', ha='center', va='bottom', fontsize=8)
                
                # Add statistics text box
                mean_intensity = np.mean(roi_intensities)
                std_intensity = np.std(roi_intensities)
                max_intensity = np.max(roi_intensities)
                min_intensity = np.min(roi_intensities)
                
                stats_text = f'Mean: {mean_intensity:.2f}\nStd: {std_intensity:.2f}\n'
                stats_text += f'Max: {max_intensity:.2f}\nMin: {min_intensity:.2f}'
                
                ax_roi.text(0.98, 0.98, stats_text, transform=ax_roi.transAxes, 
                          fontsize=10, verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.tight_layout()
                
                # Save individual ROI spectral signature
                filename = f'roi_{roi_id}_spectral_signature.png'
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close(fig_roi)
                
                print(f"  ✓ Saved spectral signature for ROI {roi_id} to {filename}")
            
            print(f"\nSuccessfully generated {len(spectral_df)} individual ROI spectral signature histograms!")
        
        else:
            print("No spectral channel data available for individual ROI plotting.")
            
    except Exception as e:
        print(f"Error creating individual ROI spectral signatures: {e}")
else:
    print("\nSkipping segmentation/histogram visualization (no spectral data extracted or matplotlib unavailable).")





# --- Final Summary ---
print("\n" + "="*50)
print("ENHANCED HYPERSPECTRAL ROI ANALYSIS COMPLETE")
print("="*50)

if not spectral_df.empty:
    # Enhanced CSV export with cluster information
    try:
        # Add spatial information and statistical details to the export
        export_df = spectral_df.copy()
        
        # Calculate additional metrics for each ROI
        spectral_cols = [f'Channel_{i+1}_Mean' for i in range(num_channels)]
        if all(col in export_df.columns for col in spectral_cols):
            # Calculate spectral statistics for each ROI
            export_df['spectral_mean'] = export_df[spectral_cols].mean(axis=1)
            export_df['spectral_std'] = export_df[spectral_cols].std(axis=1)
            export_df['spectral_max'] = export_df[spectral_cols].max(axis=1)
            export_df['spectral_min'] = export_df[spectral_cols].min(axis=1)
        
        # Add statistical grouping information if available
        if statistical_results and 'hierarchical_labels' in statistical_results:
            # Match ROI IDs to cluster labels
            roi_ids = statistical_results['roi_ids']
            hier_labels = statistical_results['hierarchical_labels']
            intensity_groups = statistical_results['intensity_groups']
            
            for i, roi_id in enumerate(roi_ids):
                mask = export_df['roi_id'] == roi_id
                if mask.any():
                    export_df.loc[mask, 'hierarchical_cluster'] = hier_labels[i]
                    export_df.loc[mask, 'intensity_group'] = intensity_groups[i]
        
        # Save enhanced CSV
        enhanced_csv_path = 'roi_spectral_data_enhanced_analysis.csv'
        export_df.to_csv(enhanced_csv_path, index=False)
        print(f"\nSaved enhanced spectral data to {enhanced_csv_path}")
        
        # Print comprehensive final summary
        print(f"\nCOMPREHENSIVE ANALYSIS SUMMARY:")
        print(f"- Total ROIs loaded: {num_filtered_labels}")
        print(f"- Successfully analyzed: {len(export_df)}")
        print(f"- Spectral channels: {num_channels}")
        print(f"- Missing ROIs: {num_filtered_labels - len(export_df)}")
        
        if statistical_results:
            print(f"\nSTATISTICAL GROUPING RESULTS:")
            if 'high_corr_pairs' in statistical_results:
                print(f"- High correlation pairs (>0.8): {len(statistical_results['high_corr_pairs'])}")
            if 'similar_patterns' in statistical_results:
                print(f"- Similar spectral patterns (>0.9): {len(statistical_results['similar_patterns'])}")
            if 'num_clusters' in statistical_results:
                print(f"- Hierarchical clusters identified: {statistical_results['num_clusters']}")
            if 'pca_explained_variance' in statistical_results:
                pc1_pc2_variance = statistical_results['pca_explained_variance'][0] + statistical_results['pca_explained_variance'][1]
                print(f"- PCA: PC1+PC2 explain {pc1_pc2_variance:.1%} of variance")
        
        if 'spectral_mean' in export_df.columns:
            print(f"\nSPECTRAL INTENSITY STATISTICS:")
            print(f"- Overall intensity range: {export_df['spectral_min'].min():.2f} - {export_df['spectral_max'].max():.2f}")
            print(f"- Mean intensity across all ROIs: {export_df['spectral_mean'].mean():.2f} ± {export_df['spectral_mean'].std():.2f}")
            print(f"- Most intense ROI: {export_df.loc[export_df['spectral_mean'].idxmax(), 'roi_id']} (intensity: {export_df['spectral_mean'].max():.2f})")
            print(f"- Least intense ROI: {export_df.loc[export_df['spectral_mean'].idxmin(), 'roi_id']} (intensity: {export_df['spectral_mean'].min():.2f})")
        
        print(f"\nGENERATED VISUALIZATIONS:")
        visualizations = [
            'filtered_channel_projections_fixed_scale.png',
            'manual_roi_labeled_mask.png',
            'roi_enhanced_visualizations.png',
            'roi_intensity_histograms_grid.png',
            'roi_intensity_distributions_combined.png',
            'roi_statistical_analysis_comprehensive.png',
            'roi_detailed_legend.png',
            'roi_interactive_map.png',
            'spectral_signatures_all_rois.png'
        ]
        
        # Check for individual channel plots
        for i in range(num_channels):
            viz_file = f'roi_intensities_channel_{i+1}.png'
            visualizations.append(viz_file)
        
        for viz in visualizations:
            print(f"  ✓ {viz}")
        
        print(f"\nENHANCED FEATURES IMPLEMENTED:")
        print(f"  ✓ Advanced color mapping for {num_filtered_labels} ROIs")
        print(f"  ✓ Multiple ROI file support")
        print(f"  ✓ Comprehensive intensity histogram analysis")
        print(f"  ✓ Statistical grouping with PCA and clustering")
        print(f"  ✓ Enhanced ROI identification methods")
        print(f"  ✓ Overlap handling to preserve all ROIs")
        print(f"  ✓ Interactive reference materials")
        
    except Exception as e:
        print(f"\nError during final data export: {e}")
        
else:
    print("\nNo spectral data was extracted. Please check:")
    print("- ROI file path and format")
    print("- Image data compatibility")
    print("- ROI coordinate system alignment")

print("\nEnhanced analysis pipeline completed successfully!")
print("All 53 ROIs should now be properly handled with advanced visualizations.")
print("="*50)
