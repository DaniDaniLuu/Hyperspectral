import glob
import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from bioio import BioImage
from scipy.ndimage import gaussian_filter
from skimage.draw import polygon
from skimage.measure import regionprops, find_contours
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from roifile import roiread

# Constants
CZI_PATH = "241121_10h40min_bra.h2b.mAp_twist.RFP_crbn.h2b.GFP_meis.kaede.czi"
FIJI_ROI_ZIP_PATH = "Rois.zip"
APPLY_SMOOTHING = False


def create_roi_colormap(num_rois):
    """
    Create a consistent, distinguishable colormap for ROI visualization.
    
    Args:
        num_rois (int): Number of ROIs to create colors for
        
    Returns:
        list: List of hex color strings
    """
    # Define a set of easily distinguishable colors
    base_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange  
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf',  # Cyan
        '#aec7e8',  # Light Blue
        '#ffbb78'   # Light Orange
    ]
    
    # If we need more colors than available, cycle through them
    colors = []
    for i in range(num_rois):
        colors.append(base_colors[i % len(base_colors)])
    
    return colors


def get_roi_color(roi_id, roi_colors):
    """
    Get color for a specific ROI ID (1-indexed).
    
    Args:
        roi_id (int): ROI identifier (1-indexed)
        roi_colors (list): List of color strings
        
    Returns:
        str: Hex color string
    """
    # Convert to int to handle numpy types from pandas
    roi_id = int(roi_id)
    return roi_colors[roi_id - 1] if roi_id <= len(roi_colors) else '#000000'


def load_image_data(czi_path):
    """
    Load and preprocess hyperspectral image data.
    
    Args:
        czi_path (str): Path to the CZI image file
        
    Returns:
        tuple: (img_data, max_proj, segmentation_input)
    """
    print(f"Loading image data from: {czi_path}")
    img = BioImage(czi_path)
    img_data = img.data.squeeze()  # Remove time dimension
    
    print(f"Image metadata: {img_data.shape}")  # (15, 31, 1069, 1069)
    
    # Do max projection along Z axis
    max_proj = np.max(img_data, axis=1)
    
    # Apply optional smoothing
    if APPLY_SMOOTHING:
        try:
            smoothed_proj = gaussian_filter(max_proj, sigma=(0, 1, 1))
            print(f"Applied Gaussian smoothing. Shape: {smoothed_proj.shape}")
            segmentation_input = smoothed_proj
        except Exception as e:
            print(f"Error applying Gaussian filter: {e}")
            print("Using original max projection without smoothing.")
            segmentation_input = max_proj
    else:
        print("Skipping Gaussian smoothing.")
        segmentation_input = max_proj
    
    print(f"Segmentation input shape: {segmentation_input.shape}")
    return img_data, max_proj, segmentation_input


def visualize_channels(segmentation_input, output_filename='filtered_channel_projections_fixed_scale.png'):
    """
    Visualize all channels with consistent scaling.
    
    Args:
        segmentation_input (np.ndarray): Channel data to visualize
        output_filename (str): Output filename for the visualization
    """
    vmin_val = np.min(segmentation_input)
    vmax_val = np.max(segmentation_input)
    print(f"\nUsing fixed intensity scale: vmin={vmin_val}, vmax={vmax_val}")
    
    num_channels = segmentation_input.shape[0]
    ncols = 5
    nrows = (num_channels + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3), squeeze=False)
    axes = axes.ravel()
    
    for i in range(num_channels):
        ax = axes[i]
        im = ax.imshow(segmentation_input[i], cmap='gray', vmin=vmin_val, vmax=vmax_val)
        ax.set_title(f'Channel {i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for j in range(num_channels, nrows * ncols):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved channel projections to {output_filename}")
    plt.close(fig)


def load_fiji_rois(fiji_roi_zip_path):
    """
    Load manual ROIs from FIJI zip file.
    
    Args:
        fiji_roi_zip_path (str): Path to the ROI zip file
        
    Returns:
        list: List of ROI objects
    """
    print(f"\nLoading manual ROIs from: {fiji_roi_zip_path}")
    fiji_rois = []
    
    try:
        # Try to read the zip file directly
        try:
            fiji_rois = roiread(fiji_roi_zip_path)
            print(f"Successfully read {len(fiji_rois)} ROIs from zip file")
        except Exception as e:
            print(f"Could not read zip file directly: {e}")
            
            # Check if we have an extracted directory
            roi_dir = "manual_rois"
            if os.path.exists(roi_dir):
                roi_files = glob.glob(os.path.join(roi_dir, "*.roi"))
                print(f"Found {len(roi_files)} ROI files in {roi_dir}")
                
                for roi_file in roi_files:
                    try:
                        roi = roiread(roi_file)
                        if isinstance(roi, list):
                            fiji_rois.extend(roi)
                        else:
                            fiji_rois.append(roi)
                        print(f"  Loaded ROI from {os.path.basename(roi_file)}")
                    except Exception as roi_e:
                        print(f"  Failed to load ROI from {os.path.basename(roi_file)}: {roi_e}")
            else:
                # Extract the zip file first
                with zipfile.ZipFile(fiji_roi_zip_path, 'r') as zip_ref:
                    zip_ref.extractall('.')
                
                # Now try reading the individual files
                roi_files = glob.glob(os.path.join(roi_dir, "*.roi"))
                for roi_file in roi_files:
                    try:
                        roi = roiread(roi_file)
                        if isinstance(roi, list):
                            fiji_rois.extend(roi)
                        else:
                            fiji_rois.append(roi)
                        print(f"  Loaded ROI from {os.path.basename(roi_file)}")
                    except Exception as roi_e:
                        print(f"  Failed to load ROI from {os.path.basename(roi_file)}: {roi_e}")
    
    except FileNotFoundError:
        print(f"ERROR: ROI file not found at {fiji_roi_zip_path}")
        raise
    except Exception as e:
        print(f"Error reading ROI files: {e}")
        raise
    
    if not fiji_rois:
        raise ValueError("No ROIs found in the provided file.")
    
    return fiji_rois


def create_roi_mask(fiji_rois, image_shape):
    """
    Create a labeled mask from FIJI ROIs.
    
    Args:
        fiji_rois (list): List of ROI objects
        image_shape (tuple): Shape of the image (height, width)
        
    Returns:
        tuple: (labeled_mask, num_labels)
    """
    image_height, image_width = image_shape
    labeled_mask = np.zeros((image_height, image_width), dtype=np.int32)
    num_labels = 0
    
    print(f"Processing {len(fiji_rois)} ROIs from FIJI...")
    
    for i, roi in enumerate(fiji_rois):
        roi_id = i + 1
        num_labels += 1
        
        # Debug: Print ROI attributes for first ROI only
        if i == 0:
            print(f"ROI attributes: {dir(roi)}")
            print(f"ROI type: {type(roi)}")
        
        # Extract coordinates
        try:
            x_coords, y_coords = extract_roi_coordinates(roi, roi_id)
        except Exception as coord_e:
            print(f"Error extracting coordinates from ROI {roi_id}: {coord_e}")
            num_labels -= 1
            continue
        
        # Draw ROI on mask
        try:
            if len(x_coords) > 2 and len(y_coords) > 2:  # Polygon
                x_coords = np.clip(x_coords, 0, image_width - 1)
                y_coords = np.clip(y_coords, 0, image_height - 1)
                
                rr, cc = polygon(y_coords, x_coords, shape=labeled_mask.shape)
                labeled_mask[rr, cc] = roi_id
                print(f"  Added polygon ROI {roi_id} with {len(x_coords)} points")
                
            elif len(x_coords) == 2 and len(y_coords) == 2:  # Rectangle
                x1, x2 = int(min(x_coords)), int(max(x_coords))
                y1, y2 = int(min(y_coords)), int(max(y_coords))
                
                x1, x2 = max(0, x1), min(image_width - 1, x2)
                y1, y2 = max(0, y1), min(image_height - 1, y2)
                
                labeled_mask[y1:y2+1, x1:x2+1] = roi_id
                print(f"  Added rectangular ROI {roi_id} at ({x1},{y1}) to ({x2},{y2})")
            else:
                print(f"Warning: ROI {roi_id} has unexpected coordinates. Skipping.")
                num_labels -= 1
                continue
                
        except Exception as draw_e:
            print(f"Error drawing ROI {roi_id}: {draw_e}")
            num_labels -= 1
            continue
        
        # Get ROI name
        roi_name = get_roi_name(roi, roi_id)
        roi_type_name = get_roi_type(roi)
        print(f"  Added ROI: {roi_name} as ID {roi_id} (type: {roi_type_name})")
    
    if num_labels > 0:
        print(f"\nSuccessfully created labeled mask with {num_labels} ROIs")
    else:
        raise ValueError("Could not create labeled mask from FIJI ROIs")
    
    return labeled_mask, num_labels


def extract_roi_coordinates(roi, roi_id):
    """
    Extract coordinates from an ROI object.
    
    Args:
        roi: ROI object from roifile
        roi_id (int): ROI identifier for error reporting
        
    Returns:
        tuple: (x_coords, y_coords) as numpy arrays
    """
    # Try different possible coordinate attribute names
    if hasattr(roi, 'coordinates') and roi.coordinates is not None:
        coords = roi.coordinates()
        if coords is not None and len(coords) > 0:
            coords = np.array(coords)
            if coords.ndim > 1:
                x_coords = coords[:, 0]
                y_coords = coords[:, 1]
            else:
                x_coords = coords[0]
                y_coords = coords[1]
        else:
            # Fallback to x1,y1,x2,y2 attributes
            x_coords = ([roi.left, roi.right] if hasattr(roi, 'left') and hasattr(roi, 'right') 
                       else [roi.x1, roi.x2])
            y_coords = ([roi.top, roi.bottom] if hasattr(roi, 'top') and hasattr(roi, 'bottom') 
                       else [roi.y1, roi.y2])
    elif hasattr(roi, 'x') and hasattr(roi, 'y'):
        x_coords = roi.x
        y_coords = roi.y
    elif hasattr(roi, 'x1') and hasattr(roi, 'y1'):
        x_coords = [roi.x1, roi.x2] if hasattr(roi, 'x2') else [roi.x1]
        y_coords = [roi.y1, roi.y2] if hasattr(roi, 'y2') else [roi.y1]
    elif hasattr(roi, 'left') and hasattr(roi, 'top'):
        x_coords = [roi.left, roi.right]
        y_coords = [roi.top, roi.bottom]
    else:
        available_attrs = [attr for attr in dir(roi) if not attr.startswith('_')]
        raise ValueError(f"Could not find coordinates for ROI {roi_id}. "
                        f"Available attributes: {available_attrs}")
    
    # Convert to numpy arrays
    x_coords = np.array(x_coords) if not isinstance(x_coords, np.ndarray) else x_coords
    y_coords = np.array(y_coords) if not isinstance(y_coords, np.ndarray) else y_coords
    
    return x_coords, y_coords


def get_roi_name(roi, roi_id):
    """Get ROI name from ROI object."""
    if hasattr(roi, 'name') and roi.name:
        return roi.name
    elif hasattr(roi, 'label') and roi.label:
        return roi.label
    else:
        return f"ROI_{roi_id}"


def get_roi_type(roi):
    """Get ROI type from ROI object."""
    if hasattr(roi, 'roitype'):
        return getattr(roi, 'roitype_name', str(roi.roitype))
    elif hasattr(roi, 'type'):
        return str(roi.type)
    else:
        return "unknown"


def extract_spectral_data(labeled_mask, segmentation_input, num_labels):
    """
    Extract spectral information for each ROI.
    
    Args:
        labeled_mask (np.ndarray): Labeled ROI mask
        segmentation_input (np.ndarray): Hyperspectral image data
        num_labels (int): Number of ROI labels
        
    Returns:
        list: List of dictionaries containing ROI spectral data
    """
    print("\nExtracting spectral data from ROIs...")
    num_channels = segmentation_input.shape[0]
    roi_spectral_data = []
    
    # Get region properties
    props = regionprops(labeled_mask, intensity_image=segmentation_input[0])
    
    if not props:
        raise ValueError("regionprops did not return any regions")
    
    print(f"regionprops found {len(props)} regions.")
    
    # Create mapping from label ID to regionprop object
    prop_dict = {prop.label: prop for prop in props}
    
    for roi_label_id in range(1, num_labels + 1):
        if roi_label_id not in prop_dict:
            print(f"Warning: ROI {roi_label_id} not found in regionprops. Skipping.")
            continue
        
        region = prop_dict[roi_label_id]
        roi_coords = region.coords  # shape (N, 2)
        
        # Extract spectral signature
        spectral_signature = []
        for c in range(num_channels):
            channel_data = segmentation_input[c]
            roi_pixels = channel_data[roi_coords[:, 0], roi_coords[:, 1]]
            mean_intensity = np.mean(roi_pixels)
            spectral_signature.append(mean_intensity)
        
        roi_spectral_data.append({
            'roi_id': region.label,
            'centroid': region.centroid,  # (row, col)
            'spectral_signature': np.array(spectral_signature)
        })
    
    print(f"Extracted spectral data for {len(roi_spectral_data)} ROIs.")
    return roi_spectral_data


def create_spectral_dataframe(roi_spectral_data, num_channels):
    """
    Create a pandas DataFrame from spectral data.
    
    Args:
        roi_spectral_data (list): List of ROI spectral data dictionaries
        num_channels (int): Number of spectral channels
        
    Returns:
        pd.DataFrame: Spectral data DataFrame
    """
    if not roi_spectral_data:
        return pd.DataFrame()
    
    spectral_df = pd.DataFrame(roi_spectral_data)
    
    # Expand spectral signatures into separate columns
    spectral_signatures_df = pd.DataFrame(
        spectral_df['spectral_signature'].tolist(),
        columns=[f'Channel_{i+1}_Mean' for i in range(num_channels)]
    )
    
    # Add centroid coordinates
    spectral_df['centroid_row'] = spectral_df['centroid'].apply(lambda x: x[0])
    spectral_df['centroid_col'] = spectral_df['centroid'].apply(lambda x: x[1])
    
    # Combine DataFrames
    spectral_df = pd.concat([
        spectral_df.drop(['spectral_signature', 'centroid'], axis=1),
        spectral_signatures_df
    ], axis=1)
    
    return spectral_df


def print_roi_summary(roi_spectral_data, num_channels):
    """Print summary statistics for ROI analysis."""
    if not roi_spectral_data:
        print("\nNo spectral data extracted from manual ROIs.")
        return
    
    print(f"\n=== ROI SEGMENTATION SUMMARY ===")
    print(f"Total ROIs processed: {len(roi_spectral_data)}")
    print(f"Channels analyzed: {num_channels}")
    
    # Calculate statistics
    all_intensities = []
    for roi_data in roi_spectral_data:
        all_intensities.extend(roi_data['spectral_signature'])
    
    if all_intensities:
        print(f"Intensity range: {np.min(all_intensities):.2f} - {np.max(all_intensities):.2f}")
        print(f"Mean intensity: {np.mean(all_intensities):.2f} ± {np.std(all_intensities):.2f}")
    
    # Print individual ROI information
    print(f"\nIndividual ROI Details:")
    for roi_data in roi_spectral_data:
        roi_id = roi_data['roi_id']
        centroid = roi_data['centroid']
        mean_spectral = np.mean(roi_data['spectral_signature'])
        print(f"  ROI {roi_id}: Centroid ({centroid[1]:.1f}, {centroid[0]:.1f}), "
              f"Mean Intensity: {mean_spectral:.2f}")


def visualize_roi_mask(labeled_mask, segmentation_input, spectral_df, num_labels, 
                      vmin_val, vmax_val):
    """
    Visualize the ROI labeled mask with consistent color mapping.
    
    Args:
        labeled_mask (np.ndarray): Labeled ROI mask
        segmentation_input (np.ndarray): Image data for reference
        spectral_df (pd.DataFrame): Spectral data DataFrame
        num_labels (int): Number of ROI labels
        vmin_val (float): Minimum intensity value for scaling
        vmax_val (float): Maximum intensity value for scaling
    """
    if num_labels == 0:
        print("\nNo manual ROIs to visualize.")
        return
    
    # Create figure with space for legend
    fig = plt.figure(figsize=(16, 6))
    
    # Create subplot layout
    ax_ref = plt.subplot(1, 3, 1)
    ax_labeled = plt.subplot(1, 3, 2)
    ax_legend = plt.subplot(1, 3, 3)
    
    # Reference image
    ax_ref.imshow(segmentation_input[10], cmap='gray', vmin=vmin_val, vmax=vmax_val)
    ax_ref.set_title('Reference Image Channel 11 w/ Max Z Projection')
    ax_ref.axis('off')
    
    # Create consistent colormap
    roi_colors = create_roi_colormap(num_labels)
    roi_colormap_colors = ['#000000'] + roi_colors  # Black background + ROI colors
    custom_cmap = ListedColormap(roi_colormap_colors)
    
    # Labeled mask
    im_labeled = ax_labeled.imshow(labeled_mask, cmap=custom_cmap, 
                                  interpolation='nearest', vmin=0, vmax=num_labels)
    ax_labeled.set_title(f'Manually Labeled ROIs ({num_labels})')
    ax_labeled.axis('off')
    
    # Add centroids if available
    if not spectral_df.empty and 'centroid_col' in spectral_df.columns:
        ax_labeled.scatter(spectral_df['centroid_col'], spectral_df['centroid_row'], 
                          s=15, c='white', marker='x', linewidth=2)
        ax_labeled.scatter(spectral_df['centroid_col'], spectral_df['centroid_row'], 
                          s=10, c='black', marker='x', linewidth=1)
    
    # Create legend
    create_roi_legend(ax_legend, roi_colors, num_labels, spectral_df)
    
    plt.tight_layout()
    plt.savefig('manual_roi_labeled_mask.png', dpi=150, bbox_inches='tight')
    print("\nSaved ROI labeled mask to manual_roi_labeled_mask.png")
    plt.close(fig)


def create_roi_legend(ax_legend, roi_colors, num_labels, spectral_df):
    """Create color legend for ROI visualization."""
    ax_legend.set_title('ROI Color Index', fontweight='bold', fontsize=12)
    ax_legend.axis('off')
    
    # Calculate layout
    n_cols = 2 if num_labels > 6 else 1
    
    for roi_id in range(1, num_labels + 1):
        color = get_roi_color(roi_id, roi_colors)
        
        # Position calculation
        col = (roi_id - 1) % n_cols
        row = (roi_id - 1) // n_cols
        
        x_pos = col * 0.5
        y_pos = 0.9 - (row * 0.15)
        
        # Add colored rectangle
        rect = Rectangle((x_pos, y_pos), 0.1, 0.08, facecolor=color, 
                        edgecolor='black', linewidth=0.5)
        ax_legend.add_patch(rect)
        
        # Add ROI text
        ax_legend.text(x_pos + 0.15, y_pos + 0.04, f'ROI {roi_id}', 
                      verticalalignment='center', fontsize=10, fontweight='bold')
        
        # Add centroid coordinates if available
        if not spectral_df.empty:
            roi_row = spectral_df[spectral_df['roi_id'] == roi_id]
            if not roi_row.empty:
                centroid_x = roi_row['centroid_col'].iloc[0]
                centroid_y = roi_row['centroid_row'].iloc[0]
                ax_legend.text(x_pos + 0.15, y_pos + 0.01, 
                              f'({centroid_x:.0f}, {centroid_y:.0f})', 
                              verticalalignment='center', fontsize=8, color='gray')
    
    # Set axis limits
    ax_legend.set_xlim(0, n_cols * 0.5)
    ax_legend.set_ylim(0, 1)


def plot_individual_channels(spectral_df, num_channels):
    """
    Plot individual channel intensity histograms.
    
    Args:
        spectral_df (pd.DataFrame): Spectral data DataFrame
        num_channels (int): Number of spectral channels
    """
    if spectral_df.empty:
        return
    
    print("\nPlotting individual channel plots (ROI ID vs Mean Intensity)...")
    
    for i in range(num_channels):
        channel_name = f'Channel_{i+1}_Mean'
        
        if channel_name not in spectral_df.columns:
            print(f"  - Skipping {channel_name} (column not found)")
            continue
        
        intensities = spectral_df[channel_name].dropna()
        roi_ids = spectral_df.loc[intensities.index, 'roi_id']
        
        if intensities.empty:
            print(f"  - Skipping {channel_name} (no valid data)")
            continue
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.bar(roi_ids, intensities, edgecolor='black', color='skyblue', alpha=0.7)
        ax.set_title(f'Mean ROI Intensities for Channel {i+1}')
        ax.set_xlabel('ROI ID')
        ax.set_ylabel('Mean Intensity')
        ax.grid(axis='y', alpha=0.75)
        
        # Set x-axis ticks
        ax.set_xticks(roi_ids)
        ax.set_xticklabels(roi_ids, rotation=45 if len(roi_ids) > 10 else 0)
        
        plt.tight_layout()
        filename = f'roi_intensities_channel_{i+1}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  - Saved {filename}")
        plt.close(fig)


def plot_spectral_signatures(spectral_df, num_channels, num_labels):
    """
    Plot combined spectral signatures for all ROIs.
    
    Args:
        spectral_df (pd.DataFrame): Spectral data DataFrame
        num_channels (int): Number of spectral channels
        num_labels (int): Number of ROI labels
    """
    if spectral_df.empty:
        return
    
    print("\nPlotting combined spectral signatures for all ROIs...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Use consistent color mapping
    roi_colors = create_roi_colormap(num_labels)
    channel_numbers = np.arange(1, num_channels + 1)
    
    plotted_rois = 0
    spectral_cols = [f'Channel_{i+1}_Mean' for i in range(num_channels)]
    
    for _, roi_row in spectral_df.iterrows():
        roi_id = roi_row['roi_id']
        
        if not all(col in spectral_df.columns for col in spectral_cols):
            print(f"  - Skipping ROI {roi_id} (missing spectral columns)")
            continue
        
        spectral_values = roi_row[spectral_cols].values
        
        if np.isnan(spectral_values).any():
            print(f"  - Skipping ROI {roi_id} (contains NaN values)")
            continue
        
        # Use consistent color for this ROI
        roi_color = get_roi_color(roi_id, roi_colors)
        ax.plot(channel_numbers, spectral_values, 
               marker='o', linestyle='-', linewidth=2, markersize=4,
               label=f'ROI {roi_id}', color=roi_color, alpha=0.8)
        plotted_rois += 1
    
    if plotted_rois > 0:
        ax.set_title(f'Spectral Signatures of All ROIs ({plotted_rois} ROIs)')
        ax.set_xlabel('Channel Number')
        ax.set_ylabel('Mean Intensity')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(channel_numbers)
        
        # Add legend
        if plotted_rois <= 10:
            ax.legend(title="ROIs", fontsize='small')
        else:
            ax.legend(title="ROIs", bbox_to_anchor=(1.05, 1), loc='upper left', 
                     fontsize='small')
            plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        plt.tight_layout()
        plt.savefig('spectral_signatures_all_rois.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved spectral signatures plot to spectral_signatures_all_rois.png")
    else:
        print("\nNo ROIs plotted in spectral signatures.")
    
    plt.close(fig)


def perform_clustering(spectral_df, num_channels, labeled_mask, segmentation_input):
    """
    Perform K-means clustering on spectral data.
    
    Args:
        spectral_df (pd.DataFrame): Spectral data DataFrame
        num_channels (int): Number of spectral channels
        labeled_mask (np.ndarray): Labeled ROI mask
        segmentation_input (np.ndarray): Original image data
        
    Returns:
        pd.DataFrame: Updated spectral DataFrame with cluster assignments
    """
    if spectral_df.empty or len(spectral_df) < 2:
        print("\nSkipping K-Means clustering (insufficient data).")
        return spectral_df
    
    print("\n=== SPECTRAL CLASSIFICATION ===")
    
    # Determine optimal number of clusters
    max_clusters = min(int(np.sqrt(len(spectral_df))), len(spectral_df) - 1, 5)
    num_clusters = max(2, max_clusters)
    
    print(f"Attempting K-Means clustering with {num_clusters} clusters...")
    
    # Prepare spectral features
    spectral_cols = [f'Channel_{i+1}_Mean' for i in range(num_channels)]
    
    if not all(col in spectral_df.columns for col in spectral_cols):
        print("Missing spectral columns for clustering.")
        return spectral_df
    
    # Remove rows with NaN values
    spectral_features_df = spectral_df.dropna(subset=spectral_cols)
    
    if len(spectral_features_df) < num_clusters:
        print("Insufficient valid ROIs for clustering.")
        return spectral_df
    
    try:
        # Normalize features
        spectral_features = spectral_features_df[spectral_cols].values
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(spectral_features)
        
        # Perform clustering
        actual_num_clusters = min(num_clusters, scaled_features.shape[0])
        kmeans = KMeans(n_clusters=actual_num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # Add cluster labels to DataFrame
        spectral_df.loc[spectral_features_df.index, 'Cluster'] = cluster_labels
        print(f"Assigned {len(cluster_labels)} ROIs to {actual_num_clusters} clusters.")
        
        # Print cluster summary
        for cluster_id in range(actual_num_clusters):
            cluster_rois = spectral_df[spectral_df['Cluster'] == cluster_id]['roi_id'].tolist()
            print(f"  Cluster {cluster_id}: ROIs {cluster_rois} ({len(cluster_rois)} ROIs)")
        
        # Visualize clusters
        visualize_clusters(spectral_df, actual_num_clusters, labeled_mask, 
                         segmentation_input, num_channels)
        
    except Exception as e:
        print(f"K-Means clustering failed: {e}")
    
    return spectral_df


def visualize_clusters(spectral_df, num_clusters, labeled_mask, segmentation_input, 
                      num_channels):
    """Visualize clustering results."""
    try:
        fig, (ax_cluster, ax_overlay) = plt.subplots(1, 2, figsize=(16, 8))
        cmap_clusters = plt.cm.get_cmap('tab10', num_clusters)
        
        # Create cluster map
        cluster_map = np.zeros_like(labeled_mask, dtype=float)
        roi_to_cluster = (spectral_df.dropna(subset=['Cluster'])
                         .set_index('roi_id')['Cluster'].to_dict())
        
        for roi_label in range(1, labeled_mask.max() + 1):
            if roi_label in roi_to_cluster:
                cluster_id = roi_to_cluster[roi_label]
                cluster_map[labeled_mask == roi_label] = cluster_id + 1
        
        # Left plot: Cluster assignments
        im_cluster = ax_cluster.imshow(cluster_map, cmap=cmap_clusters, 
                                     vmin=0, vmax=num_clusters, interpolation='nearest')
        ax_cluster.set_title(f'ROI Spectral Clusters (K-Means, k={num_clusters})')
        ax_cluster.axis('off')
        
        # Add colorbar
        if num_clusters > 0:
            tick_locs = np.arange(num_clusters) + 1.5
            cbar = plt.colorbar(im_cluster, ax=ax_cluster, ticks=tick_locs, 
                              fraction=0.046, pad=0.04)
            cbar.set_ticklabels(np.arange(num_clusters))
            cbar.set_label('Cluster ID')
        
        # Right plot: Overlay on original image
        ax_overlay.imshow(segmentation_input[0], cmap='gray', alpha=0.7)
        
        # Overlay cluster boundaries
        for cluster_id in range(num_clusters):
            cluster_mask = cluster_map == (cluster_id + 1)
            if np.any(cluster_mask):
                contours = find_contours(cluster_mask.astype(float), 0.5)
                for contour in contours:
                    ax_overlay.plot(contour[:, 1], contour[:, 0], linewidth=2, 
                                  color=cmap_clusters(cluster_id), alpha=0.8)
        
        ax_overlay.set_title('Clusters Overlaid on Reference Image')
        ax_overlay.axis('off')
        
        plt.tight_layout()
        plt.savefig('roi_clusters_detailed.png', dpi=150, bbox_inches='tight')
        print("Saved cluster visualization to roi_clusters_detailed.png")
        plt.close(fig)
        
        # Plot cluster spectral signatures
        plot_cluster_spectra(spectral_df, num_clusters, num_channels, cmap_clusters)
        
    except Exception as e:
        print(f"Error during cluster visualization: {e}")


def plot_cluster_spectra(spectral_df, num_clusters, num_channels, cmap_clusters):
    """Plot mean spectral signatures by cluster."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for cluster_id in range(num_clusters):
        cluster_data = spectral_df[spectral_df['Cluster'] == cluster_id]
        if cluster_data.empty:
            continue
        
        channel_numbers = np.arange(1, num_channels + 1)
        
        # Calculate mean and std for this cluster
        cluster_means = []
        cluster_stds = []
        for ch in range(num_channels):
            channel_name = f'Channel_{ch+1}_Mean'
            values = cluster_data[channel_name].dropna()
            cluster_means.append(np.mean(values))
            cluster_stds.append(np.std(values))
        
        cluster_means = np.array(cluster_means)
        cluster_stds = np.array(cluster_stds)
        
        color = cmap_clusters(cluster_id)
        ax.plot(channel_numbers, cluster_means, 
            marker='o', linewidth=3, markersize=6,
            label=f'Cluster {cluster_id} (n={len(cluster_data)})', 
            color=color)
        ax.fill_between(channel_numbers, 
                    cluster_means - cluster_stds,
                    cluster_means + cluster_stds,
                    alpha=0.2, color=color)
    
    ax.set_title('Mean Spectral Signatures by Cluster')
    ax.set_xlabel('Channel Number')
    ax.set_ylabel('Mean Intensity')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(channel_numbers)
    
    plt.tight_layout()
    plt.savefig('cluster_spectral_signatures.png', dpi=150, bbox_inches='tight')
    print("Saved cluster spectral signatures to cluster_spectral_signatures.png")
    plt.close(fig)


def save_results(spectral_df, num_channels):
    """
    Save analysis results to CSV files.
    
    Args:
        spectral_df (pd.DataFrame): Final spectral data DataFrame
        num_channels (int): Number of spectral channels
    """
    if spectral_df.empty:
        print("\nNo spectral data to save.")
        return
    
    try:
        # Basic CSV
        spectral_df.to_csv('roi_spectral_data_manual.csv', index=False)
        print("\nSaved basic spectral data to roi_spectral_data_manual.csv")
        
        # Enhanced CSV with additional metrics
        export_df = spectral_df.copy()
        spectral_cols = [f'Channel_{i+1}_Mean' for i in range(num_channels)]
        
        if all(col in export_df.columns for col in spectral_cols):
            export_df['spectral_mean'] = export_df[spectral_cols].mean(axis=1)
            export_df['spectral_std'] = export_df[spectral_cols].std(axis=1)
            export_df['spectral_max'] = export_df[spectral_cols].max(axis=1)
            export_df['spectral_min'] = export_df[spectral_cols].min(axis=1)
        
        enhanced_csv_path = 'roi_spectral_data_manual_enhanced.csv'
        export_df.to_csv(enhanced_csv_path, index=False)
        print(f"Saved enhanced spectral data to {enhanced_csv_path}")
        
    except Exception as e:
        print(f"\nError saving results: {e}")


def print_final_summary(spectral_df, num_channels):
    """Print final analysis summary."""
    print("\n" + "="*50)
    print("HYPERSPECTRAL ROI ANALYSIS COMPLETE")
    print("="*50)
    
    if spectral_df.empty:
        print("\nNo spectral data was extracted. Please check:")
        print("- ROI file path and format")
        print("- Image data compatibility")
        print("- ROI coordinate system alignment")
        return
    
    print(f"\nFINAL SUMMARY:")
    print(f"- Total ROIs analyzed: {len(spectral_df)}")
    print(f"- Spectral channels: {num_channels}")
    
    if 'Cluster' in spectral_df.columns:
        n_clusters = spectral_df['Cluster'].dropna().nunique()
        print(f"- Spectral clusters identified: {n_clusters}")
    
    if 'spectral_mean' in spectral_df.columns:
        print(f"- Intensity range: {spectral_df['spectral_min'].min():.2f} - "
              f"{spectral_df['spectral_max'].max():.2f}")
        print(f"- Mean intensity: {spectral_df['spectral_mean'].mean():.2f} ± "
              f"{spectral_df['spectral_mean'].std():.2f}")
    
    # List generated files
    visualizations = [
        'filtered_channel_projections_fixed_scale.png',
        'manual_roi_labeled_mask.png',
        'spectral_signatures_all_rois.png'
    ]
    
    # Add channel-specific files
    for i in range(num_channels):
        visualizations.append(f'roi_intensities_channel_{i+1}.png')
    
    # Add cluster files if clustering was performed
    if 'Cluster' in spectral_df.columns:
        visualizations.extend([
            'roi_clusters_detailed.png',
            'cluster_spectral_signatures.png'
        ])
    
    print(f"\nGenerated visualizations:")
    for viz in visualizations:
        print(f"  - {viz}")
    
    print("\nAnalysis pipeline completed successfully!")
    print("="*50)


def main():
    """Main analysis pipeline."""
    try:
        # Load and preprocess image data
        img_data, max_proj, segmentation_input = load_image_data(CZI_PATH)
        
        # Visualize channels
        vmin_val = np.min(segmentation_input)
        vmax_val = np.max(segmentation_input)
        visualize_channels(segmentation_input)
        
        # Load ROIs and create mask
        fiji_rois = load_fiji_rois(FIJI_ROI_ZIP_PATH)
        image_shape = (segmentation_input.shape[1], segmentation_input.shape[2])
        labeled_mask, num_labels = create_roi_mask(fiji_rois, image_shape)
        
        # Extract spectral data
        roi_spectral_data = extract_spectral_data(labeled_mask, segmentation_input, num_labels)
        num_channels = segmentation_input.shape[0]
        
        # Create DataFrame
        spectral_df = create_spectral_dataframe(roi_spectral_data, num_channels)
        
        # Print summary
        print_roi_summary(roi_spectral_data, num_channels)
        
        # Visualizations
        visualize_roi_mask(labeled_mask, segmentation_input, spectral_df, num_labels, vmin_val, vmax_val)
        plot_individual_channels(spectral_df, num_channels)
        plot_spectral_signatures(spectral_df, num_channels, num_labels)
        
        # Clustering analysis
        spectral_df = perform_clustering(spectral_df, num_channels, labeled_mask, segmentation_input)
        
        # Save results
        save_results(spectral_df, num_channels)
        
        # Final summary
        print_final_summary(spectral_df, num_channels)
        
    except Exception as e:
        print(f"Error in analysis pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
