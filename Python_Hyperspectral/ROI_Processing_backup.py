import numpy as np
import matplotlib.pyplot as plt
from bioio import BioImage
from scipy.ndimage import gaussian_filter, label, center_of_mass
from skimage.measure import regionprops, label as ski_label 
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans 
from roifile import roiread 
from skimage.draw import polygon, ellipse 
import zipfile
import glob
import os 


def create_roi_colormap(num_rois):
    """
    Create a consistent, distinguishable colormap for ROI visualization.
    Uses a simple color scheme that works well for up to 12 ROIs.
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
    """Get color for a specific ROI ID (1-indexed)."""
    return roi_colors[roi_id - 1] if roi_id <= len(roi_colors) else '#000000'


czi_path = "241121_10h40min_bra.h2b.mAp_twist.RFP_crbn.h2b.GFP_meis.kaede.czi"
fiji_roi_zip_path = "manual_rois.zip"

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
print(f"\nLoading manual ROIs from: {fiji_roi_zip_path}")
fiji_rois = []

try:
    # First try to read the zip file directly
    try:
        fiji_rois = roiread(fiji_roi_zip_path)
        print(f"Successfully read {len(fiji_rois)} ROIs from zip file")
    except Exception as e:
        print(f"Could not read zip file directly: {e}")
        # Try reading individual ROI files from the extracted directory
        import glob
        import os
        
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
            import zipfile
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
    print(f"ERROR: ROI file not found at {fiji_roi_zip_path}. Please check the path.")
    exit()
except Exception as e:
    print(f"Error reading ROI files: {e}")
    exit()

if not fiji_rois:
    print("No ROIs found in the provided file.")
    exit()
    
image_height = segmentation_input.shape[1]
image_width = segmentation_input.shape[2]

# Create an empty labeled mask
# This will store the ROIs, with each ROI having a unique integer ID.
filtered_labeled_mask = np.zeros((image_height, image_width), dtype=np.int32)
num_filtered_labels = 0

print(f"Processing {len(fiji_rois)} ROIs from FIJI...")
for i, roi in enumerate(fiji_rois):
    roi_id = i + 1 # Assign a unique ID (1, 2, 3, ...)
    num_filtered_labels += 1
    
    # Debug: Print ROI attributes to understand structure
    if i == 0:  # Only print for first ROI to avoid spam
        print(f"ROI attributes: {dir(roi)}")
        print(f"ROI type: {type(roi)}")
    
    # Get coordinates. Different roifile versions may have different attribute names
    try:
        # Try different possible coordinate attribute names
        if hasattr(roi, 'coordinates') and roi.coordinates is not None:
            coords = roi.coordinates()  # Call the function to get coordinates
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
                x_coords = [roi.left, roi.right] if hasattr(roi, 'left') and hasattr(roi, 'right') else [roi.x1, roi.x2]
                y_coords = [roi.top, roi.bottom] if hasattr(roi, 'top') and hasattr(roi, 'bottom') else [roi.y1, roi.y2]
        elif hasattr(roi, 'x') and hasattr(roi, 'y'):
            x_coords = roi.x
            y_coords = roi.y
        elif hasattr(roi, 'x1') and hasattr(roi, 'y1'):
            # For simple shapes, might have x1,y1,x2,y2
            x_coords = [roi.x1, roi.x2] if hasattr(roi, 'x2') else [roi.x1]
            y_coords = [roi.y1, roi.y2] if hasattr(roi, 'y2') else [roi.y1]
        elif hasattr(roi, 'left') and hasattr(roi, 'top'):
            # Rectangle using left, top, right, bottom
            x_coords = [roi.left, roi.right]
            y_coords = [roi.top, roi.bottom]
        else:
            print(f"Warning: Could not find coordinate attributes for ROI {roi_id}. Available attributes: {[attr for attr in dir(roi) if not attr.startswith('_')]}")
            num_filtered_labels -= 1
            continue
    except Exception as coord_e:
        print(f"Error extracting coordinates from ROI {roi_id}: {coord_e}")
        num_filtered_labels -= 1
        continue

    # Convert to numpy arrays if they aren't already
    x_coords = np.array(x_coords) if not isinstance(x_coords, np.ndarray) else x_coords
    y_coords = np.array(y_coords) if not isinstance(y_coords, np.ndarray) else y_coords
    
    # Get ROI type - different versions may have different type attributes
    roi_type_name = "unknown"
    if hasattr(roi, 'roitype'):
        roi_type = roi.roitype
        roi_type_name = getattr(roi, 'roitype_name', str(roi_type))
    elif hasattr(roi, 'type'):
        roi_type = roi.type
        roi_type_name = str(roi_type)
    
    # For polygon ROIs (most common for manual freehand/polygon selections)
    try:
        if len(x_coords) > 2 and len(y_coords) > 2:  # Polygon-like shape
            # skimage.draw.polygon expects rows (y) and cols (x)
            # Ensure coordinates are within image bounds
            x_coords = np.clip(x_coords, 0, image_width - 1)
            y_coords = np.clip(y_coords, 0, image_height - 1)
            
            rr, cc = polygon(y_coords, x_coords, shape=filtered_labeled_mask.shape)
            filtered_labeled_mask[rr, cc] = roi_id
            print(f"  Added polygon ROI {roi_id} with {len(x_coords)} points")
            
        elif len(x_coords) == 2 and len(y_coords) == 2:  # Rectangle
            # For rectangles: use coordinates as corners
            x1, x2 = int(min(x_coords)), int(max(x_coords))
            y1, y2 = int(min(y_coords)), int(max(y_coords))
            
            # Ensure coordinates are within bounds
            x1, x2 = max(0, x1), min(image_width - 1, x2)
            y1, y2 = max(0, y1), min(image_height - 1, y2)
            
            filtered_labeled_mask[y1:y2+1, x1:x2+1] = roi_id
            print(f"  Added rectangular ROI {roi_id} at ({x1},{y1}) to ({x2},{y2})")
            
        else:
            print(f"Warning: ROI {roi_id} has unexpected coordinate structure (x: {len(x_coords)}, y: {len(y_coords)}). Skipping.")
            num_filtered_labels -= 1
            continue
            
    except Exception as draw_e:
        print(f"Error drawing ROI {roi_id}: {draw_e}")
        num_filtered_labels -= 1
        continue
        
    # Try to get ROI name
    roi_name = f"ROI_{roi_id}"
    if hasattr(roi, 'name') and roi.name:
        roi_name = roi.name
    elif hasattr(roi, 'label') and roi.label:
        roi_name = roi.label
        
    print(f"  Added ROI: {roi_name} as ID {roi_id} (type: {roi_type_name})")


if num_filtered_labels > 0:
    print(f"\nSuccessfully created a labeled mask with {num_filtered_labels} ROIs from FIJI selections.")
else:
    print("\nCould not create a labeled mask from FIJI ROIs. Exiting.")
    exit()



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
    # Create figure with space for legend
    fig_manual_rois = plt.figure(figsize=(16, 6))
    
    # Create subplot layout: reference image, labeled ROIs, and legend
    ax_ref = plt.subplot(1, 3, 1)
    ax_labeled = plt.subplot(1, 3, 2)
    ax_legend = plt.subplot(1, 3, 3)

    # Display one channel of the original image (e.g., first channel of segmentation_input)
    # This provides context for where the ROIs are.
    ax_ref.imshow(segmentation_input[14], cmap='gray', vmin=vmin_val, vmax=vmax_val)
    ax_ref.set_title('Reference Image Channel 14 w/ Max Z Projection')
    ax_ref.axis('off')

    # Display the labeled mask from FIJI ROIs using consistent color mapping
    # Create consistent colormap for ROIs
    roi_colors = create_roi_colormap(num_filtered_labels)
    
    # Create a custom colormap from our consistent colors
    from matplotlib.colors import ListedColormap
    # Add black for background (0) plus our ROI colors
    roi_colormap_colors = ['#000000'] + roi_colors  # Black background + ROI colors
    custom_cmap = ListedColormap(roi_colormap_colors)

    im_labeled = ax_labeled.imshow(filtered_labeled_mask, cmap=custom_cmap, interpolation='nearest', vmin=0, vmax=num_filtered_labels)
    ax_labeled.set_title(f'Manually Labeled ROIs ({num_filtered_labels})')
    ax_labeled.axis('off')
    
    # Optionally, add centroids to the labeled image if spectral_df is populated
    if not spectral_df.empty and 'centroid_col' in spectral_df.columns and 'centroid_row' in spectral_df.columns:
        ax_labeled.scatter(spectral_df['centroid_col'], spectral_df['centroid_row'], s=15, c='white', marker='x', linewidth=2, label='Centroids')
        ax_labeled.scatter(spectral_df['centroid_col'], spectral_df['centroid_row'], s=10, c='black', marker='x', linewidth=1)

    # Create color index/legend for ROI numbers
    ax_legend.set_title('ROI Color Index', fontweight='bold', fontsize=12)
    ax_legend.axis('off')
    
    # Create color patches for each ROI
    from matplotlib.patches import Rectangle
    legend_elements = []
    
    # Calculate layout for legend
    n_cols = 2 if num_filtered_labels > 6 else 1
    n_rows = (num_filtered_labels + n_cols - 1) // n_cols
    
    for roi_id in range(1, num_filtered_labels + 1):
        # Get the color for this ROI from our consistent colormap
        color = get_roi_color(roi_id, roi_colors)
        
        # Calculate position for this legend entry
        col = (roi_id - 1) % n_cols
        row = (roi_id - 1) // n_cols
        
        x_pos = col * 0.5
        y_pos = 0.9 - (row * 0.15)
        
        # Add colored rectangle
        rect = Rectangle((x_pos, y_pos), 0.1, 0.08, facecolor=color, edgecolor='black', linewidth=0.5)
        ax_legend.add_patch(rect)
        
        # Add ROI number text
        ax_legend.text(x_pos + 0.15, y_pos + 0.04, f'ROI {roi_id}', 
                      verticalalignment='center', fontsize=10, fontweight='bold')
        
        # Add ROI name if available
        if not spectral_df.empty:
            roi_row = spectral_df[spectral_df['roi_id'] == roi_id]
            if not roi_row.empty:
                centroid_x = roi_row['centroid_col'].iloc[0]
                centroid_y = roi_row['centroid_row'].iloc[0]
                ax_legend.text(x_pos + 0.15, y_pos + 0.01, f'({centroid_x:.0f}, {centroid_y:.0f})', 
                              verticalalignment='center', fontsize=8, color='gray')
    
    # Set legend axis limits
    ax_legend.set_xlim(0, n_cols * 0.5)
    ax_legend.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('manual_roi_labeled_mask.png', dpi=150, bbox_inches='tight')
    print("\nSaved manual ROI labeled mask visualization with color index to manual_roi_labeled_mask.png")
    plt.close(fig_manual_rois)
else:
    print("\nNo manual ROIs to visualize.")



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
        
        # Use our consistent color mapping for ROIs
        roi_colors = create_roi_colormap(num_filtered_labels)

        channel_numbers = np.arange(1, num_channels + 1)
        
        plotted_rois = 0
        for idx, (_, roi_row) in enumerate(spectral_df.iterrows()):
            roi_id = roi_row['roi_id']
            spectral_cols = [f'Channel_{i+1}_Mean' for i in range(num_channels)]
            
            if all(col in spectral_df.columns for col in spectral_cols):
                spectral_values = roi_row[spectral_cols].values
                
                # Check for NaN values
                if not np.isnan(spectral_values).any():
                    # Use consistent color for this ROI
                    roi_color = get_roi_color(roi_id, roi_colors)
                    ax_spectral.plot(channel_numbers, spectral_values, 
                                   marker='o', linestyle='-', linewidth=2, markersize=4,
                                   label=f'ROI {roi_id}', color=roi_color, alpha=0.8)
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

else:
    print("\nSkipping segmentation/histogram visualization (no spectral data extracted or matplotlib unavailable).")





# --- Classification (using K-Means)
if spectral_df.empty or num_filtered_labels < 2:
    print("\nSkipping K-Means clustering (no data or not enough ROIs).")
else:
    print("\n=== SPECTRAL CLASSIFICATION ===")
    # Automatically determine optimal number of clusters (up to sqrt(n_samples))
    max_clusters = min(int(np.sqrt(len(spectral_df))), len(spectral_df) - 1, 5)  # Cap at 5 for visualization
    num_clusters = max(2, max_clusters)  # At least 2 clusters
    
    print(f"Attempting K-Means clustering with {num_clusters} clusters...")
    
    # Ensure all spectral channel columns exist
    spectral_cols = [f'Channel_{i+1}_Mean' for i in range(num_channels)]
    if all(col in spectral_df.columns for col in spectral_cols):
        # Drop rows with any NaN values in the spectral columns before clustering
        spectral_features_df = spectral_df.dropna(subset=spectral_cols)

        if not spectral_features_df.empty and spectral_features_df.shape[0] >= num_clusters : # Check if enough data points remain
            spectral_features = spectral_features_df[spectral_cols].values

            # Normalize features (important for K-Means)
            try:
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(spectral_features)

                # Ensure number of clusters doesn't exceed number of samples
                actual_num_clusters = min(num_clusters, scaled_features.shape[0])
                if actual_num_clusters < 2:
                    print("Skipping K-Means: Need at least 2 valid ROIs/clusters after cleaning.")
                else:
                    kmeans = KMeans(n_clusters=actual_num_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(scaled_features)

                    # Add cluster labels back to the original DataFrame (matching by index)
                    spectral_df.loc[spectral_features_df.index, 'Cluster'] = cluster_labels
                    print(f"Assigned {len(cluster_labels)} ROIs to {actual_num_clusters} clusters using K-Means.")
                    
                    # Print cluster summary
                    for cluster_id in range(actual_num_clusters):
                        cluster_rois = spectral_df[spectral_df['Cluster'] == cluster_id]['roi_id'].tolist()
                        print(f"  Cluster {cluster_id}: ROIs {cluster_rois} ({len(cluster_rois)} ROIs)")

                    # Visualize clusters on the image (if matplotlib available)
                    try:
                        fig_cluster, (ax_cluster, ax_cluster_overlay) = plt.subplots(1, 2, figsize=(16, 8))
                        cmap_clusters = plt.cm.get_cmap('tab10', actual_num_clusters)
                        
                        # Left plot: Cluster assignments only
                        cluster_map = np.zeros_like(filtered_labeled_mask, dtype=float)

                        # Create a mapping from ROI ID to Cluster ID for ROIs that were clustered
                        roi_to_cluster = spectral_df.dropna(subset=['Cluster']).set_index('roi_id')['Cluster'].to_dict()

                        # Fill the cluster map based on the labeled ROIs and the clustering result
                        for roi_label in range(1, filtered_labeled_mask.max() + 1):
                            if roi_label in roi_to_cluster:
                                cluster_id = roi_to_cluster[roi_label]
                                # Add 1 to cluster_id so background is 0 and clusters start at 1
                                cluster_map[filtered_labeled_mask == roi_label] = cluster_id + 1

                        im_cluster = ax_cluster.imshow(cluster_map, cmap=cmap_clusters, vmin=0, vmax=actual_num_clusters, interpolation='nearest')
                        ax_cluster.set_title(f'ROI Spectral Clusters (K-Means, k={actual_num_clusters})')
                        ax_cluster.axis('off')

                        # Add a colorbar for cluster map
                        if actual_num_clusters > 0:
                            tick_locs = np.arange(actual_num_clusters) + 1.5 # Centered ticks
                            cbar = plt.colorbar(im_cluster, ax=ax_cluster, ticks=tick_locs, fraction=0.046, pad=0.04)
                            cbar.set_ticklabels(np.arange(actual_num_clusters))
                            cbar.set_label('Cluster ID')
                        
                        # Right plot: Overlay on original image
                        ax_cluster_overlay.imshow(segmentation_input[0], cmap='gray', alpha=0.7)
                        
                        # Overlay cluster boundaries
                        for cluster_id in range(actual_num_clusters):
                            cluster_mask = cluster_map == (cluster_id + 1)
                            if np.any(cluster_mask):
                                # Create contour for this cluster
                                from skimage.measure import find_contours
                                contours = find_contours(cluster_mask.astype(float), 0.5)
                                for contour in contours:
                                    ax_cluster_overlay.plot(contour[:, 1], contour[:, 0], linewidth=2, 
                                                          color=cmap_clusters(cluster_id), alpha=0.8)
                                
                        ax_cluster_overlay.set_title('Clusters Overlaid on Reference Image')
                        ax_cluster_overlay.axis('off')

                        plt.tight_layout()
                        plt.savefig('roi_clusters_detailed.png', dpi=150, bbox_inches='tight')
                        print("Saved detailed ROI cluster visualization to roi_clusters_detailed.png")
                        plt.close(fig_cluster)
                        
                        # Create cluster-based spectral signature plot
                        fig_cluster_spectra, ax_cluster_spectra = plt.subplots(figsize=(12, 8))
                        
                        for cluster_id in range(actual_num_clusters):
                            cluster_data = spectral_df[spectral_df['Cluster'] == cluster_id]
                            if not cluster_data.empty:
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
                                ax_cluster_spectra.plot(channel_numbers, cluster_means, 
                                                      marker='o', linewidth=3, markersize=6,
                                                      label=f'Cluster {cluster_id} (n={len(cluster_data)})', 
                                                      color=color)
                                ax_cluster_spectra.fill_between(channel_numbers, 
                                                               cluster_means - cluster_stds,
                                                               cluster_means + cluster_stds,
                                                               alpha=0.2, color=color)
                        
                        ax_cluster_spectra.set_title(f'Mean Spectral Signatures by Cluster')
                        ax_cluster_spectra.set_xlabel('Channel Number')
                        ax_cluster_spectra.set_ylabel('Mean Intensity')
                        ax_cluster_spectra.grid(True, alpha=0.3)
                        ax_cluster_spectra.legend()
                        ax_cluster_spectra.set_xticks(channel_numbers)
                        
                        plt.tight_layout()
                        plt.savefig('cluster_spectral_signatures.png', dpi=150, bbox_inches='tight')
                        print("Saved cluster spectral signatures to cluster_spectral_signatures.png")
                        plt.close(fig_cluster_spectra)
                        
                    except Exception as e:
                        print(f"Error during cluster visualization: {e}")
                        if 'fig_cluster' in locals() and fig_cluster is not None:
                            plt.close(fig_cluster)
                        if 'fig_cluster_spectra' in locals() and fig_cluster_spectra is not None:
                            plt.close(fig_cluster_spectra)

            except Exception as e:
                print(f"K-Means clustering failed: {e}")
        else:
            print("\nSkipping K-Means clustering: Not enough valid spectral features after removing NaNs or fewer ROIs than clusters.")
    else:
        print("\nSkipping K-Means clustering: Not all spectral channel columns found in DataFrame.")

# --- Final Summary ---
print("\n" + "="*50)
print("HYPERSPECTRAL ROI ANALYSIS COMPLETE")
print("="*50)

if not spectral_df.empty:
    # Enhanced CSV export with cluster information
    try:
        # Add spatial information and cluster details to the export
        export_df = spectral_df.copy()
        
        # Calculate additional metrics for each ROI
        spectral_cols = [f'Channel_{i+1}_Mean' for i in range(num_channels)]
        if all(col in export_df.columns for col in spectral_cols):
            # Calculate spectral statistics for each ROI
            export_df['spectral_mean'] = export_df[spectral_cols].mean(axis=1)
            export_df['spectral_std'] = export_df[spectral_cols].std(axis=1)
            export_df['spectral_max'] = export_df[spectral_cols].max(axis=1)
            export_df['spectral_min'] = export_df[spectral_cols].min(axis=1)
        
        # Save enhanced CSV
        enhanced_csv_path = 'roi_spectral_data_manual_enhanced.csv'
        export_df.to_csv(enhanced_csv_path, index=False)
        print(f"\nSaved enhanced spectral data to {enhanced_csv_path}")
        
        # Print final summary statistics
        print(f"\nFINAL SUMMARY:")
        print(f"- Total ROIs analyzed: {len(export_df)}")
        print(f"- Spectral channels: {num_channels}")
        if 'Cluster' in export_df.columns:
            n_clusters = export_df['Cluster'].dropna().nunique()
            print(f"- Spectral clusters identified: {n_clusters}")
        if 'spectral_mean' in export_df.columns:
            print(f"- Overall spectral intensity range: {export_df['spectral_min'].min():.2f} - {export_df['spectral_max'].max():.2f}")
            print(f"- Mean spectral intensity: {export_df['spectral_mean'].mean():.2f} ± {export_df['spectral_mean'].std():.2f}")
        
        print(f"\nGenerated visualizations:")
        visualizations = [
            'filtered_channel_projections_fixed_scale.png',
            'manual_roi_labeled_mask.png',
            'spectral_signatures_all_rois.png'
        ]
        
        # Check for ROI intensity plots
        for i in range(num_channels):
            viz_file = f'roi_intensities_channel_{i+1}.png'
            visualizations.append(viz_file)
        
        # Check for cluster visualizations
        if 'Cluster' in export_df.columns:
            visualizations.extend([
                'roi_clusters_detailed.png',
                'cluster_spectral_signatures.png'
            ])
        
        for viz in visualizations:
            print(f"  - {viz}")
        
    except Exception as e:
        print(f"\nError during final data export: {e}")
        
else:
    print("\nNo spectral data was extracted. Please check:")
    print("- ROI file path and format")
    print("- Image data compatibility")
    print("- ROI coordinate system alignment")

print("\nAnalysis pipeline completed successfully!")
print("="*50)
