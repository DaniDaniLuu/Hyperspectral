import numpy as np
import matplotlib.pyplot as plt
from bioio import BioImage
from scipy.ndimage import gaussian_filter, label, center_of_mass
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label as ski_label 
from skimage.morphology import reconstruction
from skimage import morphology
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans 


czi_path = "241121_10h40min_bra.h2b.mAp_twist.RFP_crbn.h2b.GFP_meis.kaede.czi"
img = BioImage(czi_path)
img_data = img.data.squeeze() # Removes the time dimension bc we only have one timepoint

# --- Parameters
segmentation_method = 'h_maxima' # Options: otsu, h_maxima
thresholding_channel_index = 0 
# ---

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




#1 --- Segmentation and Spectral Extraction
num_channels = segmentation_input.shape[0] # Get number of channels from the input used
spectral_df = pd.DataFrame() # Initialize empty DataFrame
binary_mask = None # Initialize binary mask
segmentation_image = None # Image used to generate the mask

segmentation_image = segmentation_input[thresholding_channel_index]
binary_mask = np.zeros_like(segmentation_image, dtype=bool) # Default empty mask
print(f"\nGenerating segmentation mask using channel {thresholding_channel_index+1} with method: {segmentation_method}")

# --- H-Maxima Segmentation ---
h_value = 0
if segmentation_method == 'h_maxima':
    try:
        print(f"Applying H-Maxima with h = {h_value}")
        # Define mask and markers for reconstruction
        mask_img = segmentation_image
        # Ensure markers do not have negative values if input is float
        markers = np.clip(mask_img - h_value, np.min(mask_img), None) # Clip at image min or higher

        # Perform morphological reconstruction by dilation
        reconstructed_img = reconstruction(markers, mask_img, method='dilation')

        # Calculate the h-maxima image (difference between mask and reconstruction)
        h_maxima_img = mask_img - reconstructed_img

        # Threshold the h-maxima image (peaks higher than h)
        # Using > 0 is often sufficient as reconstruction suppresses lower peaks.
        # User requested > h, so using that. Consider changing if needed.
        binary_mask = h_maxima_img > h_value
        print(f"H-Maxima segmentation complete. Found {np.sum(binary_mask)} foreground pixels.")

    except Exception as e:
        print(f"Error during H-Maxima segmentation: {e}")
        binary_mask = np.zeros_like(segmentation_image, dtype=bool) # Ensure mask is empty on error

# --- Otsu Thresholding Segmentation ---
elif segmentation_method == 'otsu':
    try:
        print("Applying Otsu thresholding")
        # Check if the image data is constant before thresholding
        if np.all(segmentation_image == segmentation_image.flat[0]):
                print(f"Warning: Segmentation image data (Channel {thresholding_channel_index+1}) is constant. Otsu thresholding cannot be applied.")
                # Keep binary_mask as all False
                thresh_value = segmentation_image.flat[0]
        else:
            thresh_value = threshold_otsu(segmentation_image)
            binary_mask = segmentation_image > thresh_value
            print(f"Otsu threshold value for channel {thresholding_channel_index+1}: {thresh_value:.2f}")
            print(f"Otsu segmentation complete. Found {np.sum(binary_mask)} foreground pixels.")

    except ValueError as e:
        print(f"Otsu thresholding failed for channel {thresholding_channel_index+1}: {e}.")
        binary_mask = np.zeros_like(segmentation_image, dtype=bool) # Ensure mask is empty on error
    except Exception as e:
        print(f"An unexpected error occurred during Otsu thresholding: {e}")
        binary_mask = np.zeros_like(segmentation_image, dtype=bool) # Ensure mask is empty on error
        
#2 --- Label connected regions in the binary mask ---
labeled_mask = np.zeros_like(binary_mask, dtype=int)
num_labels = 0
if np.any(binary_mask): # Proceed only if mask is not empty
    try:
        # connectivity=2 corresponds to 8-connectivity for 2D images
        labeled_mask, num_labels = ski_label(binary_mask, connectivity=2, return_num=True)
        print(f"Number of potential ROIs found: {num_labels}")
    except Exception as e:
        print(f"Error during labeling: {e}")
else:
    print("Binary mask is empty, skipping labeling.")

# --- 3. Filter ROIs by size ---
min_roi_size = 20 # Minimum number of pixels for an ROI to be kept
filtered_labeled_mask = np.copy(labeled_mask)
num_filtered_labels = 0

if num_labels > 0: # Proceed only if labels were found
    try:
        component_sizes = np.bincount(labeled_mask.ravel())
        # Identify labels that are too small (excluding background label 0)
        # Ensure indices are within the bounds of component_sizes
        if len(component_sizes) > 1: # Check if there are any non-background labels
                valid_labels_mask = np.arange(len(component_sizes)) > 0
                small_label_indices = np.where(valid_labels_mask & (component_sizes < min_roi_size))[0]

                if len(small_label_indices) > 0:
                    # Create a boolean mask for labels that are too small
                    too_small_mask = np.isin(labeled_mask, small_label_indices)
                    filtered_labeled_mask[too_small_mask] = 0 # Set small components to background

        # Relabel sequentially after filtering to ensure labels are consecutive (1, 2, 3...)
        filtered_labeled_mask, num_filtered_labels = ski_label(filtered_labeled_mask, connectivity=2, return_num=True)
        print(f"Number of ROIs after size filtering (min size {min_roi_size}): {num_filtered_labels}")
    except IndexError:
            print("Warning: Index error during ROI size filtering (likely no ROIs found initially). Using initial labeled mask.")
            # This can happen if labeled_mask only contains 0s
            filtered_labeled_mask = labeled_mask
            num_filtered_labels = 0 # No labels remain after filtering attempt
    except Exception as e:
        print(f"Error during ROI size filtering: {e}")
        # Fallback: use the unfiltered labeled mask
        filtered_labeled_mask = labeled_mask
        num_filtered_labels = num_labels
        print("Warning: Using unfiltered ROIs due to filtering error.")
else:
    print("No ROIs found to filter.")




#4 --- Extract Spectral Information for each ROI ---
num_channels = segmentation_input.shape[0]
roi_spectral_data = []

# Use regionprops to get properties of labeled regions
props = regionprops(filtered_labeled_mask, intensity_image=np.moveaxis(segmentation_input, 0, -1)) # Move channel axis to last for intensity_image

for region in props:
    # region.label gives the ID of the current ROI
    roi_id = region.label
    # Get the coordinates of the pixels belonging to this ROI
    roi_coords = region.coords # shape (N, 2) where N is number of pixels in ROI

    # Extract the mean intensity for this ROI from *each* channel
    spectral_signature = []
    for c in range(num_channels):
        channel_data = segmentation_input[c]
        # Extract pixel values for the current ROI from this channel
        roi_pixels_in_channel = channel_data[roi_coords[:, 0], roi_coords[:, 1]]
        # Calculate mean intensity for this channel within the ROI
        mean_intensity = np.mean(roi_pixels_in_channel)
        spectral_signature.append(mean_intensity)

    # Store the signature along with the ROI ID (optional)
    roi_spectral_data.append({
        'roi_id': roi_id,
        'centroid': region.centroid, # (row, col)
        'spectral_signature': np.array(spectral_signature)
    })

print(f"Extracted spectral data for {len(roi_spectral_data)} ROIs.")

# # Convert to DataFrame for easier analysis (optional)
# if roi_spectral_data:
#     spectral_df = pd.DataFrame(roi_spectral_data)
#     # Expand the spectral signature list into separate columns
#     spectral_signatures_df = pd.DataFrame(spectral_df['spectral_signature'].tolist(),
#                                         columns=[f'Channel_{i+1}_Mean' for i in range(num_channels)])
#     spectral_df = pd.concat([spectral_df.drop('spectral_signature', axis=1), spectral_signatures_df], axis=1)
#     print("\nSpectral Data DataFrame Head:")
#     print(spectral_df.head())
#     # Save spectral data to CSV
#     try:
#         spectral_df.to_csv('roi_spectral_data.csv', index=False)
#         print("\nSaved spectral data to roi_spectral_data.csv")
#     except Exception as e:
#         print(f"\nError saving spectral data to CSV: {e}")

# else:
#     print("\nNo ROIs found or extracted, skipping DataFrame creation and plotting.")
#     spectral_df = pd.DataFrame() # Empty DataFrame





# --- Vis: Segmentation and Combined Histogram
if not spectral_df.empty:
    # Plot 1: Show the thresholding channel, binary mask, and labeled ROIs
    try:
        fig_seg, axes_seg = plt.subplots(1, 3, figsize=(18, 6))

        ax1 = axes_seg[0]
        im1 = ax1.imshow(segmentation_image, cmap='gray')
        ax1.set_title(f'Channel {thresholding_channel_index+1} for Thresholding')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = axes_seg[1]
        ax2.imshow(binary_mask, cmap='gray')
        ax2.set_title('Binary Mask (Otsu)')
        ax2.axis('off')

        ax3 = axes_seg[2]
        if num_filtered_labels > 0:
            # Use a colormap that distinguishes labels well, skip background (0)
            num_unique_labels = num_filtered_labels + 1 # +1 for background
            # Use a robust colormap like 'viridis' or 'plasma' if 'nipy_spectral' causes issues
            try:
                cmap_labels = plt.cm.get_cmap('nipy_spectral', num_unique_labels)
            except ValueError:
                print("Warning: Could not create nipy_spectral colormap with specified labels, using default.")
                cmap_labels = plt.cm.get_cmap('viridis', num_unique_labels)

            im3 = ax3.imshow(filtered_labeled_mask, cmap=cmap_labels, interpolation='nearest')
            ax3.set_title(f'Labeled ROIs ({num_filtered_labels})')
            # Add centroids to the labeled image
            if 'centroid_col' in spectral_df.columns and 'centroid_row' in spectral_df.columns:
                # Plot centroids (y, x) -> (col, row) for scatter
                ax3.scatter(spectral_df['centroid_col'], spectral_df['centroid_row'], s=10, c='red', marker='x', label='Centroids')
        else:
            # Show empty mask if no labels
            ax3.imshow(filtered_labeled_mask, cmap='gray')
            ax3.set_title('No ROIs Found')

        ax3.axis('off')

        plt.tight_layout()
        plt.savefig('roi_segmentation.png', dpi=150, bbox_inches='tight')
        print("\nSaved ROI segmentation visualization to roi_segmentation.png")
        plt.close(fig_seg) # Close the specific figure
    except Exception as e:
        print(f"Error during segmentation visualization: {e}")
        if 'fig_seg' in locals() and fig_seg is not None:
            plt.close(fig_seg)
            
    # Plot 2: Individual Histograms for each channel ---
    print("\nPlotting individual histograms for each channel...")
    num_bins = 30 # Number of bins for the histogram
    individual_hist_figs = [] # Keep track of figures to close
    try:
        for i in range(num_channels):
            channel_name = f'Channel_{i+1}_Mean'
            if channel_name in spectral_df.columns:
                intensities = spectral_df[channel_name].dropna()
                if not intensities.empty:
                    # Create a new figure for each channel's histogram
                    fig_hist_ind, ax_hist_ind = plt.subplots(figsize=(8, 5))
                    individual_hist_figs.append(fig_hist_ind) # Add fig to list

                    ax_hist_ind.hist(intensities, bins=num_bins, edgecolor='black', color='skyblue')
                    ax_hist_ind.set_title(f'Histogram of Mean ROI Intensities for Channel {i+1}')
                    ax_hist_ind.set_xlabel('Mean Intensity per ROI')
                    ax_hist_ind.set_ylabel('Number of ROIs')
                    ax_hist_ind.grid(axis='y', alpha=0.75)

                    plt.tight_layout()
                    filename = f'histogram_channel_{i+1}.png'
                    plt.savefig(filename, dpi=150, bbox_inches='tight')
                    print(f"  - Saved histogram to {filename}")
                    plt.close(fig_hist_ind) # Close the figure after saving
                    individual_hist_figs.remove(fig_hist_ind) # Remove from list after closing
                else:
                    print(f"  - Skipping {channel_name} (no valid data)")
            else:
                print(f"  - Skipping {channel_name} (column not found)")
    except Exception as e:
        print(f"Error during individual histogram plotting: {e}")
    finally:
        # Ensure all created figures are closed even if an error occurred mid-loop
        for fig in individual_hist_figs:
            try:
                plt.close(fig)
            except Exception:
                pass # Ignore errors during cleanup closing


    # Plot 3: Combined Histogram of mean intensities for ALL channels across all ROIs
    try:
        fig_hist_all, ax_hist_all = plt.subplots(figsize=(12, 7)) # Adjusted size
        num_bins = 30 # Number of bins for the histogram

        # Define a colormap to cycle through for different channels
        colors = plt.cm.viridis(np.linspace(0, 1, num_channels))

        print("\nPlotting combined histogram for all channels...")
        plotted_channels = 0
        for i in range(num_channels):
            channel_name = f'Channel_{i+1}_Mean'
            if channel_name in spectral_df.columns:
                # Drop NaNs before plotting histogram
                intensities = spectral_df[channel_name].dropna()
                if not intensities.empty:
                    # Plot histogram for this channel with transparency and label
                    ax_hist_all.hist(intensities, bins=num_bins, edgecolor='black', alpha=0.6,
                                    label=f'Ch {i+1}', color=colors[i], histtype='stepfilled') # Use stepfilled or step
                            # Alternatively use histtype='step' for outlines only:
                            # ax_hist_all.hist(intensities, bins=num_bins, alpha=0.8,
                            # label=f'Ch {i+1}', color=colors[i], histtype='step', linewidth=1.5)
                    plotted_channels += 1
                else:
                    print(f"  - Skipping {channel_name} (no valid data)")
            else:
                print(f"  - Skipping {channel_name} (column not found)")

        if plotted_channels > 0:
            ax_hist_all.set_title(f'Histogram of Mean ROI Intensities (All {plotted_channels} Channels)')
            ax_hist_all.set_xlabel('Mean Intensity per ROI')
            ax_hist_all.set_ylabel('Number of ROIs')
            ax_hist_all.grid(axis='y', alpha=0.75)
            # Add legend - place outside plot area if too crowded
            ax_hist_all.legend(title="Channels", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            # Or inside if space permits: ax_hist_all.legend(title="Channels", fontsize='small')

            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend if outside
            # Or just plt.tight_layout() if legend is inside

            plt.savefig(f'histogram_all_channels.png', dpi=150, bbox_inches='tight')
            print(f"\nSaved combined histogram for all channels to histogram_all_channels.png")
        else:
            print("\nNo channels plotted in the combined histogram.")

        plt.close(fig_hist_all) # Close the specific figure

    except Exception as e:
        print(f"Error during combined histogram plotting: {e}")
        if 'fig_hist_all' in locals() and fig_hist_all is not None:
            plt.close(fig_hist_all)

else:
    print("\nSkipping segmentation/histogram visualization (no spectral data extracted or matplotlib unavailable).")





# --- Classification (using K-Means)
if spectral_df.empty or num_filtered_labels < 2:
    print("\nSkipping K-Means clustering (no data or not enough ROIs).")
else:
    print("\nAttempting K-Means clustering...")
    num_clusters = 3 # Define how many spectral clusters you expect
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

                    # Visualize clusters on the image (if matplotlib available)
                    try:
                        fig_cluster, ax_cluster = plt.subplots(figsize=(8, 8))
                        cmap_clusters = plt.cm.get_cmap('viridis', actual_num_clusters)
                        # Create an image showing cluster assignments
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

                        # Add a colorbar
                        # Create ticks centered within the colorbar segments
                        if actual_num_clusters > 0:
                            tick_locs = np.arange(actual_num_clusters) + 1.5 # Centered ticks
                            cbar = plt.colorbar(im_cluster, ticks=tick_locs, fraction=0.046, pad=0.04)
                            # Set tick labels to be the cluster numbers (0 to k-1)
                            cbar.set_ticklabels(np.arange(actual_num_clusters))
                            cbar.set_label('Cluster ID')


                        plt.tight_layout()
                        plt.savefig('roi_clusters.png', dpi=150, bbox_inches='tight')
                        print("Saved ROI cluster visualization to roi_clusters.png")
                        plt.close(fig_cluster)
                    except Exception as e:
                        print(f"Error during cluster visualization: {e}")
                        if 'fig_cluster' in locals() and fig_cluster is not None:
                            plt.close(fig_cluster)

            except Exception as e:
                print(f"K-Means clustering failed: {e}")
        else:
            print("\nSkipping K-Means clustering: Not enough valid spectral features after removing NaNs or fewer ROIs than clusters.")
    else:
        print("\nSkipping K-Means clustering: Not all spectral channel columns found in DataFrame.")
