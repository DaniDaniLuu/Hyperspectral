import numpy as np
import matplotlib.pyplot as plt
from bioio import BioImage 
from skimage import filters, measure, segmentation, feature, morphology, util
from scipy import ndimage as ndi, signal
import cv2

def load_hyperspectral_image(path):
    """
    Load a hyperspectral .czi image and return a numpy array of shape (H, W, C).
    """
    # Read the image using czifile (returns a multi-dimensional numpy array)
    img = BioImage(path)
    img_data = img.data  # This is a 5D array (T, C, Z, Y, X)
    
    arr = np.array(img_data)  # Ensure it's a numpy array (if not already)
    # Identify the channel dimension (assuming 15 channels):
    channel_dim = None

    for dim, size in enumerate(arr.shape):
        if size == 15:  # likely channel axis
            channel_dim = dim
            break
    if channel_dim is None:
        raise ValueError("Channel dimension of size 15 not found. Please adjust dimension handling.")
    # Move channel dimension to last position for convenience
    arr = np.moveaxis(arr, channel_dim, -1)
    # If the image has more than 3 dims (e.g. Time, Z, etc.), take the first in those dimensions
    if arr.ndim > 3:
        # Collapse extra dims by taking the first index of each (assuming order: T, Z, etc.)
        arr = arr[0,...]  # take first time if exists
        arr = arr[0,...]  # take first Z if exists (repeat if multiple extra dims)
    return arr

def detect_rois(image):
    """
    Apply Otsu thresholding on a combined intensity projection to detect ROIs.
    Returns a binary mask and labeled ROIs.
    """
    # Combine spectral channels into a single grayscale image (mean projection)
    gray = image.mean(axis=2)
    # Compute Otsu's threshold&#8203;:contentReference[oaicite:5]{index=5} for the grayscale image
    thresh_val = filters.threshold_otsu(gray)
    # Create binary mask of foreground vs background
    binary_mask = gray >= thresh_val
    # Optional: remove very small objects/noise (less than e.g. 10 pixels area)
    binary_mask = morphology.remove_small_objects(binary_mask, min_size=10)
    # Label connected components in the binary mask
    roi_labels = measure.label(binary_mask)
    num_rois = roi_labels.max()
    print(f"Detected {num_rois} initial ROI(s) using Otsu threshold = {thresh_val:.3f}")
    return binary_mask, roi_labels

def segment_cells(binary_mask):
    """
    Perform watershed segmentation on the binary ROI mask to separate individual cells.
    """
    # 1. Distance transform of the binary mask
    dist_map = ndi.distance_transform_edt(binary_mask)

    # 2. Find peak coordinates (no 'indices' arg anymore)
    coords = feature.peak_local_max(
        dist_map,
        footprint=np.ones((3, 3)),
        labels=binary_mask
    )
    
    # 3. Build marker image from coordinates
    markers = np.zeros(dist_map.shape, dtype=int)
    for idx, (r, c) in enumerate(coords, start=1):
        markers[r, c] = idx

    # 4. Apply watershed on the negated distance map
    cell_labels = segmentation.watershed(-dist_map, markers, mask=binary_mask)
    num_cells = cell_labels.max()
    print(f"Segmented into {num_cells} individual cells using watershed.")
    return cell_labels

def extract_cell_spectra(image, cell_labels):
    """
    Compute the spectral intensity histogram for each cell.
    Returns a list of spectra (array of length 15) for each cell.
    """
    print(cell_labels)
    num_cells = cell_labels.max()
    spectra = []
    for cell_id in range(1, num_cells+1):
        # Create a mask for the current cell
        cell_mask = (cell_labels == cell_id)
        # If no pixels (should not happen for valid labels), skip
        if not cell_mask.any():
            spectra.append(np.zeros(image.shape[2]))
            continue
        # Extract all pixel intensities for this cell across all channels
        # and compute the mean intensity per channel
        cell_pixels = image[cell_mask]  # shape: (num_pixels_in_cell, 15)
        spectrum = cell_pixels.mean(axis=0)
        spectra.append(spectrum)
    return spectra


def classify_cells_by_spectrum(cell_spectra):
    """
    Determine the number of significant spectral peaks for each cell 
    and classify cells by peak count.
    Returns a list of peak counts and a dictionary grouping cell IDs by peak count.
    """
    peak_counts = []
    for idx, spectrum in enumerate(cell_spectra, start=1):
        # Find local maxima in the spectrum with a minimum prominence threshold&#8203;:contentReference[oaicite:15]{index=15}
        peaks, _ = signal.find_peaks(spectrum, prominence=0.2 * np.max(spectrum))
        peak_counts.append(len(peaks))
        print(f"Cell {idx}: {len(peaks)} significant peak(s) at channels {peaks}" if len(peaks)>0 
              else f"Cell {idx}: no significant peaks")
    # Group cells by peak count
    grouping = {}
    for cell_id, count in enumerate(peak_counts, start=1):
        grouping.setdefault(count, []).append(cell_id)
    return peak_counts, grouping

def annotate_image(image, roi_labels, cell_spectra, peak_counts):
    """
    Draw ROI bounding boxes and per-cell spectral histograms on a copy of the image.
    Returns the annotated color image.
    """
    # Create a display image (grayscale composite of all channels)
    base_gray = image.mean(axis=2)  # average projection for visualization
    # Normalize the grayscale image to [0, 255] for display
    disp_img = util.img_as_ubyte((base_gray - base_gray.min()) / (base_gray.max() - base_gray.min() + 1e-8))
    # Convert to BGR color for annotations
    disp_color = cv2.cvtColor(disp_img, cv2.COLOR_GRAY2BGR)
    # Draw bounding box for each ROI (connected component in roi_labels)
    props = measure.regionprops(roi_labels)
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        cv2.rectangle(disp_color, (minc, minr), (maxc, maxr), color=(0,0,255), thickness=2)  # red box&#8203;:contentReference[oaicite:18]{index=18}
    # Draw spectral histogram for each cell
    hist_height = 50  # fixed height for histogram plot
    bar_width = 3     # width of each channel bar
    spacing = 1       # spacing between bars
    num_channels = image.shape[2]
    props_cells = measure.regionprops(cell_labels)
    for i, prop in enumerate(props_cells, start=1):
        # Determine where to place the histogram (at top-left of the cell's bounding box by default)
        minr, minc, maxr, maxc = prop.bbox
        x0 = minc
        # If histogram would go beyond image width, adjust its x-position
        hist_width = num_channels * (bar_width + spacing)
        if x0 + hist_width > W:
            x0 = max(0, W - hist_width - 1)
        # Choose y-position above the cell if space, else below
        if minr - hist_height >= 0:
            y0 = minr - hist_height  # place histogram above the cell
        else:
            y0 = maxr  # place below if at top edge
        # Normalize the spectrum to the histogram height
        spectrum = cell_spectra[i-1]
        if spectrum.max() > 0:
            norm_spectrum = spectrum / spectrum.max()
        else:
            norm_spectrum = spectrum
        # Draw bars for each channel
        for c in range(num_channels):
            val = norm_spectrum[c]
            bar_h = int(val * hist_height)
            # Coordinates for the bar rectangle
            x_start = x0 + c * (bar_width + spacing)
            x_end = x_start + bar_width
            y_start = y0 + hist_height - bar_h  # top of bar
            y_end = y0 + hist_height  # bottom of bar (base)
            # Draw the rectangle (colored bar). Using yellow color for histogram bars.
            cv2.rectangle(disp_color, (x_start, y_start), (x_end, y_end), color=(0,255,255), thickness=-1)
        # (Optional) annotate peak count as text near the histogram
        peak_count = peak_counts[i-1]
        cv2.putText(disp_color, f"{peak_count} peaks", (x0, y0-5 if y0-5 >= 0 else y0+hist_height+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    return disp_color


image_path = "241121_10h40min_bra.h2b.mAp_twist.RFP_crbn.h2b.GFP_meis.kaede.czi"
image = load_hyperspectral_image(image_path)
H, W, C = image.shape
print(f"Loaded image of shape {image.shape}, with {C} spectral channels.")

binary_mask, roi_labels = detect_rois(image)
cell_labels = segment_cells(binary_mask)
cell_spectra = extract_cell_spectra(image, cell_labels)
peak_counts, cells_by_peaks = classify_cells_by_spectrum(cell_spectra)

# Print grouping summary
for peaks, cells in sorted(cells_by_peaks.items()):
    print(f"{len(cells)} cell(s) with {peaks} spectral peak(s): {cells}")
    
# Create annotated image
annotated_img = annotate_image(image, roi_labels, cell_spectra, peak_counts)
# Save the annotated image to disk
output_path = "annotated_output.png"
cv2.imwrite(output_path, annotated_img)
print(f"Annotated image saved as {output_path}")