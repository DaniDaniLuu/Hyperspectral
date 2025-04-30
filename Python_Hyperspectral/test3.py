from bioio import BioImage
import numpy as np
from skimage import filters, morphology
import matplotlib.pyplot as plt
from skimage import measure
from collections import defaultdict, Counter

# instantiate; this returns a 5D TCZYX numpy array when you access .data
img = BioImage("/Users/dani/repos/Hyperspectral/Python_Hyperspectral/241121_10h40min_bra.h2b.mAp_twist.RFP_crbn.h2b.GFP_meis.kaede.czi")  
# img.data.shape == (T, C, Z, Y, X)
print("dims order:", img.dims.order)     # e.g. "TCZYX"
print("shape:   ", img.data.shape)  

# if you only have one timepoint and one Z-plane, drop those dims
data = img.get_image_data("CZYX", T=0, Z=0)
# now data.shape == (15, Y, X)

nuc = data[0]               # channel 0 → 2D (Y, X)
th = filters.threshold_otsu(nuc)
mask = nuc > th             # boolean mask of “cell pixels”
mask = morphology.remove_small_objects(mask, min_size=500)

# coords is a (2, N) array of y,x positions
zs, ys, xs = np.nonzero(mask)

plt.figure(figsize=(8,5))
for c in range(data.shape[0]):      # 0…14
    vals = data[c, zs, ys, xs]          # all pixel intensities in channel c
    plt.hist(vals, bins=50, alpha=0.4, label=f"Ch {c+1}")
plt.xlabel("Intensity")
plt.ylabel("Count")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

labels = measure.label(mask)
regions = measure.regionprops(labels)
# regions[i].coords is an (M_i, 2) array of pixel coords for cell i

thresholds = [
    np.percentile(data[c, ys, xs], 99) 
    for c in range(data.shape[0])
]


cell_channel_counts = {}
channel_to_cells = defaultdict(list)

for region in regions:
    yx = tuple(region.coords.T)   # coords as two 1D arrays
    means = [data[c][yx].mean() for c in range(data.shape[0])]
    expressed = [c for c, m in enumerate(means) if m > thresholds[c]]
    n_expr = len(expressed)
    cell_channel_counts[region.label] = n_expr
    for c in expressed:
        channel_to_cells[c].append(region.label)
        
count_dist = Counter(cell_channel_counts.values())
# e.g. count_dist[3] = number of cells expressing exactly 3 channels