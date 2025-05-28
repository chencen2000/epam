
import matplotlib.pyplot as plt
from cv2 import cvtColor, COLOR_BGR2RGB, COLOR_GRAY2BGR


def visualize_results(original, no_bg, dirt_diff, mask, title="Background Removal and Mask Creation"):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1)
    plt.imshow(cvtColor(original, COLOR_BGR2RGB))
    plt.title("Original Image"); plt.axis('off')
    plt.subplot(1, 4, 2)
    estimated_clean_background_bgr = cvtColor(no_bg, COLOR_GRAY2BGR)
    plt.imshow(cvtColor(estimated_clean_background_bgr, COLOR_BGR2RGB))
    plt.title("Estimated Clean Background"); plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.imshow(dirt_diff, cmap='gray', vmin=0, vmax=255)
    plt.title("Norm. Abs Diff (for Mask)"); plt.axis('off') # Clarified title
    plt.subplot(1, 4, 4)
    plt.imshow(mask, cmap='gray', vmin=0, vmax=255)
    plt.title("Binary Mask"); plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def visualize_dual_threshold_results(original, no_bg, dirt_diff, mask_low, mask_high, title="Dual Threshold Mask Creation"):
    """
    Visualize results with dual threshold masks
    """
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 5, 1)
    plt.imshow(cvtColor(original, COLOR_BGR2RGB))
    plt.title("Original Image"); plt.axis('off')
    
    plt.subplot(1, 5, 2)
    estimated_clean_background_bgr = cvtColor(no_bg, COLOR_GRAY2BGR)
    plt.imshow(cvtColor(estimated_clean_background_bgr, COLOR_BGR2RGB))
    plt.title("Estimated Clean Background"); plt.axis('off')
    
    plt.subplot(1, 5, 3)
    plt.imshow(dirt_diff, cmap='gray', vmin=0, vmax=255)
    plt.title("Norm. Abs Diff"); plt.axis('off')
    
    plt.subplot(1, 5, 4)
    plt.imshow(mask_low, cmap='gray', vmin=0, vmax=255)
    plt.title("Low Threshold Mask"); plt.axis('off')
    
    plt.subplot(1, 5, 5)
    plt.imshow(mask_high, cmap='gray', vmin=0, vmax=255)
    plt.title("High Threshold Mask"); plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

