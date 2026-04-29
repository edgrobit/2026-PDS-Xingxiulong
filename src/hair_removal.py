import cv2
import numpy as np
import os
import glob

# ==========================================
# PART 1: ALGORITHM FUNCTIONS
# ==========================================

def get_valid_area_mask(img_gray):
    """
    Creates a mask of the actual skin/lesion area, ignoring the black 
    vignette corners often found in dermoscopy images.
    """
    # Threshold out very dark pixels (e.g., borders)
    _, mask = cv2.threshold(img_gray, 15, 255, cv2.THRESH_BINARY)
    return mask

def calculate_hair_coverage(img_bgr):
    """
    Extracts hair coverage using CLAHE, Sobel, and Laplacian.
    Returns the coverage ratio and the binary hair mask.
    """
    # The green channel offers the best contrast for skin lesions
    base_channel = img_bgr[:, :, 1]
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Get the valid image area to ensure accurate coverage calculation
    valid_mask = get_valid_area_mask(img_gray)
    
    # 1. Boost visibility using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_base = clahe.apply(base_channel)
    
    # 2. Edge Detection (Sobel for thick hairs, Laplacian for thin)
    sobel_x = cv2.Sobel(enhanced_base, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(enhanced_base, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)
    sobel_norm = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    laplacian = cv2.Laplacian(enhanced_base, cv2.CV_64F)
    laplacian_abs = np.absolute(laplacian)
    laplacian_norm = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    averaged_edges = cv2.addWeighted(sobel_norm, 0.5, laplacian_norm, 0.5, 0)
    
    # 3. Smooth and Threshold to get binary mask
    blurred_edges = cv2.GaussianBlur(averaged_edges, (3, 3), 0)
    _, binary_hair_mask = cv2.threshold(blurred_edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Calculate Coverage
    hair_in_valid_area = cv2.bitwise_and(binary_hair_mask, valid_mask)
    hair_pixels = cv2.countNonZero(hair_in_valid_area)
    valid_pixels = cv2.countNonZero(valid_mask)
    
    coverage = 0.0 if valid_pixels == 0 else hair_pixels / valid_pixels
    
    return coverage

def process_image(img_bgr):
    """
    Determines if hair removal is needed, adjusts parameters dynamically, 
    and returns the processed image.
    """
    coverage = calculate_hair_coverage(img_bgr)
    
    # Skip processing if hair coverage is insignificant
    if coverage < 0.005:
        return img_bgr, coverage, False # False means no processing was applied
        
    # Adjust kernel size based on hair density
    k_size = 15 if coverage <= 0.035 else 25
    
    # Use MORPH_CROSS as recommended for thin dark structures
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (k_size, k_size))
    base_channel = img_bgr[:, :, 1]
    
    # Combine BlackHat (dark hair) and TopHat (light hair)
    blackhat = cv2.morphologyEx(base_channel, cv2.MORPH_BLACKHAT, kernel)
    tophat = cv2.morphologyEx(base_channel, cv2.MORPH_TOPHAT, kernel)
    combined_morph = cv2.max(blackhat, tophat)
    
    # Adaptive Thresholding for non-uniform lighting
    block_size = (k_size * 2) + 1 
    removal_mask = cv2.adaptiveThreshold(
        combined_morph, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, -5
    )
    
    # Morphological closing to reduce artifacts
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    removal_mask = cv2.morphologyEx(removal_mask, cv2.MORPH_CLOSE, closing_kernel)
    
    # Inpaint the original image
    cleaned_img = cv2.inpaint(img_bgr, removal_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return cleaned_img, coverage, True

# ==========================================
# PART 2: BATCH PROCESSING PIPELINE
# ==========================================

def run_dataset_pipeline(input_dir, output_dir):
    search_path = os.path.join(input_dir, "*.png")
    image_files = glob.glob(search_path)
    
    if not image_files:
        print(f"No PNG images found in '{input_dir}'. Please check the path.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    total_images = len(image_files)
    print(f"Found {total_images} images. Starting batch processing...\n")
    
    processed_count = 0
    skipped_count = 0
    
    for i, filepath in enumerate(image_files, 1):
        filename = os.path.basename(filepath)
        
        img = cv2.imread(filepath)
        if img is None:
            print(f"[{i}/{total_images}] Error reading: {filename}")
            continue
            
        # Run the logic
        final_img, coverage_val, was_processed = process_image(img)
        
        # Determine output string for logging
        if was_processed:
            status = "Cleaned"
            processed_count += 1
        else:
            status = "Skipped (No hair)"
            skipped_count += 1
            
        # Save output (using original filename so it maps perfectly back to your labels)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, final_img)
        
        # Print progress every 100 images to avoid spamming the console
        if i % 100 == 0 or i == total_images:
            print(f"[{i}/{total_images}] Processed {filename} | Coverage: {coverage_val:.4f} | Status: {status}")
            
    print("\n" + "="*40)
    print("PIPELINE COMPLETE")
    print(f"Total processed: {total_images}")
    print(f"Images cleaned:  {processed_count}")
    print(f"Images skipped:  {skipped_count}")
    print(f"Output saved to: '{output_dir}'")
    print("="*40)

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # Define your paths here
    INPUT_FOLDER = "imgs"          # <-- Change to your input folder name
    OUTPUT_FOLDER = "dataset_hairless"    # <-- Change to your desired output folder name
    
    run_dataset_pipeline(INPUT_FOLDER, OUTPUT_FOLDER)