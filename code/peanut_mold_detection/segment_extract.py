import sys
import os
import csv
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

args = len(sys.argv)
batch_type = 'nc'
seg_type = 'watershed'

if args > 3:
    sys.exit("ERROR: Too many line arguments. There are only two optional arguments: <batch_type> <seg_type>.")
elif args >= 2:
    batch_type = sys.argv[1]    # label of batch ['nc' = non-contamnated (default) / 'c' = contaminated]
                                # segmentation type
    if batch_type not in ['nc', 'c']:
        sys.exit("ERROR: batch_type only accepts 'nc' or 'c' as inputs.")
if args == 3:
    seg_type = sys.argv[2]      # ['watershed' = marker-based watershed segmentation (default) / 
                                # 'colseg' = color segmentation]
    if seg_type not in ['watershed', 'colseg']:
        sys.exit("ERROR: batch_type only accepts 'watershed' or 'colseg' as inputs.")

if batch_type == 'nc':
    sample_type = 0
    INPUT_PATH = "samples\\Non-Contaminated\\Test_batch"  # input of image batch folder
    OUTPUT_PATH = "samples\\Non-Contaminated\\Output"       # output of individual peanuts
else:
    sample_type = 1
    INPUT_PATH = "samples\\Contaminated\\Test_batch"  # input of image batch folder
    OUTPUT_PATH = "samples\\Contaminated\\Output"       # output of individual peanuts

CSV_PATH = "datasets\\pmd_dataset_1.csv"         # output feature set


for img_path in os.listdir(INPUT_PATH): 
    print("\n\nExamining image sample: {}".format(img_path))

    img = cv2.imread(INPUT_PATH+"\\"+img_path)

    # color segmentation (base)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # thresholding
    low_peanut = (0,0,0)
    high_peanut = (105, 255, 255)

    mask = cv2.inRange(hsv_img, low_peanut, high_peanut)
    result = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)


    color_segmented = cv2.cvtColor(result, cv2.COLOR_HSV2RGB)
    gray = cv2.cvtColor(color_segmented, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)

    # noise removal
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 10)
    # closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel, iterations = 1)
    
    if seg_type == 'colseg':
        final_proc = opening
    else: # marker-based watershed segmentation
        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=2)

        # Finding sure foreground threshold
        dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,5)

        # sure foreground area
        ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)


        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1

        # Apply watershed
        markers[unknown==255] = 0

        segmented = cv2.watershed(img_rgb,markers)
        img[markers == -1] = [255,0,0]

        segmented_bin = segmented.copy()
        segmented_bin[segmented < 2] = 0 # -1 is dividing regions, no 0s, 1 is background
        segmented_bin[segmented > 1] = 255 # all above 1 are distinct regions

        final_proc = segmented_bin.astype('uint8')

    # Component Analysis
    connectivity = 8
    numComponents, labels, stats, centroids = cv2.connectedComponentsWithStats(final_proc, connectivity, cv2.CV_32S)
    print("Number of objects: ", numComponents-1) # labels[0] is just the background

    temp_img = img_rgb.copy()
    temp_img[labels == 0] = 0   # ignore other labeled components

    # Create CSV file
    with open(CSV_PATH, 'a', newline='') as fh:
        csv_writer = csv.writer(fh)
        
        # Create header
        if os.path.exists(CSV_PATH) and os.stat(CSV_PATH).st_size == 0:
            csv_header = ['sample_id', 'area', 'red', 'green', 'blue', 'hue', 'sat', 'value', 'gray', 'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'asm', 'class']
            csv_writer.writerow(csv_header)

        # Feature Extraction
        for i in range(1, numComponents):
            print("Examining object no. {}/{} ".format(i, numComponents-1))

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]

            # Isolate sample
            temp_img = img_rgb.copy()
            temp_img[labels != i] = 0   # ignore other labeled components
            obj_rect = temp_img[y:y+h, x:x+w] # component rectangular frame

            # File write
            img_name = OUTPUT_PATH+"\\"+"{}_{:03d}.jpg".format(img_path[:-4], i)
            cv2.imwrite(img_name, cv2.cvtColor(obj_rect, cv2.COLOR_RGB2BGR))
            
            print("Output created: {}".format(img_name))

            # Compute Mean RGB values
            obj_rect_T = np.transpose(obj_rect)

            r_mean = np.sum(obj_rect_T[0]) / area
            g_mean = np.sum(obj_rect_T[1]) / area
            b_mean = np.sum(obj_rect_T[2]) / area

            # Compute Mean HSV values
            obj_hsv = cv2.cvtColor(obj_rect, cv2.COLOR_RGB2HSV)
            obj_hsv_T = np.transpose(obj_hsv)

            h_mean = np.sum(obj_hsv_T[0]) / area
            s_mean = np.sum(obj_hsv_T[1]) / area
            v_mean = np.sum(obj_hsv_T[2]) / area

            # Compute Mean Gray values
            obj_gray = cv2.cvtColor(obj_rect, cv2.COLOR_RGB2GRAY)
            gray_mean = np.sum(obj_gray) / area

            # GLCM Features Extraction
            glcm = graycomatrix(obj_gray, distances=[5], angles=[np.pi/2], levels=256, symmetric=True, normed=True)

            contrast = graycoprops(glcm, prop='contrast')
            dissimilarity = graycoprops(glcm, prop='dissimilarity')
            homogeneity = graycoprops(glcm, prop='homogeneity')
            energy = graycoprops(glcm, prop='energy')
            correlation = graycoprops(glcm, prop='correlation')
            asm = graycoprops(glcm, prop='ASM')

            

            # Write Data to CSV
            sample_data = [img_name, area, r_mean, g_mean, b_mean, h_mean, s_mean, v_mean, gray_mean, contrast[0,0], dissimilarity[0,0], homogeneity[0,0], energy[0,0], correlation[0,0], asm[0,0], sample_type]
            csv_writer.writerow(sample_data)

print("Dataset CSV file is written to ", CSV_PATH, "\n")
