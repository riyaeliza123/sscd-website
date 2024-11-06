# -------
# Author: Riya Eliza Shaju
# Contractor, Pacific Salmon Foundation - Vancouver, British Columbia
# -------

import streamlit as st
import os
from utils import main
from PIL import Image

st.title('Salmon Scale Circuli Detector App')

col1, col2 = st.columns(2)

with col1:
    uploaded_files = st.file_uploader("Upload Image", 
                                        type=["jpg", "jpeg", "png", "tif"])
                                        # accept_multiple_files=True)

    output_dir = st.text_input("Enter the output directory", value="C:/Users/hp/Desktop/SSCD_results")
    transect_angles = st.multiselect("Select Transect Angles", [0, 45, 90, 135, 180], default=[0, 45, 90, 135, 180])
    plot_dets = st.checkbox("Plot Detections", value=True)
    dets_separate_files = st.checkbox("Save Detections in Separate Files", value=True)
    transect_max_boxes = st.number_input("Enter the maximum number of detections per transect image", value=200)

    if st.button("Process Images"):
        if uploaded_files:
            # Create temporary directory to save uploaded images
            img_dir = './riya_drafts/temp_uploads'
            os.makedirs(img_dir, exist_ok=True)
            #for uploaded_file in uploaded_files:
            with open(os.path.join(img_dir, uploaded_files.name), "wb") as f:
                f.write(uploaded_files.getbuffer())

            # Call the processing function
            main(
                img_dir=img_dir,
                output_dir=output_dir,
                transect_angles=transect_angles,
                plot_dets=plot_dets,
                dets_separate_files=dets_separate_files,
                transect_max_boxes = transect_max_boxes
            )

        st.success("Image processing completed. Result has been saved to output directory.")


with col2:

    # Print input image
    if uploaded_files is not None:
        image = Image.open(uploaded_files)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    else:
        st.write("Please upload an image file.")

    # # Print output
    # op_name = os.path.splitext(uploaded_files.name)[0]
    # op_img_path = f"{output_dir}/detections/focus/detection_images/{op_name}detections.jpg"
    # output_image = Image.open(op_img_path)
    # st.image(output_image, caption="Result", use_column_width=True)

