# COMP646 Project: Image Collages

## Remove Anything
1. cd COMP646_Project/remove_anything
2. python remove_anything.py \
    --input_img ./remove-anything/dog.jpg \
    --coords_type key_in \
    --point_coords 200 450 \
    --point_labels 1 \
    --dilate_kernel_size 15 \
    --output_dir ./results \
    --sam_model_type "vit_h" \
    --sam_ckpt {YOUR PRETRAINED MODEL PATH}/pretrained_models/sam_vit_h_4b8939.pth \
    --lama_config ./lama/configs/prediction/default.yaml \
    --lama_ckpt {YOUR PRETRAINED MODEL PATH}/pretrained_models/big-lama
3. Directly download pretrained_models from this link: https://drive.google.com/drive/folders/1ST0aRbDRZGli0r7OVVOQvXwtadMCuWXg
pip install gdown
gdown https://drive.google.com/drive/folders/1wpY-upCo4GIW4wVPnlMh_ym779lLIG2A-O {removing_anything folder}/pretrained_models--folder




## Main components and technologies

* Dataset and Segment Pool
  * dataset: pascal voc 2012 for segmentation
  * segment pool: extract segments from the dataset and get a segment pool
* Core Models(Modules) for Image Collages
  * Gounded SAM & SAM: segments input image from users (with/without prompts) and get the object the users what
  * LaMa: remove the original object in the image and inpaint the original image
  * CLIP: retrieve segments that are most relevant to users' prompt from our segment pool
  * Our own algorithm:
    * replace the original object with the most suitable object from the retrieved segments
    * adjust the position/pose of our segment to make the image better and reasonable
  
  **Pay attention** Inpant-anything (remove-anything) is the combination of SAM and LaMa, we can use it
* Front End
  * Gradio: implement the use case
* Other Optimizations/New Features if possible



## Use case

We have a user-friendly GUI. The general use case is 
1. user inputs an image
2. the GUI shows the input image
3. user inputs prompt (clicks the image or uses text prompt) to select an object in the image
4. user inputs a text prompt then our system replaces the selected object with the most relevant object retrieved from the segment pool
5. user can repeat step 3-4 to modify the input image 

[maybe there is a figure to show use case]


# Plan for Progress Report

* Introduce
* (Our main components and use case)
* Dataset
* Core Module (our test on the core models, if not, just copy images from others)
* Initial Result (if possible)

============ Before ======================
* Validate idea (use Pascal as segments pool)
* Match images' seams (SURF...)
* Output image
