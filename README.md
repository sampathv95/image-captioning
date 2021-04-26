# Image caption generator
This is a neural network model trained on Flickr8k and MS-COCO datasets to automatically give suitable caption to an image. Check my medium story (https://medium.com/swlh/image-captioning-using-multimodal-neural-networks-ec274cfceb93) for elaborate explanations.

The Flicker8k and MS-COCO datasets used for this project can be downloaded from the links below

link for flicker8k images : https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip

link for flicker8k captions: https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip

link for MS-COCO dataset: https://cocodataset.org/#download

the zip files downloaded must be extracted and placed in the same directory where your notebooks are present (source directory)

order of execution of notebooks:
1. data_exploration.ipynb
2. glove_embeddings.ipynb
3. caption_generator.ipynb

### Dataset Description ###
For this project, we use the famous FLickr 8k dataset. This dataset contains 8000 images (hence the name 8k) and each image has 5 captions telling us what is happening in the image. These 8000 images are split as follows:

a)6000 Training images and their descriptions.

b)1000 Development/Validation images and their descriptions.

c)1000 Test images and their descriptions
