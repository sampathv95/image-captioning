# Image caption generator
This is a neural network model trained on Flickr8k dataset to automatically give suitable caption to an image.

The Flicker8k dataset used for this project can be downloaded from the tow links below
link for images : https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
link for captions: https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip

the zip files downloaded must be etracted and placed in the same directory where your notebooks are present (source directory)

order of execution of notebooks:
1. data_exploration.ipynb
2. glove_embeddings.ipynb
3. caption_generator.ipynb

### Dataset Description ###
For this project, we use the famous FLickr 8k dataset. This dataset contains 8000 images (hence the name 8k) and each image has 5 captions telling us what is happening in the image. These 8000 images are split as follows:

1.6000 Training images and their descriptions.

2.1000 Development/Validation images and their descriptions.

3.1000 Test images and their descriptions
