# Indian-soldier-uniform-classification

**To run the inference file** --
1. load the all pretrained model from the given [link](https://drive.google.com/drive/folders/1sh1kmIl-y6sysgnzuHboYLFrWZegm5Dz?usp=sharing)
2. Store into models folder
3. Add path of them in inference file.
4. Run the below given command
 
   ``
python inference.py --image_path "C:\User\Your\image\path.jpeg"
   ``

**To run the code with GUI**
1. Give the all 3 paths of the model in app.py file
2. Run below given command in terminal

   ``
   python app.py
   ``

**To run tflite(Tensorflow lite version edge based case)**
1. Download pre-trained tflite converted models from this [link](https://drive.google.com/drive/folders/1sh1kmIl-y6sysgnzuHboYLFrWZegm5Dz?usp=sharing)
2. Give the path in tflite.py file
3. Run below given command
   
``
python tflite.py --image_path "C:\User\Your\image\path.jpeg"
``  

   
**To train the model from scratch**
1. Put the dataset in dataset folder and add path of dataset in `train.py` file.
2. And, run the following command.

``
python train.py
``
