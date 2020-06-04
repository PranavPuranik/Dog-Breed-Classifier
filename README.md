## Project Overview

A simple CNN classifier to differentiate dog breeds, uses InceptionV3 pretrained weights

![Sample Output](images/sample_dog_output.png)

The dataset contains 8351 dog images of 133 breeds. The problem statements requires use to design an algorithm that predicts whether an image contains a human or dog, and predict it's breed (even if it's a human!). 


### Table of Contents

1. [Requirements](#reqirements)
2. [Installation](#installation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Acknowledgements](#ack)

### Requirements <a name="reqirements"></a>
Python3 and the following libraries: 
* opencv-python
* h5py
* matplotlib
* numpy
* scipy
* tqdm
* scikit-learn
* keras
* tensorflow
* jupyterlabs

### Installation <a name="installation"></a>

1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/udacity/dog-project.git
cd dog-project
```

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`. 

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

4. Donwload the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`.

5. (Optional) __If you plan to install TensorFlow with GPU support on your local machine__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step.

6. (Optional) **If you are running the project on your local machine (and not using AWS)**, create (and activate) a new environment.

	- __Linux__ (to install with __GPU support__, change `requirements/dog-linux.yml` to `requirements/dog-linux-gpu.yml`): 
	```
	conda env create -f requirements/dog-linux.yml
	source activate dog-project
	```  
	- __Mac__ (to install with __GPU support__, change `requirements/dog-mac.yml` to `requirements/dog-mac-gpu.yml`): 
	```
	conda env create -f requirements/dog-mac.yml
	source activate dog-project
	```  
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/dog-windows.yml` to `requirements/dog-windows-gpu.yml`):  
	```
	conda env create -f requirements/dog-windows.yml
	activate dog-project
	```

7. (Optional) **If you are running the project on your local machine (and not using AWS)** and Step 6 throws errors, try this __alternative__ step to create your environment.

	- __Linux__ or __Mac__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`): 
	```
	conda create --name dog-project python=3.5
	source activate dog-project
	pip install -r requirements/requirements.txt
	```
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):  
	```
	conda create --name dog-project python=3.5
	activate dog-project
	pip install -r requirements/requirements.txt
	```
	
8. (Optional) **If you are using AWS**, install Tensorflow.
```
sudo python3 -m pip install -r requirements/requirements-gpu.txt
```
	
9. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
		```
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
		```
	- __Windows__: 
		```
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
		```

10. (Optional) **If you are running the project on your local machine (and not using AWS)**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog-project` environment. 
```
python -m ipykernel install --user --name dog-project --display-name "dog-project"
```

11. Open the notebook.
```
jupyter notebook dog_app.ipynb
```


### File Descriptions <a name="files"></a>

Some important files used in the project include:
1. haarcascades/haarcascade_frontalface_alt.xml: Viola-Jones face detector provided by OpenCV.
2. saved_models/weights.best.InceptionV3.hdf5: Inception v3 model trained using transfer learning.
3. dog_app.ipynb: a notebook that explains the classifier code.
4. extract_bottleneck_features.py: functions to compute bottleneck features of image
5. images/: contains few random images

### Results <a name="results"></a>

1. saved_models/weights.best.from_scratch.hdf5: 2.2727%
2. saved_models/weights.best.VGG16.hdf5: 40.7895%
3. saved_models/weights.best.InceptionV3.hdf5: 83.4928%

Medium article link : https://medium.com/@pranavpuranik10/dog-breed-classifier-udacity-data-science-nano-degree-program-50141e92dea2

### Acknowledgements <a name="ack"></a>
Thanks to Udacity's Data Science Nano Degree Program team!
