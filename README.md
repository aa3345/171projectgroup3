# ECS 171 Project - Group 3
## Front-End
To see the front end model visit the About section in the repo or type in tinyurl.com/171group3 which will lead to a website that is hosted through streamlit with a form that asks a couple of questions.
[Here is the demo link for instructions on how to use the front-end.](https://youtu.be/0xFOsGqz2eg?si=MrZ2UDme3uRuvlOR)

## About the Project
This repository hosts code used to generate models for ECS171 Winter Quarter. Our models are built using the obesity.csv file, which was taken from Estimation of Obesity Levels Based On Eating Habits and Physical Condition by Fabio Mendoza Palechor and Alexis De la Manotas. ([link to original dataset on UCI Machine Learning Archive](https://doi.org/10.24432/C5H31Z))

Additionally, you will need the following dependencies:
- streamlit
- pandas
- numpy
- xgboost==2.0.3
- scikit-learn

### Cloning the Repository
If you would like to clone this repository, you can use the following command to clone it: `git clone https://github.com/aa3345/171projectgroup3`
After, navigate to the cloned repository in a terminal and run: `python3 -m pip install requirements.txt` to install all needed dependencies.

### Downloading Individual Models
If you haven not cloned this repository in it's entirety, you can still run the files assuming you have the dataset in the working directory and have installed the required dependencies using `python3 -m pip install dependency-name`. Make sure you replace "dependecy-name" with the actual name of the dependency listed.


## Files
DNN.ipynb - Notebook for the Deep Neural Network model being trained <br>
EDA.ipynb - Exploratory Data Analysis is being run on the dataset<br>
MLP.ipynb - Notebook for Multilayer Perceptron training<br>
XGB.ipynb - Notebook for XGB model training<br>
dnn_visualization.png - visualization of the deep neural net structure we used<br>
frontend.py - python file that uses streamlit to create a form to connect to the front end<br>
obesity.csv - dataset<br>
requirements.txt - requirements for the frontend to run<br>
xgb_model.sav - saved file used in the frontend process<br>
