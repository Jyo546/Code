Title: Classification of Rumor Labels using Machine Learning Models

Description: This repository contains code for training and evaluating various machine learning models to predict rumor labels based on text data. The models are trained on a dataset (newdataset.csv) containing text samples and corresponding rumor labels.

Files: newdataset.csv: Dataset containing sourceid,source,rumor labels and text. coapps_x.ipynb: Jupyter Notebook containing the Python code for data preprocessing, model training, evaluation, and saving the trained model as a pickle file. rumor_classifier.pkl: Pickle file containing the trained best_model. README.md: This file. Usage: Data Exploration and Preprocessing:

The newdataset.csv file contains the dataset. Load the dataset and explore its structure using pandas. Preprocess the text data (e.g., removing stopwords, tokenization, vectorization). Model Training and Evaluation:

Train multiple classification models (LinearSVC, Random Forest, KNN, Decision Tree) on the preprocessed text data. Evaluate each model's performance using accuracy and classification report. Model Serialization:

Save the trained best_model as a pickle file (rumor_classifier.pkl) using the pickle module. Model Deployment:

Load the serialized model from rumor_classifier.pkl into rumor_classification_app.py and run streamlit run rumor_classification_app.py in command prompt to make classification of rumors. Requirements: Python 3.x Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn, streamlit Instructions: Install the required libraries: pip install -r requirements.txt Execute the Jupyter Notebook rumor_label_prediction.ipynb to train models and save the LinearSVC model. Use the saved model (rumor_classifier.pkl) for prediction by providing input text.

Contributors: 1.Pallaka chakradhar 2.Paidi Ramyasri 3.Kakunuri Naga jyothi

Contact: For any inquiries or suggestions, feel free to contact kakunurinagajyoth@gmail.com or open an issue in the repository.
