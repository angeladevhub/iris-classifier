# Iris classifier

This project trains and evaluates a machine learning model using Decision Tree Classifier on the classic Iris dataset.

- Jupyter Notebook contains experiments, exploration, and testing of different models, located in notebook folder.
- The Python scripts focus on the final model using the Decision Tree classifier, located in src folder.
- The confusion matrix plot and the saved model stored in the outputs folder. 

# project structure
- ├── notebooks/
-   └── iris_model.ipynb # Jupyter notebook (experiments, exploration)
- ├── src/
-   └── train.py # Train model, save confusion matrix + model file
-   └── test.py # Load model and evaluate on test set
- ├── outputs/
-  ├── confusion_matrix.png # Confusion matrix plot
-   └── decision_tree_model.pkl# Saved model
- ├── requirements.txt # Python dependencies
- └── README.md # Project documentation

# setup instructions
- download the folder
- create environment:
run : ```bash 
py -m venv venv 

Linux/macOS: source venv/bin/activate
Windows: venv\Scripts\activate

# Install dependencies by running:
pip install -r requirements.txt

# train the model by running in terminal:
py src/train.py

# test the model by running in terminal:
py src/test.py