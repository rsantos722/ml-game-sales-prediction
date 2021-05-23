from PySide6 import QtGui, QtWidgets, QtCore, QtUiTools
from PySide6.QtCore import QFile
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QFontDatabase
from qt_material import QtStyleTools, apply_stylesheet
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from cryptography.fernet import Fernet
import re
import os.path
from os import path

from sklearn.model_selection import train_test_split


class Application:
    def __init__(self):
        # Loading and styling
        self.app = QApplication(sys.argv)
        self.loader = QUiLoader()
        self.file = QFile("main_window.ui")
        self.file.open(QFile.ReadOnly)
        self.window = self.loader.load(self.file)
        self.file.close()
        self.window.show()
        apply_stylesheet(self.window, 'light_blue.xml', invert_secondary=True)

        # Setup functions
        self.window.centralwidget.findChild(QtWidgets.QPushButton, "generate_pushButton").clicked.connect(
            self.prediction_button_handler)
        self.window.centralwidget.findChild(QtWidgets.QPushButton, "ygsbreakdown_pushbutton").clicked.connect(
            self.ygsb_button_handler)
        self.window.centralwidget.findChild(QtWidgets.QPushButton, "gsby_pushbutton").clicked.connect(
            self.gsby_pushbutton_handler)
        self.window.centralwidget.findChild(QtWidgets.QPushButton, "transform_data_button").clicked.connect(
            self.transform_dataset)
        self.window.centralwidget.findChild(QtWidgets.QPushButton, "score_sales_analysis_button").clicked.connect(
            self.score_sales_analysis_handler)
        self.window.centralwidget.findChild(QtWidgets.QPushButton, "retrain_model_button").clicked.connect(
            self.retrain_ai_model)
        self.window.centralwidget.findChild(QtWidgets.QPushButton, "encrypt_button").clicked.connect(
            self.encrypt_csv)
        self.window.centralwidget.findChild(QtWidgets.QPushButton, "decrypt_button").clicked.connect(
            self.decrypt_csv)
        self.window.centralwidget.findChild(QLabel, 'encryption_text').hide()
        # Check Health Metrics
        if not path.exists("vg_dataset.csv"):
            self.window.centralwidget.findChild(QLabel, 'dataset_status_text').setText("Not Found")
        if not path.exists("filekey.key"):
            self.window.centralwidget.findChild(QLabel, 'encryption_key_status_text').setText("Not Found")
        # Load encryption key
        with open('filekey.key', 'rb') as fkey:
            loaded_key = fkey.read()
        self.fernet_crypt = Fernet(loaded_key)
        # Init ML Model
        self.vgdf_model = RandomForestRegressor()
        # Load model through pickle
        self.vgdf_model = pickle.load(
            open(r"vgdf_model.clf", "rb"))
        # Start
        self.app.exec_()

    def encrypt_csv(self):
        # Encrypt file and save into new file
        with open('vgdf_decrypted.csv', 'rb') as file:
            vgdf_decrypted = file.read()
        vgdf_encrypted = self.fernet_crypt.encrypt(vgdf_decrypted)
        with open('vgdf_encrypted.csv', 'wb') as vgdf_encrypted_file:
            vgdf_encrypted_file.write(vgdf_encrypted)
        self.window.centralwidget.findChild(QLabel, 'encryption_text').show()
        self.window.centralwidget.findChild(QLabel, 'encryption_text').setText("File Successfully Encrypted!")

    def decrypt_csv(self):
        # Open encrypted file and decrypt
        with open('vgdf_encrypted.csv', 'rb') as csv_encrypted:
            vgdf_encrypted = csv_encrypted.read()
        vgdf_decrypted = self.fernet_crypt.decrypt(vgdf_encrypted)
        with open('vgdf_decrypted.csv', 'wb') as csv_decrypted:
            csv_decrypted.write(vgdf_decrypted)
        self.window.centralwidget.findChild(QLabel, 'encryption_text').show()
        self.window.centralwidget.findChild(QLabel, 'encryption_text').setText("File Successfully Decrypted!")

    def retrain_ai_model(self):
        vgdf = pd.read_csv("vg_dataset_transformed.csv", index_col=0)
        # Table Set Up
        x = vgdf.drop("Global_Sales", axis=1)
        y = vgdf["Global_Sales"]
        # Model Definition
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        self.vgdf_model = RandomForestRegressor().fit(x_train, y_train)
        # Update model score
        score = self.vgdf_model.score(x_train, y_train)
        score = score * 100
        score = str(score)[:6] + "%"
        self.window.centralwidget.findChild(QLabel, 'model_accuracy_text').setText(score)
        self.window.centralwidget.findChild(QLabel, 'health_ai_accuracy_text').setText(score)

    def score_sales_analysis_handler(self):
        vgdf = pd.read_csv("vgdf_no_ka.csv")
        vgdf_reduced = vgdf[["Global_Sales", "Critic_Score", "User_Score"]].copy()
        vgdf_reduced.dropna(inplace=True)
        vgdf_reduced.reset_index(inplace=True, drop=True)
        col = vgdf_reduced.loc[:, "Critic_Score":"User_Score"]
        vgdf_reduced["score_avg"] = col.mean(axis=1)
        scatter_plot = vgdf_reduced.plot.scatter(x="score_avg", y="Global_Sales", figsize=(10, 7),
                                                 xlabel="Review Score", ylabel="Global Sales (millions of units)")
        fig = scatter_plot.get_figure()
        fig.savefig("scatter_plot_fig.png")
        # Display image
        pix = QtGui.QPixmap('scatter_plot_fig.png')
        self.window.centralwidget.findChild(QLabel, 'chart_label').setPixmap(pix)

    @staticmethod
    def transform_dataset():
        # Read file
        vgdf = pd.read_csv("vg_dataset.csv")
        # Reduce columns, drop NA rows, reset index
        vgdf_reduced = vgdf[["Year_of_Release", "Genre", "Global_Sales", "Critic_Score", "User_Score", "Rating"]].copy()
        vgdf_reduced.dropna(inplace=True)
        # Drop rows with "tbd" in user score
        vgdf_us_reduced = vgdf_reduced[vgdf_reduced["User_Score"] != "tbd"]
        vgdf_us_reduced.reset_index(inplace=True, drop=True)
        # Encode categorical columns
        vgdf_encoded = pd.get_dummies(vgdf_us_reduced, columns=["Genre"], prefix=["Genre_"])
        vgdf_encoded_2 = pd.get_dummies(vgdf_encoded, columns=["Rating"], prefix=["Rating_"])
        vgdf_encoded_2.drop(axis=1,labels="Rating__K-A",inplace=True)
        vgdf_encoded_2.reset_index(inplace=True, drop=True)
        # Save to new CSV file
        vgdf_encoded_2.to_csv("vg_dataset_transformed.csv")

    def prediction_button_handler(self):
        # Get data
        prediction_year = self.window.centralwidget.findChild(QtWidgets.QSpinBox, "year_spinBox").value()
        prediction_userscore = self.window.centralwidget.findChild(QtWidgets.QSpinBox, "user_spinBox").value()
        prediction_criticscore = self.window.centralwidget.findChild(QtWidgets.QSpinBox, "critic_spinBox").value()
        prediction_genre = self.window.centralwidget.findChild(QtWidgets.QComboBox, "genre_comboBox").currentText()
        prediction_agerating = self.window.centralwidget.findChild(QtWidgets.QComboBox,
                                                                   "agerating_comboBox").currentText()
        # Dict object with array index
        genre_index = {"Action": 3, "Adventure": 4, "Fighting": 5, "Misc": 6, "Platform": 7, "Puzzle": 8, "Racing": 9,
                       "Role-Playing": 10, "Shooter": 11, "Simulation": 12, "Sports": 13, "Strategy": 14}
        rating_index = {"AO": 15, "E": 16, "E10+": 17, "M": 18, "RP": 19, "T": 20}

        # Year,Critic,User,Action,Adventure,Fighting,Misc,Platform,Puzzle,Racing,Role-Playing,Shooter,Simulation,Sports,Strategy,AO,E,E10+,M,RP,T

        # Create the array
        # Create initial array of zeros with numpy, then add various things
        prediction_array_np = np.zeros(21)
        prediction_array_np[0] = prediction_year
        prediction_array_np[1] = prediction_criticscore
        prediction_array_np[2] = prediction_userscore
        prediction_array_np[genre_index.get(prediction_genre)] = 1
        prediction_array_np[rating_index.get(prediction_agerating)] = 1

        # Predict
        sales_prediction = self.vgdf_model.predict([prediction_array_np])
        # Convert to string, remove square brackets from string
        sales_prediction_string = np.array2string(sales_prediction)
        sps_slice_first = sales_prediction_string[1:]
        sps_slice_last = sps_slice_first[:-1]
        print(sps_slice_last)
        # Display prediction on GUI in prepared string
        display_string = "Predicted Sales: " + sps_slice_last + " Million Units"
        self.window.centralwidget.findChild(QLabel, 'prediction_label').setText(display_string)

    # Handles Game Sales by Year Chart
    def gsby_pushbutton_handler(self):
        # Load csv
        vgdf_dataset = pd.read_csv("vgdf_dropna.csv")
        # Get Selected Genre
        selected_genre = self.window.centralwidget.findChild(QtWidgets.QComboBox, "gsby_combobox").currentText()
        plot_data = vgdf_dataset.loc[vgdf_dataset['Genre'] == selected_genre].groupby("Year_of_Release")[
            "Global_Sales"].sum().to_frame()
        # Plot, save to image
        plot = plot_data.plot.line(y="Global_Sales", xlabel="Year of Release", legend=False,
                                   ylabel="Sales (millions of units)", figsize=(10, 7))
        fig = plot.get_figure()
        fig.savefig("genre_sales_line_fig.png")
        # Display image
        pix = QtGui.QPixmap('genre_sales_line_fig.png')
        self.window.centralwidget.findChild(QLabel, 'chart_label').setPixmap(pix)

    # Handles yearly game genre breakdown bar chart
    def ygsb_button_handler(self):
        # Load csv
        vgdf_dataset = pd.read_csv("vgdf_dropna.csv")
        selected_year = \
            int(self.window.centralwidget.findChild(QtWidgets.QComboBox, "ygsbreakdown_combobox").currentText())
        plot_data = \
            vgdf_dataset.loc[vgdf_dataset['Year_of_Release'] == selected_year].groupby("Genre")[
                "Global_Sales"].sum().to_frame()
        plot_data["Percent"] = (plot_data["Global_Sales"] / plot_data["Global_Sales"].sum() * 100)

        # Plot, save to image
        plot = plot_data.plot.pie(y="Percent", figsize=(10, 7), ylabel="", legend=False)
        fig = plot.get_figure()
        fig.savefig("year_genre_breakdown_fig.png")
        # Display image
        pix = QtGui.QPixmap('year_genre_breakdown_fig.png')
        self.window.centralwidget.findChild(QLabel, 'chart_label').setPixmap(pix)


if __name__ == "__main__":
    Application()
    sys.exit()
