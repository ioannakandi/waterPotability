# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 12:26:47 2022

@author: Ioanna Kandi & Kostis Mavrogiorgos 
"""
#import libraries 
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from flask import Flask, request, Response
from flask_cors import CORS, cross_origin

# App config.
app = Flask(__name__, static_url_path='',
            static_folder='templates\classimax-premium',
            template_folder='templates\classimax-premium')
DEBUG = True
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
CORS(app)

#those are dummy parameters to perform tests ph = 1 hardness = 1 solids = 1 chloramines= 1 
#sulfate = 1 conductivity = 1 organic_carbon = 1 trihalomethanes = 1 turbidity = 1 

# endpoint for predicting the water potability
@app.route("/predictPotability",methods=['GET', 'POST'])
@cross_origin()
def predictPotability():
    if request.method == 'GET':
        #get the needed arguments for performing the prediction
        ph = int(request.args.get('ph'))
        hardness = int(request.args.get('hardness'))
        solids = int(request.args.get('solids'))
        chloramines= int(request.args.get('chloramines'))
        sulfate= int(request.args.get('sulfate'))
        conductivity= int(request.args.get('conductivity'))
        organic_carbon= int(request.args.get('organic_carbon'))
        trihalomethanes= int(request.args.get('trihalomethanes'))
        turbidity= int(request.args.get('turbidity'))   

        # load the dataset
        dataset = pd.read_csv('water_potability.csv', delimiter=',')
        
        #print(dataset.head())
        #print(dataset.isnull().sum())

        #deal with missing values 
        #Replace null values based on the group/sample mean
        dataset['ph']=dataset['ph'].fillna(dataset.groupby(['Potability'])['ph'].transform('mean'))
        dataset['Sulfate']=dataset['Sulfate'].fillna(dataset.groupby(['Potability'])['Sulfate'].transform('mean'))
        dataset['Trihalomethanes']=dataset['Trihalomethanes'].fillna(dataset.groupby(['Potability'])['Trihalomethanes'].transform('mean'))
        print(dataset.isnull().sum())

        # Drop first row 
        # by selecting all rows from first row onwards
        new_header = dataset.iloc[0] #grab the first row for the header
        dataset = dataset[1:] #take the data less the header row
        dataset.columns = new_header #set the header row as the df header
        print(dataset.head())

        # split into input (X) and output (y) variables
        X = dataset.iloc[:,0:9]
        print(X)
        y = dataset.iloc[:,9]
        print(y)
        # define the keras model
        model = Sequential()
        model.add(Dense(12, input_dim=9, activation='relu'))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # compile the keras model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit the keras model on the dataset
        model.fit(X, y, epochs=10, batch_size=10)
        # evaluate the keras model
        _, accuracy = model.evaluate(X, y)
        print('Accuracy: %.2f' % (accuracy*100))

        #use parameters given by user in order to make the prediction
        water_sample=[ph,hardness,solids,chloramines,sulfate,
                    conductivity,organic_carbon,trihalomethanes,turbidity]
        # make class predictions with the model
        prediction = (model.predict([water_sample])).astype(int)
        #print(prediction)
        #transform the array of new_output to string and check the value.
        prediction_result = np.array2string(prediction)
        #if it is 0 then return a message that says that the water is not potable and if it is 1 then return a message that says 
        #that the water is potable
        if "0" in prediction_result:
            #print("This water is not potable.")
            return Response('{"message":"Based on Neural Networks ML algorithm, this water is not potable."}', status=200, mimetype="application/json")
        elif "1" in prediction_result:
            #print("This water is potable.")
            return Response('{"message":"Based on Neural Networks ML algorithm, this water is potable."}', status=200, mimetype="application/json")
        else:
            return Response('{"message":"Please try again."}', status=500, mimetype="application/json")

if __name__ == "__main__":
   app.run(debug=True, host='0.0.0.0', port=5000)