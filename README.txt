Project by Boris Alzamora
Earthquakes Magnitude Prediction in Peru
A comparison between Multilayer Neural Network and Decision tree classifiers
============================================================================
PROJECT ELEMENTS
----------------------------------------------------------------------------
	Solution:
	-	create_earthquake_database.sql
	-	eqNeuralNet.py
	-	README.txt
	
	Documentation:
	-	earthquakes_peru.kml
	-	global_earthquake_data.xlsx
	-	report.pdf
============================================================================
SYSTEM REQUIREMENTS
----------------------------------------------------------------------------
	-	SciKit-Learn (v0.18)
	-	PyMySQL for Python 3.5
	-	MySQL (v5.7.9)
* OS: Windows or Mac.
============================================================================
ENVIRONMENT SETUP INSTRUCTIONS
----------------------------------------------------------------------------
In order to get the version 0.18 of SciKit Learn (which requires numpy), we
first need to install the v0.17 version, the easiest way is to get it is by
download and install Anaconda which is available in here: 
https://www.continuum.io/downloads
Then, download and unpack version 0.18 from here:
https://github.com/scikit-learn/scikit-learn
Once unpacked, the resulting files must be copied to the following path:
C:\Anaconda3\Lib\site-packages\sklearn (on windows/ equivalent in Mac).
In order to test it, open Idle and test:
from sklearn.neural_network import MLPClassifier
If passes, it means we have successfully installed v0.18 of SciKitLearn.

Use the same procedure for installing PyMySQL which can be downloaded from:
https://github.com/PyMySQL/PyMySQL

Next, install MySQL v5.7.9.
After installing, open and execute the given query file:
	create_earthquake_database.sql

In order to test it, execute the following tsql command:
	use earthquake;
	select count(1) from earthquakes;
Should be: 7732

Now test procedures by executing the following tsql command:
	select * from mysql.proc where db="earthquake";
The output should be 3 rows with the following procedures:
convertEarthquake, getEqCountPosTilesFormat and getEqTimeGapPosTilesFormat

============================================================================
THE PROBLEM
----------------------------------------------------------------------------
The following solution was created to predict the magnitude  of an earthquake
with certain accuracy by assuming that the earthquake happens.
Multiple features and sometimes noise data is found in this data, for example
multiple magnitude types for a unique magnitude value of an earthquake.

============================================================================
IMPLEMENTATION
----------------------------------------------------------------------------
The Collected data was obtained from USGS using close to 20 years of data.
(earthquake.usgs.gov/earthquakes/search/)

SciKit Learn provided an easy way to use a Multiple Layer Perceptron as our
classifier and also helped us to identify the accuracy of the problem under
different scenarios describe in the report.

Why compare against a Decision Tree?
	A Decision Tree is a classifier that could reach that level of precission
	required for analizing and then learning this type of problem.
	After training, results were useful to determine the best configuration 
	for the MultiLayer Perceptron.

============================================================================
STRUCTURE OF THE PROGRAM
----------------------------------------------------------------------------
The python program has 5 clearly defined parts:

	-	Database Connection Management:
		Using PyMySQL to connect to the db.

	-	User Input Management:
		Capturing user preferences, like Pre-Processing Strategy,
		progress, rounded magnitude values and sample space reduction to 
		significant earthquakes only.

	-	Dataset management:
		This section manages the relation between the training and testing 
		dataset sizes (split).

	-	Classifier Configuration and accuracy calculation:
		This section provides the configuration values for the Multilayer
		Perceptron Classifier.

	-	Prediction on New Unseen Data:
		This section will convert some new data to our pre-processing format
		in order to predict using the Multilayer Perceptron and a Decission
		Tree algorithm.

		Then after prediction, the Mean Absolute Error and the Mean Squared
		Error are calculated for both classifiers.
