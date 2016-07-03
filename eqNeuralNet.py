# Project by Boris Alzamora
# Earthquakes Magnitude Prediction in Peru -
# 	A comparison between Multilayer Neural Network and Decision tree classifiers

# Magnitude prediction for the next earthquake in the Peru/Ecuador area.
# by Comparing a Supervised Neural Network Classifier and a Decision Tree Classifier
# Comparison by Accuracy, MSA and MSE measurements.

import sys
import pymysql
from sklearn.neural_network import MLPClassifier # Multi-Layer Perceptron Classifier
from sklearn import tree

########################################################################################
# DATABASE CONNECTION MANAGEMENT
# MySQL Singleton Class with pymysql for Anaconda
# Thanks to: http://www.manejandodatos.es/2014/02/anaconda-mysql/
class MySQLConnector(object):
    #	http://famousphil.com/blog/2012/01/mysql-singleton-classes-in-php-and-python/
	_connection = None
	_instance = None

	def __init__(self, host="localhost", user="root", passwd="root", database="earthquake", debug=False):	# Versión 1.0.1
		try:
			if MySQLConnector._instance == None:
				MySQLConnector._instance = self
				self.dbhost = host
				self.dbuser = user
				self.dbpassword = passwd
				self.dbname = database
				MySQLConnector._instance.connect(debug)	# Versión 1.0.1
		except Exception as e:
			print("MySQL Error "+str(e))

	def instance(self):
		return MySQLConnector._instance

	def get_connection(self):
		return MySQLConnector._connection

	def connect(self, debug=False):
		try:
			MySQLConnector._connection = pymysql.connect(self.dbhost, self.dbuser, self.dbpassword, self.dbname)
			if debug:
					print("INFO: Database connection successfully established")
		except Exception as e:
			print("ERROR: MySQL Connection Couldn't be created... Fatal Error! "+str(e))
			sys.exit()

	def disconnect(self):
		try:
			MySQLConnector._connection.close()
		except:
			pass;#connection not open

	#returns escaped data for insertion into mysql
	#def esc(self, esc):
	#	return MySQLdb.escape_string(str(esc));

	#query with no result returned
	def query(self, sql):
		cur = MySQLConnector._connection.cursor()
		return cur.execute(sql)

	def tryquery(self, sql):
		try:
			cur = MySQLConnector._connection.cursor()
			return cur.execute(sql)
		except:
			return False

	#inserts and returns the inserted row id (last row id in PHP version)
	def insert(self, sql):
		cur = MySQLConnector._connection.cursor()
		cur.execute(sql)
		return self._connection.insert_id()

	def tryinsert(self, sql):
		try:
			cur = MySQLConnector._connection.cursor()
			cur.execute(sql)
			return self._connection.insert_id()
		except:
			return -1

	#returns the first item of data
	def queryrow(self, sql):
		cur = MySQLConnector._connection.cursor()
		cur.execute(sql)
		return cur.fetchone()

	#returns a list of data (array)
	def queryrows(self, sql):
		cur = MySQLConnector._connection.cursor()
		cur.execute(sql)
		return cur.fetchall()

########################################################################################
# MAGNITUDE PREDICTION SECTION STARTS
# For Sklearn Documentation refer to: http://goo.gl/oLkLrc

# Establish MySQL Connection
conn = MySQLConnector("localhost", "root", "root", "earthquake", True)

# Capture User input -----------------------------------------------------
# 0 : rounded magnitudes, 1 :exact magnitudes.
exactMagnitudes = 0	if input("Consider rounded magnitudes? Y/N: ").upper() == "Y" else 1
significantMagnitudeAt = 4.5 if input("Consider only significant earthquakes? Y/N: ").upper() == "Y" else 0
showProgress = True if input("Show progress? Y/N: ").upper() == "Y" else False

# Captured desired pre-processing strategy
applySAUD = False
if input("Use Time Gaps(G) or Earthquake Sequence Number(S)? G/S: (default G)").upper() == "S":
	query = "CALL getEqCountPosTilesFormat('mb', " + str(significantMagnitudeAt) + ", " + str(exactMagnitudes) + ");"
	applySAUD = True
else:
	query = "CALL getEqTimeGapPosTilesFormat('mb', "+str(significantMagnitudeAt)+", "+str(exactMagnitudes)+");"

# Retrieve relevant data by feature extraction
data = conn.queryrows(query)
_data = []
_target = []
for row in data:
	_target.append(row[4])
	_data.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
	maxDate = float(row[0]) if applySAUD else 0

# train and test data split
percent = 0.8
dataSplit = int(round(len(data)*percent, 0))

# Train and Test Data Split
X, y = _data, _target
X_train, X_test = X[:dataSplit], X[dataSplit:]
y_train, y_test = y[:dataSplit], y[dataSplit:]

# Declare and configure the Multilayer Perceptron Classifier
mlp = MLPClassifier(hidden_layer_sizes=(3, 3),
					max_iter=20000,
					alpha=1e-5, # avoid overfitting by penalizing
					algorithm='sgd', # stochastic gradient descent
					verbose=showProgress, # Show weight update by loss,
					# The loss function for classification is Cross-Entropy
					activation='logistic', # Activation function for the hidden layer: returns f(x) = 1 / (1 + exp(-x))
					tol=1e-5, # Tolerance for the optimization when loss decreases
					random_state=1, # seed for random values
					learning_rate_init=.001, # learning rate for weight update
					learning_rate='adaptive') # Adapts the learning rate when reaching loss improvement boundary
mlp.fit(X_train, y_train) # Train

# Show size of train and test sets
print("\nTrain split: "+str(percent))
print("Train set size: %i" % len(y_train))
print("Test set size: %i" % len(y_test))

# score: Returns the mean accuracy on the given test data and labels
print("\nMLP Classifier Training set score: %f" % mlp.score(X_train, y_train))
print("MLP Classifier Test set score: %f" % mlp.score(X_test, y_test))

# DECISION TREE SECTION STARTS
dtc = tree.DecisionTreeClassifier()
dtc.fit(X_train, y_train)
# score: Returns the mean accuracy on the given test data and labels
print("DTC Classifier Training set score: %f" % dtc.score(X_train, y_train))
print("DTC Classifier Test set score: %f" % dtc.score(X_test, y_test))

##############################################################################
# PREDICTING UNSEEN EARTHQUAKES
# In order to verify them, we will get new samples that actually happened right after our last sample.
# Our last sample happened on 2016-05-09 01:22:44.740.
# So, our unseen data starts on 2016-05-09 01:22:44.741
# Source: http://earthquake.usgs.gov/earthquakes/search/

# time						latitude	longitude		depth		mag		magType
# 2016-06-02T00:01:28.470Z	-16.8394	-70.4347		15.46		5.2		mb
# 2016-06-01T21:31:00.980Z	-13.5072	-75.0698		78.88		5.1		mb
# 2016-06-01T00:51:11.750Z	-15.856		-69.5436		252.34		4.4		mb
# 2016-05-27T10:00:49.080Z	-8.4499		-74.9269		132.38		5.2		mb
# 2016-05-24T16:35:22.500Z	-2.7359		-78.6925		100.43		4.7		mb
# 2016-05-16T00:51:13.980Z	-13.7595	-73.0132		25.61		4.5		mb
# 2016-05-12T23:09:27.650Z	-15.79		-75.0643		10			4.7		mb
# 2016-05-11T21:01:44.720Z	-8.0686		-74.4916		160.9		4.5		mb
# 2016-05-09T01:22:44.860Z	-7.2326		-74.8907		131.25		4.5		mb

queries = ["call convertEarthquake('2016-06-02 00:01:28.470' ,-16.8394 , -70.4347,	15.46 ,5.2 ,'mb');",
		   "call convertEarthquake('2016-06-01 21:31:00.980' ,-13.5072 , -75.0698,	78.88 ,5.1 ,'mb');",
		   "call convertEarthquake('2016-06-01 00:51:11.750' ,-15.856 , -69.5436,	252.34 ,4.4 ,'mb');",
		   "call convertEarthquake('2016-05-27 10:00:49.080' ,-8.4499 , -74.9269,	132.38 ,5.2 ,'mb');",
		   "call convertEarthquake('2016-05-24 16:35:22.500' ,-2.7359 , -78.6925,	100.43 ,4.7 ,'mb');",
		   "call convertEarthquake('2016-05-16 00:51:13.980' ,-13.7595 , -73.0132,	25.61 ,4.5 ,'mb');",
		   "call convertEarthquake('2016-05-12 23:09:27.650' ,-15.79 , -75.0643,	10 ,4.7 ,'mb');",
		   "call convertEarthquake('2016-05-11 21:01:44.720' ,-8.0686 , -74.4916,	160.9 ,4.5 ,'mb');",
		   "call convertEarthquake('2016-05-09 01:22:44.860' ,-7.2326 , -74.8907,	131.25 ,4.5 ,'mb');"
		   ]
magnitudes = [5.2, 5.1, 4.4, 5.2, 4.7, 4.5, 4.7, 4.5, 4.5]
shift = 10 if exactMagnitudes else 1

print("\nPREDICTING UNSEEN EARTHQUAKES - - - - - - - - - - - - - - - - - - -");

idx = 0
errorSum = 0
mse = 0
for qry in queries:
	unseen = conn.queryrows(qry)[0]
	if applySAUD:
		maxDate += 1
		prd = mlp.predict([[float(maxDate), unseen[1], unseen[2], unseen[3]]])
	else:
		prd = mlp.predict([[int(unseen[0]), unseen[1], unseen[2], unseen[3]]])
	tgtMag = magnitudes[idx]
	error = abs(tgtMag - prd / shift)
	errorSum = errorSum + error
	mse = mse + pow(error, 2)
	print("MLP Classifier predicts... " + str(tgtMag) + "M as " + str(prd/shift) + " with error:" + str(error))
	idx = idx +1
print("Mean Absolute Error: " + str(errorSum / len(magnitudes)))
print("Mean Squared Error: " + str(mse / len(magnitudes)))

print(" ")
idx = 0
errorSum = 0
mse = 0
for qry in queries:
	unseen = conn.queryrows(qry)[0]
	if applySAUD:
		maxDate += 1
		prd = dtc.predict([[float(maxDate), unseen[1], unseen[2], unseen[3]]])
	else:
		prd = dtc.predict([[int(unseen[0]), unseen[1], unseen[2], unseen[3]]])
	tgtMag = magnitudes[idx]
	error = abs(tgtMag - prd / shift)
	errorSum = errorSum + error
	mse = mse + pow(error, 2)
	print("DTC Classifier predicts... " + str(tgtMag) + "M as " + str(prd / shift) + " with error:" + str(error))
	idx = idx + 1
print("Mean Absolute Error: " + str(errorSum / len(magnitudes)))
print("Mean Squared Error: " + str(mse / len(magnitudes)))