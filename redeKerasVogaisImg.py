from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import pandas as pd

numpy.random.seed(4)
num_pixels = 13
second_layer = 1500
third_layer = 300
fourth_layer = 80
num_classes = 7
df = pd.read_csv('treino.csv', header=None, sep=',')


df = pd.read_csv('treino.csv', header=None, sep=',')
X_train = df.iloc[:, 0:13].values
y_train = df.iloc[:, 13:20].values

df = pd.read_csv('teste.csv', header=None, sep=',')
ytest = df.iloc[:, 0:13].values
#print(ytest)
#print(ytest.shape)

def base_model():
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(second_layer, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(third_layer, input_dim=second_layer, kernel_initializer='normal', activation='relu'))
	model.add(Dense(fourth_layer, input_dim=second_layer, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax', name='preds'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	#model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	return model

model = base_model()
model.summary()

# Fit the model
model.fit(X_train, y_train, epochs=200, batch_size=100, verbose=2)
# evaluate the model
scores = model.evaluate(X_train, y_train, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



# serialize model to JSON
model_json = model.to_json()
with open("vogaisrede.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("vogaisrede.h5")
print("Saved model to disk")

# later...
# load json and create model
json_file = open('vogaisrede.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("vogaisrede.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X_train, y_train, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

#codigo para printar as classificacoes finais 
classes = loaded_model.predict(ytest, batch_size=1)
print("\n\n\n CLASSIFICACOES FINAIS")
print(classes)
response = list(classes)
contResponse = 0
while contResponse<7:
	fResponse = max(response[contResponse])
	if (fResponse == response[contResponse][0]):
		print ('A')
	if (fResponse == response[contResponse][1]):
		print ('Eh')
	if (fResponse == response[contResponse][2]):
		print ('E')
	if (fResponse == response[contResponse][3]):
		print ('I')
	if (fResponse == response[contResponse][4]):
		print ('Oh')
	if (fResponse == response[contResponse][5]):
		print ('O')
	if (fResponse == response[contResponse][6]):
		print ('U')
	contResponse = contResponse + 1
	#print(numpy.argmax(classes))

