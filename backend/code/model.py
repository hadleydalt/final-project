import tensorflow
from tensorflow.karas import Sequential
print(tensorflow.__version__)

openPath = ""
closedPath = ""

model = Sequential()
model.add(Dense(100, input_shape=(8,)))
model.add(Dense(80))
model.add(dense(30))
model.add(dense(10))
model.add(dense(5))
model.add(dense(1))


opt = Adam(learning_rate=0.01, momentum = 0.9)
model.compile(optimizer = 'adam', loss = 'binary_corssentropy', matrics=['accuracy'])
model.fit(X,y epochs=100, batch_size=32, verbose=0)


#Following this model: https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/
#I haven't done too much here, but I've started and I feel decent about this direciton if we want to use tensorflow for our modeling