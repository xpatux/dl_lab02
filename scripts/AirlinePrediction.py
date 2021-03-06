from keras.models import Sequential
from keras.layers import Dense, Dropout, TimeDistributed, Flatten, SimpleRNN,LSTM
import airlineUtils
import logging
from logger import set_logger

set_logger()
logger = logging.getLogger(__name__)


logger.debug('Test case: SimpleRNN')

for case in  ['a','b','c','d']: 

    logger.debug('Running case %s',case)
    #set sequence length
    if case == 'b': 
        history=10
    else:
        history=4

    #read training and test sets
    trainX, trainY, testX, testY, scaler, dataset=airlineUtils.readAirlineData(history)

    # create and fit the LSTM network
    modelRNN = Sequential()
    modelRNN.add(SimpleRNN(5,input_shape=(history,1), return_sequences=False))
    
    if case=='c':
        modelRNN.add(Dense(100)) # ACTIVIDAD 10
    
    if case=='d':
        modelRNN.add(Dropout(0.5))
    
    # 
    modelRNN.add(Dense(1))

    #Train model
    modelRNN.compile(loss='mean_squared_error', optimizer='adam')
    modelRNN.fit(trainX, trainY, epochs=150, batch_size=5, verbose=2)

    # Make predictions
    trainPredict = modelRNN.predict(trainX)
    testPredict = modelRNN.predict(testX)

    #Display results
    airlineUtils.displayResult(dataset, trainPredict, trainY, testPredict, testY, scaler, history, filename='simplernn_'+case)


