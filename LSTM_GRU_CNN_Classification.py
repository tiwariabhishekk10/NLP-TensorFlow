import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense,Flatten,Bidirectional,GRU, LSTM,Conv1D,GlobalMaxPool1D


imdb,info=tfds.load('imdb_reviews',with_info=True, as_supervised=True)

train,test=imdb['train'],imdb['test']

training_sentences=[]
training_labels=[]

testing_sentences=[]
testing_labels=[]

for s,l in train:
    training_sentences.append(str(s.numpy()))
    training_labels.append(str(l.numpy()))

for s,l in test:
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(str(l.numpy()))

training_labels_final=np.array(training_labels)
testing_labels_final=np.array(testing_labels)
print(training_labels.shape)

vocab_size=1000
embedding_dim=16
max_length=120
trunc_type='post'
oov_tok='<OOV>'

tokenizer=Tokenizer(num_words=vocab_size,oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index=tokenizer.word_index
sequences=tokenizer.texts_to_sequences(training_sentences)
padded=pad_sequences(sequences,maxlen=max_length,truncating=trunc_type)

testing_sequences=tokenizer.texts_to_sequences(testing_sentences)
testing_padded=pad_sequences(testing_sequences,maxlen=max_length)

#model with single layer LSTM
model=Sequential()
model.add(Embedding(vocab_size,embedding_dim,input_length=max_length))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(6,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit(padded, training_labels_final, epochs=10, validation_data=(testing_padded, testing_labels_final))

#MOdel with double layer LSTM
model=Sequential()
model.add(Embedding(vocab_size,embedding_dim,input_length=max_length))
model.add(Bidirectional(LSTM(64,return_sequences=True)))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(6,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit(padded, training_labels_final, epochs=10, validation_data=(testing_padded, testing_labels_final))


##CNN

model=Sequential()
model.add(Embedding(vocab_size,embedding_dim,input_length=max_length))
model.add(Conv1D(128,5,activation='relu')
model.add(GlobalMaxPool1D())
model.add(Dense(24,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit(padded, training_labels_final, epochs=10, validation_data=(testing_padded, testing_labels_final))

##GRU
model=Sequential()
model.add(Embedding(vocab_size,embedding_dim,input_length=max_length))
model.add(GRU(32)
model.add(Dense(6,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit(padded, training_labels_final, epochs=10, validation_data=(testing_padded, testing_labels_final))

