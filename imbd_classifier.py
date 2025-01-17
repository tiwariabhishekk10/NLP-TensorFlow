import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense,Flatten


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

model=Sequential()
model.add(Embedding(vocab_size,embedding_dim,input_length=max_length))
model.add(Flatten())
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit(padded, training_labels_final, epochs=10, validation_data=(testing_padded, testing_labels_final))

#visualizing embeddings
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

import io

#decode back
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(padded[3]))
print(training_sentences[3])

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")#downlads the file which can be used on tf website of embedding projection 'projector.tensorflow.org'
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")#similar file
out_v.close()
out_m.close()

sentence = "I really think this is amazing. honest."
sequence = tokenizer.texts_to_sequences(sentence)
print(sequence)