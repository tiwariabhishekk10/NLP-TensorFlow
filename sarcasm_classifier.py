##standard dataset on sarcasm by risabh mishra on kaggle
#to load data
import json

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense,Flatten,GlobalAveragePooling1D

vocab_size=1000
embedding_dim=16
max_length=16
trunc_type='post'
padding_type='post'
oov_tok='<OOV>'
training_size=2000

#Reading Json file
with open(r'D:\Python Code\NLP_Using_TensorFlow\Sarcasm_Headlines_Dataset.json','r') as f:
    data = json.loads("[" + f.read().replace("}\n{", "},\n{") + "]")# in json file data was not in list for that '[' ']' at the start and at the end and also adding comma after } in the file

sentences=[]
labels=[]
urls=[]

for item in data:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

training_sentences=sentences[0:training_size]
testing_sentences=sentences[training_size:]

training_labels=labels[0:training_size]
testing_labels=labels[training_size:]


tokenizer=Tokenizer(num_words=vocab_size,oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index=tokenizer.word_index
sequences=tokenizer.texts_to_sequences(training_sentences)#into set of sequences of list of a each sentence and word in each sentence into list of integers
padded=pad_sequences(sequences,padding=padding_type,maxlen=max_length,truncating=trunc_type)

testing_sequences=tokenizer.texts_to_sequences(testing_sentences)
testing_padded=pad_sequences(testing_sequences,maxlen=max_length)

model=Sequential()
model.add(Embedding(vocab_size,embedding_dim,input_length=max_length))
model.add(GlobalAveragePooling1D())
model.add(Dense(24,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit(padded, training_labels, epochs=30, validation_data=(testing_padded, testing_labels))
