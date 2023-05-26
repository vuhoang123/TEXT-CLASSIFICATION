import gensim
from pyvi import ViTokenizer
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pickle

X_data = pickle.load(open('data/X_data.pkl', 'rb'))

# word level - we choose max number of words equal to 30000 except all words (100k+ words)
tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
tfidf_vect.fit(X_data) # learn vocabulary and idf from training set
X_data_tfidf =  tfidf_vect.transform(X_data)

svd = TruncatedSVD(n_components=300, random_state=42)
svd.fit(X_data_tfidf)
model4 = load_model('model/cnn_model4.h5') 
def preprocessing_doc(doc):
    lines = gensim.utils.simple_preprocess(doc)
    lines = ' '.join(lines)
    lines = ViTokenizer.tokenize(lines)

    return lines

# model = load_model('model/cnn_model.h5')   # rcnn with data tf/idf (svd)
# model2 = load_model('model/cnn_model2.h5') # cnn with data tf/idf ngram level
# model3 = load_model('model/cnn_model3.h5') # cnn with data tf/idf ngram base char


def predict_category(text):
    classes = ['Chính trị xã hội', 'Đời sống', 'Khoa học', 'Kinh doanh',
               'Pháp luật', 'Sức khoẻ', 'Thế giới', 'Thể thao', 'Văn hoá',
               'Vi tính']
    # Tokenize input text and add special tokens
    text = preprocessing_doc(text)
    # cnn with data tf/idf (svd)
    text = preprocessing_doc(text)
    test_doc_tfidf = tfidf_vect.transform([text])
    test_doc_svd = svd.transform(test_doc_tfidf)
    predictions = model4.predict(test_doc_svd)
    predicted_labels =classes[np.argmax(predictions, axis=1)[0]]

    return predicted_labels
