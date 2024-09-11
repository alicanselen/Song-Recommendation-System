import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle

# nltk_data dizininin yolunu belirtin
nltk.data.path.append(os.path.abspath('nltk_data'))

# Veri kümesini yükleyin
data = pd.read_csv("spotify_millsongdata.csv")

# İlk 10 satırı göster
print(data.head(10))

# Veri kümesinin şeklini kontrol et
print(data.shape)

# Boş değerlerin sayısını kontrol et
print(data.isnull().sum())

# Veri kümesinden örneklem alın ve 'link' sütununu düşür
data = data.sample(5000).drop('link', axis=1).reset_index(drop=True)

# Metin verilerini ön işleme tabi tut
data['text'] = data['text'].str.lower().replace(r'\n', ' ', regex=True)

# PorterStemmer örneği oluşturun
stemmer = PorterStemmer()

# Tokenizasyon ve kök bulma fonksiyonu tanımlayın
def token(txt):
    tokens = nltk.word_tokenize(txt)
    stemmed_tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(stemmed_tokens)

# Örnek tokenizasyon işlemi
print(token("you are beautiful, beauty"))

# 'text' sütunundaki metinleri tokenizasyon işlemiyle işleyin
data['text'] = data['text'].apply(lambda x: token(x))

# TF-IDF vektörizer oluşturun ve metinleri vektörleştirin
tfid = TfidfVectorizer(analyzer='word', stop_words='english')
matrix = tfid.fit_transform(data['text'])

# Kosinüs benzerlik matrisini hesaplayın
similarity = cosine_similarity(matrix)

# Belirli bir şarkının indeksini alın
def get_song_index(song_name):
    if 'song' not in data.columns:
        print("'song' sütunu bulunamadı.")
        return -1
    if song_name in data['song'].values:
        return data[data['song'] == song_name].index[0]
    else:
        print(f"'{song_name}' şarkısı veri kümesinde bulunamadı.")
        return -1

# Şarkı öneri fonksiyonu tanımlayın
def recommend(song_name):
    idx = get_song_index(song_name)
    if idx == -1:
        return f"'{song_name}' şarkısı veri kümesinde bulunamadı."
    distance = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    song = []
    for song_id in distance[1:5]:
        song.append(data.iloc[song_id[0]].song)
    return song

# Veri kümesindeki sütun adlarını kontrol et
print("Sütun Adları:", data.columns)

# 'song' sütunundaki örnek değerleri kontrol et
print("Song Sütunundaki Örnek Değerler:", data['song'].head(10))

# Tüm şarkı adlarını küçük harfe çevirip baştaki ve sondaki boşlukları kaldırarak normalleştirin
data['song'] = data['song'].str.lower().str.strip()

# Normalleştirilmiş şarkı adlarını kontrol et
print("Normalleştirilmiş Song Sütunundaki Örnek Değerler:", data['song'].head(10))

# Belirli bir şarkının veri kümesinde olup olmadığını kontrol et
song_name_to_check = "bang".lower().strip()
print(f"'{song_name_to_check}' şarkısı veri kümesinde var mı?:", song_name_to_check in data['song'].values)

# Örnek bir şarkı için öneriler alın
print(recommend(song_name_to_check))

# similarity ve data dosyalarını .pkl uzantılı olarak kaydet
with open("similarity.pkl", "wb") as f:
    pickle.dump(similarity, f)

with open("data.pkl", "wb") as f:
    pickle.dump(data, f)
