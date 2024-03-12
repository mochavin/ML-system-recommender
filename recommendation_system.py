# -*- coding: utf-8 -*-
"""recommendation_system.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aMLN9oeRUyBr6p8KQ7kQ9GcDs4nClHZc

# Data Loading

## Download datasets dan unzip
"""

!mkdir ~/.kaggle
!mv ./kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d sunilgautam/movielens

! unzip "movielens.zip"

"""## Import Library yang dibutuhkan"""

# Commented out IPython magic to ensure Python compatibility.
import zipfile
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
# %matplotlib inline

from google.colab import files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv('/content/ml-latest-small/ratings.csv')
movies = pd.read_csv('/content/ml-latest-small/movies.csv')

"""# Data Understanding

## Memastikan movies dan rating termuat dengan baik
"""

movies.head()

ratings.head()

"""## Melihat detail dari movies dan ratings"""

movies.info()
print('\n')
ratings.info()

"""## Shape dari movies dan ratings"""

print(movies.shape)
print(ratings.shape)

"""Terlhat bahwa movies terdapat 3 kolom dan 9742 baris dan ratings terdapat 4 kolom 100836 baris

## Mengecek Missing Value
"""

print(movies.isnull().sum())
print('\n')
print(ratings.isnull().sum())

"""Tidak ada *Missing Value* baik di movies dan di ratings

## Mengecek data duplikat
"""

print(movies.duplicated().sum())
print(ratings.duplicated().sum())

"""Tidak ada data duplikat baik di movies dan ratings

## Menghitung Data Unik
"""

print(movies.nunique())
print('\n')
print(ratings.nunique())

"""Pada variabel movies terdaoat 9742 movieId, 9737 title, dan 951 genres unik.
Pada variabel ratings terdaoat 610 userId, 9724 movieId, 10 rating, dan 85043 timestamp unik.

## Mengecek sebaran rating
"""

print(ratings.rating.value_counts());
ratings_counts = ratings.rating.value_counts()

# Mengambil index (nilai rating) dan nilai (jumlah rating)
ratings_index = ratings_counts.index
ratings_values = ratings_counts.values

# Membuat plot distribusi
plt.bar(ratings_index, ratings_values)

# Menambahkan judul dan label sumbu
plt.title('Distribusi Rating')
plt.xlabel('Rating')
plt.ylabel('Jumlah')

# Menampilkan plot
plt.show()

"""Terlihat sebaran ratings kelipatan 0.5 dengan rating terendah 0.5 dan rating tertinggi 5.0

## Analisa Distribusi Tahun Rilis
"""

movies_year = movies.copy()
movies_year['year'] = movies['title'].str.extract('(\d+)').astype(float)
movies_year.year.dropna(inplace=True)

"""Membuat variabel __movies_year__ dan mengekstract judul dari variabel __movies__ dan drop *missing value*"""

new_movies_year = movies_year[(movies_year['year'] > 1800.0) & (movies_year['year'] < 2100.0)]
new_movies_year.year.astype(int)
data = new_movies_year

"""Melakukan filter tahun antara setelah 1800 dan 2100"""

plt.figure(figsize=(12,6))
sns.histplot(data=data, x='year')
plt.title('Distribusi Sebaran Tahun Rilis Film', fontsize=15, pad=15)
plt.xlim(min(data['year']), max(data['year']))
plt.tight_layout()
plt.show()

"""Terlihat sebaran tahun rilis film paling banyak di tahun 2000-an

## Analisa Distribusi Genre Film
"""

movies_genres = movies.copy()
genres=[]
for i in range(len(movies.genres)):
    for x in movies.genres[i].split('|'):
        if x not in genres:
            genres.append(x)
genres

"""Mengekstrak genre, genre di dataset untuk satu film bisa lebih dari satu dan dipisahkan dengan '|'"""

for x in genres:
    movies_genres[x] = 0

for i in range(len(movies.genres)):
    for x in movies.genres[i].split('|'):
        movies_genres[x][i]=1

movies_genres.head()

"""Untuk genre buat *One Hot Encoding* untuk memudahkan pengelohan data"""

data = movies_genres.iloc[:,3:].sum().reset_index()
data.columns = ['title','total']

plt.figure(figsize=(20,10))
sns.barplot(x='title', y='total', data=data)
plt.title('Distribusi Sebaran Genre Film', fontsize=15)
plt.tight_layout()
plt.show()

"""Terlihat sebaran genre di dataset, yang terbanyak adalah **Drama** diikuti dengan **Comedy**

## Analisa Rating
"""

# Jumlah pengguna yang memberikan rating
jumlah_pengguna = len(ratings['userId'].unique())

# Jumlah film yang diberi rating oleh pengguna
jumlah_film = len(ratings['movieId'].unique())

print(f'Jumlah pengguna yang memberikan rating: {jumlah_pengguna}')
print(f'Jumlah film yang diberi rating oleh pengguna: {jumlah_film}')

rating_movies = pd.merge(ratings, movies, on='movieId', how='inner')
rating_movies.drop(['timestamp','genres'],axis=1, inplace=True)
rating_movies.head()

"""merge ratings dan movies berdasar __movieId__ drop *timestamp* dan *genres* yang tidak diperlukan pada analisa rating kali ini"""

rating_movies_count = rating_movies.groupby('title')['rating'].count()
rating_movies_count = pd.DataFrame(rating_movies_count).reset_index().rename(columns={'rating':'total_rating'})
rating_movies_count.head()

"""Kelompokkan berdasarkan judul untuk melihat rating yang diperoleh suatu film"""

data = rating_movies_count.sort_values(by ='total_rating')

plt.figure(figsize=(25,10))
sns.barplot(data=data.iloc[-10:,:],
            y='title', x='total_rating',
            )
plt.title('Frekuensi Rating Film Tertinggi', pad=0, fontsize=30)
plt.tight_layout()
plt.show()

"""Terlihat bahwa film yang paling banyak dirating adalah __Forrest Gump(1994)__

# Data Preparation

## Membersihkan *Missing Value*
"""

movies.dropna(axis=0, inplace=True)
ratings.dropna(axis=0, inplace=True)

"""## Sorting Rating Berdasarkan userId"""

ratings = ratings.sort_values('userId').astype('int')

"""## Drop Data Duplikat"""

movies.drop_duplicates(subset=['title'], keep='first', inplace=True)
ratings.drop_duplicates(subset=['userId','movieId'], keep='first', inplace=True)

"""## Menggabungkan Data"""

merge_df = pd.merge(ratings, movies, how='left', on='movieId')
df = merge_df.copy().drop('timestamp', axis=1)
df.head()

"""Menggabungkan ratings dan movies berdasarkan *MovieId*"""

df = df[~pd.isnull(df['genres'])]
df.shape

"""Menghapus data *null* dan melihat *shape* akhir, terdapat 100830 baris dan 5 kolom

# Model Development

## Content Based Filtering
"""

tfid = TfidfVectorizer(stop_words='english')
tfid.fit(movies['genres'])
tfid.get_feature_names_out()

"""Melakukan inisialisasi TF-IDF dan melakukan mapping

## Ubah Data ke Bentuk Matriks
"""

tfidf_matrix = tfid.fit_transform(movies['genres'])
tfidf_matrix.shape

"""## Menghitung cosine similarity"""

cos_similar = cosine_similarity(tfidf_matrix)
cos_similar

"""## Membuat Dataframe cosine similarity"""

cos_similar = pd.DataFrame(cos_similar, index=movies['title'],
                             columns=movies['title'])
cos_similar

"""Terlihat nilai *cosine similarity* untuk tiap film

## Uji Coba Model Content Based Filtering

### Fungsi untuk mendapatkan 10 film dengan kemiripan tertinggi berdasarkan cosine similarity
"""

def MovieRecommendations(movies_title, similarity_data=cos_similar,
                         items=movies[['movieId','title','genres']], k=10):

    index = similarity_data.loc[:, movies_title].to_numpy().argpartition(
        range(-1, -k, -1)
    )

    closest = similarity_data.columns[index[-1:-(k+2):-1]]

    closest = closest.drop(movies_title, errors='ignore')

    return pd.DataFrame(closest).merge(items).head(k)

"""Mengambil 10 data dengan nilai *cosine similarity* terbesar dan drop dirinya sendiri agar tidak muncul dalam daftar rekomendasi

## Menguji Model Rekomendasi
"""

find_title = movies[movies['title'] == 'Flint (2017)']
find_title

movie_title = 'Flint (2017)'
movie_recomend = MovieRecommendations(movie_title)
movie_recomend

"""Terlihat dari 10 film yang direkomendasikan, semuanya memilki genre yang sama dengan 'Flint (2017)' yaitu drama.

## Collaborative Filtering

### Encoding useId dan movieId
"""

user_id = df['userId'].unique().tolist()
user_to_user_encoded = {x: i for i, x in enumerate(user_id)}
user_encoded_to_user = {i: x for i, x in enumerate(user_id)}

movie_id = df['movieId'].unique().tolist()
movie_to_movie_encoded = {x: i for i, x in enumerate(movie_id)}
movie_encoded_to_movie = {i: x for i, x in enumerate(movie_id)}

"""### Mapping user dan movie dengan Encoded user dan movie"""

df['user'] = df['userId'].map(user_to_user_encoded)
df['movie'] = df['movieId'].map(movie_to_movie_encoded)
df.head()

num_users = len(user_to_user_encoded)
num_movie = len(movie_encoded_to_movie)
df['rating'] = df['rating'].values.astype(np.float32)
min_rating = min(df['rating'])
max_rating = max(df['rating'])

print(f'Banyaknya user: {num_users}')
print(f'Banyaknya movie: {num_movie}')
print(f'Rating tertinggi: {min_rating}')
print(f'Rating terendah: {max_rating}')

df = df.sample(frac=1, random_state=12)
df.head()

"""### Pembagian Data untuk Training dan Validasi"""

x = df[['user', 'movie']].values
y = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

"""membagi data train sebesar 80% dan data validasi sebesar 20%

### Pelatihan Model

#### Pembuatan Fungsi
"""

class RecommenderNet(tf.keras.Model):
  def __init__(self, num_users, num_movies, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_movies = num_movies
    self.embedding_size = embedding_size

    # Embedding layers
    self.user_embedding = layers.Embedding(
        num_users,
        embedding_size,
        embeddings_initializer='he_normal',
        embeddings_regularizer=keras.regularizers.l2(1e-6)
    )
    self.movie_embedding = layers.Embedding(
        num_movies,
        embedding_size,
        embeddings_initializer='he_normal',
        embeddings_regularizer=keras.regularizers.l2(1e-6)
    )

    # Bias layers
    self.user_bias = layers.Embedding(num_users, 1)
    self.movie_bias = layers.Embedding(num_movies, 1)

    # Output layer
    self.output_layer = layers.Dense(1, activation='sigmoid')

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:, 0])
    user_bias = self.user_bias(inputs[:, 0])
    movie_vector = self.movie_embedding(inputs[:, 1])
    movie_bias = self.movie_bias(inputs[:, 1])

    # Dot product of user and movie embeddings
    dot_user_movie = tf.tensordot(user_vector, movie_vector, axes=2)

    # Combine embeddings and biases
    x = dot_user_movie + user_bias + movie_bias

    # Apply output layer
    return self.output_layer(x)

"""#### Kompilasi dan Konfigurasi Model"""

model = RecommenderNet(num_users, num_movie, 50)

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[[tf.keras.metrics.MeanAbsoluteError(name='mae'),tf.keras.metrics.RootMeanSquaredError(name='rmse')]]
)
callbacks = EarlyStopping(
    min_delta=0.0001,
    patience=7,
    restore_best_weights=True,
)

"""#### Training Model"""

history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 8,
    epochs = 20,
    validation_data = (x_val, y_val),
    callbacks=[callbacks]
)

"""### Uji Coba Model Collaborative Filtering

#### Mengambil sample userId
"""

user_ID = df.userId.sample(1).iloc[0]
movie_watched_by_user = df[df.userId == user_ID]

movie_not_watched = movies[~movies['movieId'].isin(movie_watched_by_user.movieId.values)]['movieId']
movie_not_watched = list(
    set(movie_not_watched)
    .intersection(set(movie_to_movie_encoded.keys()))
)


movie_not_watched = [[movie_to_movie_encoded.get(x)] for x in movie_not_watched]
user_encoder = user_to_user_encoded.get(user_ID)
user_movie_array = np.hstack(
    ([[user_encoder]] * len(movie_not_watched), movie_not_watched)
)

"""#### Menguji Model Collaborative Filtering"""

ratings = model.predict(user_movie_array).flatten()

top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_movie_ids = [
    movie_encoded_to_movie.get(movie_not_watched[x][0]) for x in top_ratings_indices
]

print('Rekomendasi untuk user {}'.format(user_ID))
print('\nmovie with high ratings')

top_movie_user = (
    movie_watched_by_user.sort_values(
        by = 'rating',
        ascending=False
    )
    .head(5)
    .movieId.values
)

movie_df_rows = movies[movies['movieId'].isin(top_movie_user)]
for row in movie_df_rows.itertuples():
    print(row.title)


print('\nTop movies recommendation: ')

recommended_movie = movies[movies['movieId'].isin(recommended_movie_ids)]
for row in recommended_movie.itertuples():
    print(row.title)

"""# Evaluasi

Metriks yang digunakan adalah *Mean Absolute Error* (MAE) dan *Root Mean Squared Error* (RMSE) pada *Collaborative Filtering* dan *Precision* dan *recall* pada *Content Based Filtering*

## Content Based Filtering
"""

def precision(recommended_movies, relevant_movies, k=10):
    recommended_movies = recommended_movies['title'].tolist()[:k]
    relevant_recommended = [movie for movie in recommended_movies if movie in relevant_movies]
    return len(relevant_recommended) / k

def recall(recommended_movies, relevant_movies, k=10):
    recommended_movies = recommended_movies['title'].tolist()[:k]
    relevant_recommended = [movie for movie in recommended_movies if movie in relevant_movies]
    return len(relevant_recommended) / len(relevant_movies)

movie_title = 'WALL·E (2008)'
relevant_movies = ['Titan A.E. (2000)', 'Transformers: The Movie (1986)', 'Chicken Little (2005)', 'Toy Story 3 (2010)', 'BURN-E (2008)', 'Ratchet & Clank (2016)', 'Meet the Robinsons (2007)', 'Lilo & Stitch (2002)	']

recommended_movies = MovieRecommendations(movie_title)

print(f"Precision@10: {precision(recommended_movies, relevant_movies, k=10)}")
print(f"Recall@10: {recall(recommended_movies, relevant_movies, k=10)}")

"""mendefinisikan fungsi precision dan recall untuk menghitung metrik evaluasi Precision@10 dan Recall@10 pada sistem rekomendasi film Content Based Filtering. sample yang diuji adalah WALL-E (2008). Dengan memasukkan judul film 'WALL·E (2008)' dan daftar film yang dianggap relevan, didapatkan rekomendasi film dari Content Based Filtering, menghitung Precision@10 sebesar 0.5 dan Recall@10 sebesar 0.625. Precision@10 mengukur berapa banyak rekomendasi yang relevan di antara 10 rekomendasi teratas, sedangkan Recall@10 mengukur berapa banyak film relevan yang berhasil direkomendasikan.

## Plot Mean Absolute Error
"""

plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('model_metrics')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['mae', 'val_mae'])
plt.show()

"""metriks MAE konvergen di sekitar 0.1300 untuk training dan 0.14250 untuk validasi

## Plot Root Mean Squared Error
"""

plt.plot(history.history['rmse'])
plt.plot(history.history['val_rmse'])
plt.title('model_metrics')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['rmse', 'val_rmse'])
plt.show()

"""metriks RMSE konvergen di sekitar 0.1700 untuk training dan 0.1850 untuk validasi"""