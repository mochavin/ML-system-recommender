# Laporan Proyek _Machine Learning_ - Moch. Avin

# Project Overview

Di era digital saat ini, keberadaan konten multimedia yang melimpah telah menyebabkan fenomena kelebihan informasi, membuat pengguna sering kali merasa kewalahan ketika harus memilih film yang sesuai dengan preferensi mereka. Kondisi ini tidak hanya mempengaruhi efisiensi dalam mencari dan memilih film tetapi juga dapat menurunkan kepuasan pengguna terhadap pengalaman mereka menggunakan platform multimedia seperti Youtube, Netflix, dan lain sebagainya. Dalam mengatasi masalah ini, pengembangan sistem rekomendasi film menjadi sangat penting, yang mana dapat menyajikan pengalaman yang lebih personal dan memuaskan bagi pengguna melalui pemanfaatan teknik _Content Based Filtering_ dan _Collaborative Filtering_ [1].

Teknik _Content Based Filtering_ bekerja dengan merekomendasikan film berdasarkan fitur konten dari film yang telah ditonton sebelumnya oleh pengguna [2], sementara teknik _Collaborative Filtering_ merekomendasikan film berdasarkan preferensi dari pengguna lain yang memiliki kesamaan selera [3]. Kedua teknik ini diharapkan dapat mengatasi masalah kelebihan informasi dengan menyediakan rekomendasi yang lebih akurat dan sesuai dengan preferensi pengguna, sekaligus meningkatkan interaksi pengguna dengan platform [1].

Dari perspektif bisnis, implementasi sistem rekomendasi yang efektif dapat meningkatkan kepuasan pengguna, yang pada gilirannya dapat meningkatkan retensi pengguna serta menarik lebih banyak pengguna baru. Hal ini memberikan peluang bisnis yang lebih baik bagi penyedia platform multimedia, baik dari segi peningkatan pendapatan iklan maupun potensi pendapatan dari layanan berbayar. Diharapkan, sistem rekomendasi ini tidak hanya dapat mengurangi waktu yang dihabiskan pengguna untuk mencari film yang diinginkan tetapi juga meningkatkan kepuasan pengguna dengan merekomendasikan film yang lebih sesuai dengan preferensi mereka [2].

Salah satu contoh pemanfaatan _machine learning_ dalam proyek ini adalah pengembangan model _Collaborative Filtering_. Model ini dapat belajar dari data interaksi pengguna terdahulu untuk mengidentifikasi pola kesamaan antar pengguna atau antar item (film) dan kemudian menggunakan pola tersebut untuk membuat prediksi tentang film mana yang mungkin disukai oleh pengguna berdasarkan kesamaan selera dengan pengguna lain. Dengan demikian, sistem rekomendasi yang berbasis pada teknologi _machine learning_ dapat secara otomatis menyesuaikan rekomendasinya dengan dinamika selera dan preferensi pengguna, secara efektif mengatasi masalah kelebihan informasi dan meningkatkan pengalaman pengguna dalam menemukan konten yang mereka sukai [3].

# Business Understanding

Sistem rekomendasi merupakan sebuah aplikasi yang berfungsi untuk memberikan saran atau rekomendasi kepada pengguna dalam mengambil keputusan sesuai dengan preferensi mereka. Untuk meningkatkan pengalaman pengguna dalam menemukan film-film yang menarik dan sesuai dengan minat mereka, penerapan sistem rekomendasi menjadi pilihan yang tepat. Dengan adanya sistem rekomendasi, pengalaman pengguna akan lebih baik karena mereka dapat memperoleh rekomendasi judul film yang sesuai dengan harapan dan keinginan mereka.

### Problem Statement

Berdasarkan latar belakang yang telah dipaparkan, terdapat beberapa permasalahan yang perlu diselesaikan dalam proyek ini:

- Bagaimana melakukan proses pengolahan data dengan baik agar data tersebut dapat dimanfaatkan untuk membuat model sistem rekomendasi yang berkualitas?
- Bagaimana cara mengembangkan model machine learning untuk merekomendasikan film yang relevan kepada pengguna dengan tingkat error di bawah 20%?

### Goals

Tujuan dibuatnya proyek ini adalah sebagai berikut:

- Melakukan serangkaian proses pengolahan data dengan metode yang tepat sehingga data tersebut siap dimanfaatkan untuk membangun model sistem rekomendasi yang berkualitas dan akurat.
- Mengembangkan model machine learning yang mampu menganalisis pola dan preferensi pengguna, serta memberikan rekomendasi judul film yang relevan dengan tingkat error di bawah 20%.

### Solution

Untuk mengatasi permasalahan ini, penulis akan mengimplementasikan dua pendekatan algoritma, yaitu _content-based filtering_ dan _collaborative filtering_. Berikut penjelasan dari kedua teknik tersebut:

- _Content-Based Filtering_ adalah metode untuk memberikan rekomendasi film kepada pengguna berdasarkan kemiripan genre atau fitur-fitur yang terdapat pada film-film yang telah disukai oleh pengguna tersebut di masa lalu. Pendekatan ini mempelajari profil preferensi pengguna baru dengan menganalisis data dari objek-objek (film) yang telah dinilai sebelumnya oleh pengguna.
- _Collaborative Filtering_ adalah metode untuk memberikan rekomendasi film kepada pengguna berdasarkan penilaian (_rating_) yang diberikan oleh komunitas pengguna lain dengan preferensi yang serupa. Pendekatan ini tidak memerlukan atribut atau fitur spesifik dari setiap film, melainkan menggunakan pola _rating_ yang diberikan oleh pengguna untuk menemukan kemiripan preferensi di antara mereka.

# Data Understanding

### Dataset

Proyek ini akan menggunakan dataset film dengan informasi genre dan rating. Dataset tersebut bersumber dari [Kaggle](https://www.kaggle.com/sunilgautam/movielens). Dataset ini masuk dalam kategori Movies & TV Shows dengan tingkat kegunaan (usability) sebesar 5.3. Dataset tersebut berbentuk file ZIP berukuran 3.3 MB yang berisi 4 file CSV.

Dataset ini berisi 4 file:

- links.csv
- ratings.csv
- movies.csv
- tags.csv

Dalam proyek ini, penulis hanya menggunakan 2 file dataset, yaitu:

1. _movies.csv_

   - _Jumlah Data: 9742, dengan 3 kolom_
     - `movieId`: ID dari film. Terdapat 9742 data unik.
     - `title`: Judul dari film. Terdapat 9737 data unik.
     - `genres`: Genre dari film. Terdapat 951 data unik.

2. _ratings.csv_
   - _Jumlah Data: 100836, dengan 4 kolom_
     - `userId`: ID pengguna pemberi rating. Terdapat 610 data unik.
     - `movieId`: ID film yang diberi rating. Terdapat 9724 data unik.
     - `rating`: Rating dari film. Terdapat 10 data unik dengan range 0 - 5 dan skala 0.5.
     - `timestamp`: Waktu rating terekam. Terdapat 85043 data unik.

### Exploratory Data Analysis

#### Distribusi Rating

Tabel 1. Distribusi rating
| Rating | Count |
| ------ | ----- |
| 4.0 | 26818 |
| 3.0 | 20047 |
| 5.0 | 13211 |
| 3.5 | 13136 |
| 4.5 | 8551 |
| 2.0 | 7551 |
| 2.5 | 5550 |
| 1.0 | 2811 |
| 1.5 | 1791 |
| 0.5 | 1370 |

![Distribusi rating](https://raw.githubusercontent.com/mochavin/ML-system-recommender/main/images/rating-distribution.png)
Gambar 1. Distribusi rating

Terlihat sebaran ratings kelipatan 0.5 dengan rating terendah 0.5 dan rating tertinggi 5.0 dan rating user terbanyak adalah 4.0.

#### Distribusi Tahun Rilis

![Tahun Rilis](https://raw.githubusercontent.com/mochavin/ML-system-recommender/main/images/tahun-rilis.png)
Gambar 2. Distribusi tahun rilis

Berdasarkan grafik yang disajikan, dapat diamati bahwa rata-rata tahun rilis film berada dalam rentang 1990 hingga 2000-an ke atas. Distribusi terbanyak terjadi di atas tahun 2000, di mana jumlah film yang dirilis cenderung mengalami peningkatan yang signifikan seiring berjalannya waktu.

#### Distribusi Genre

![Distribusi Genre](https://raw.githubusercontent.com/mochavin/ML-system-recommender/main//images/genre.png)
Gambar 3. Distribusi Genre

Dari gambar yang ditampilkan, dapat dilihat bahwa terdapat 20 kategori atau genre dalam dataset ini. Genre yang paling banyak muncul adalah **Drama**, diikuti oleh genre **Comedy**. Selain itu, terdapat beberapa film yang tidak memiliki genre yang terdaftar, ditandai dengan keterangan **no genres listed**.

#### Film yang Memiliki Paling Banyak Rating

![Frekuensi Rating Film](https://raw.githubusercontent.com/mochavin/ML-system-recommender/main/images/frekuensi-rating-film.png)
Gambar 4. Film dengan rating terbanyak

Terlihat bahwa film yang paling banyak dirating adalah **Forrest Gump(1994)** dengan lebih dari 300 rating.

# Data Preparation

_Data Preparation_ diperlukan untuk mempersiapkan data agar saat dilakukan proses pengembangan model, akurasi model dapat ditingkatkan dan meminimalisir terjadinya bias pada data. Tahapan-tahapan dalam melakukan _preprocessing_ data adalah sebagai berikut:

1. **Penanganan _Missing Value_**, Penanganan _missing value_ dilakukan dengan menghapus data yang memiliki nilai yang hilang. Keputusan ini diambil karena dataset yang digunakan cukup bersih, dan _missing value_ hanya terjadi saat penggabungan dataset. Pertimbangannya adalah untuk menjaga integritas data dan menghindari distorsi dalam analisis yang mungkin terjadi jika _missing value_ diisi dengan estimasi. Menghapus baris dengan _missing value_ adalah solusi yang paling tepat karena jumlahnya yang relatif kecil dibandingkan dengan keseluruhan dataset, sehingga dampak terhadap volume data yang tersedia untuk analisis dianggap minimal.

2. **Pengurutan Data Rating**, Data rating diurutkan berdasarkan ID pengguna untuk memudahkan penghapusan data duplikat di langkah selanjutnya. Pengurutan ini memungkinkan proses identifikasi dan penghapusan duplikat menjadi lebih efisien, karena data yang serupa akan bersebelahan setelah pengurutan.
3. **Penghapusan Data Duplikat**, Penghapusan data duplikat dilakukan untuk mencegah terjadinya bias pada data. Duplikat dalam data rating dapat mengakibatkan _skewness_ dalam distribusi rating dan mempengaruhi akurasi model. Oleh karena itu, penghapusan duplikat sangat penting untuk menghindari _skewness_.

4. **Penggabungan Data**, Penggabungan data yang telah diolah sebelumnya dilakukan untuk membangun model. Data yang memiliki missing value pada variabel genre dihapus karena genre merupakan variabel penting dalam sistem rekomendasi berbasis konten. Setelah penggabungan, dataset memiliki 100.830 baris dengan 5 kolom, yang menunjukkan bahwa proses penggabungan telah berhasil menciptakan dataset yang komprehensif untuk pemodelan.

5. **Normalisasi Nilai Rating**, Normalisasi nilai rating dilakukan untuk menghasilkan rekomendasi yang sesuai dan akurat. Proses ini biasanya melibatkan penyesuaian skala rating agar berada dalam rentang yang seragam, yang dapat membantu dalam mengurangi bias yang mungkin disebabkan oleh perbedaan skala rating antarpengguna. Teknik digunakan adalah min-max normalization. Hal ini memastikan bahwa model dapat membandingkan rating secara adil, terlepas dari perbedaan skala penggunaan rating oleh pengguna yang berbeda.

6. **Pembagian Dataset**, Pembagian dataset menjadi dataset _train_ dan validasi dilakukan, dengan 80% data digunakan untuk dataset _train_ dan 20% untuk dataset validasi. Pembagian ini penting untuk menguji kinerja model pada data yang tidak terlibat dalam proses pelatihan, memungkinkan evaluasi yang lebih objektif tentang seberapa baik model dapat memprediksi rating pada data baru. Pengembangan model _Collaborative Filtering_ bergantung pada kemampuan untuk mempelajari preferensi pengguna dari data historis, dan memiliki dataset validasi yang terpisah membantu dalam mengoptimalkan parameter model dan menghindari _overfitting_.

# Modeling

Dalam proyek ini, digunakan dua pendekatan utama untuk pembuatan model: _`Neural Network`_ dan _`Cosine Similarity`_. pemilihan _Neural Network_ untuk _Collaborative Filtering_ dan _Cosine Similarity_ untuk _Content-Based Filtering_ didasarkan pada beberapa pertimbangan.

_Neural Network_ dipilih karena kemampuannya yang luar biasa dalam mengidentifikasi pola dan hubungan tersembunyi dalam dataset yang besar dan kompleks, yang sangat cocok untuk menganalisis interaksi pengguna dalam _Collaborative Filtering_. Selain itu, _Neural Network_ mempunyai skalabilitas dan fleksibilitas yang dibutuhkan untuk menyesuaikan dengan pertumbuhan data, serta kemampuan untuk menyediakan personalisasi tingkat tinggi dalam rekomendasi.

Di sisi lain, _Cosine Similarity_ digunakan dalam _Content-Based Filtering_ karena efisiensinya dalam mengukur kemiripan antara berdasarkan fitur-fiturnya, membuatnya ideal untuk rekomendasi berbasis konten. Pendekatan ini efektif bahkan di dataset dengan dimensi tinggi dan dengan implementasi yang sederhana namun akurat. Gabungan kedua metode ini memungkinkan sistem rekomendasi untuk menyajikan rekomendasi yang tidak hanya akurat dan relevan tetapi juga sangat disesuaikan dengan preferensi pengguna.

### Content Based Filtering

Dalam proses _Content Based Filtering_, langkah awalnya adalah menggunakan _TF-IDF Vectorizer_ untuk mengidentifikasi fitur penting dari setiap genre film. Dengan menggunakan fungsi _tfidfvectorizer()_ dari _library sklearn_, akan ditransformasi data ke dalam bentuk matriks dengan ukuran (9737, 23), di mana 9737 adalah jumlah data dan 23 adalah jumlah genre film yang direpresentasikan dalam matriks.

Untuk menghitung (_similarity degree_) antar movie, digunakan teknik _cosine similarity_ dengan fungsi _`cosine_similarity`_ dari _library_ sklearn. Berikut dibawah ini adalah rumusnya:
$$ \text{CosineSimilarity}(u, v) = \frac{{u \cdot v}}{{\|u\| \|v\|}} $$

di mana:

- \( u ⋅ v \) adalah hasil perkalian titik antara vektor \( u \) dan \( v \).
- \( \|u\| \) adalah norma Euclidean dari vektor \( u \).
- \( \|v\| \) adalah norma Euclidean dari vektor \( v \).

Rumus ini digunakan untuk mengukur kemiripan antara dua vektor dalam ruang berdimensi banyak, seperti dalam kasus perbandingan kemiripan antara fitur-fitur film dalam sistem rekomendasi _Content Based Filtering_.

Langkah berikutnya melibatkan penggunaan _`argpartition`_ untuk mengambil k nilai tertinggi dari _similarity_ data dan mengekstrak data dari tingkat kesamaan tertinggi ke terendah. Kemudian, akurasi dari sistem rekomendasi diuji untuk menemukan rekomendasi film yang mirip dengan film yang dicari.

`Kelebihan` dari metode ini adalah semakin banyak informasi yang diberikan pengguna, semakin baik akurasi sistem rekomendasi. Namun, ada beberapa `kekurangan`, seperti keterbatasan dalam fitur yang bisa digunakan, seperti film dan buku, serta ketidakmampuan untuk menentukan profil pengguna baru.

#### Hasil uji Content Based Filtering

Berikut _sample_ untuk menguji hasil model _Content Based Filtering_

Tabel 2. _Sample_ yang akan diuji
| movieId | title | genres |
|---------|--------------|--------|
| 193585 | Flint (2017) | Drama |

_Sample_ yang diuji adalah **Flint (2017)** dengan genre **Drama**

Berikut ini merupakan hasil rekomendasi yang dihasilkan menggunakan _Content Based Filtering_ dan sample **Flint (2017)**:

Tabel 3. Hasil rekomendasi dari sample Flint (2017)
| title | movieId | genres |
| ------------------------------------------------- | ------- | ------ |
| Monsieur Ibrahim (Monsieur Ibrahim et les fleu... | 7299 | Drama |
| Proof (2005) | 36527 | Drama |
| Miss Meadows (2014) | 117107 | Drama |
| Melvin and Howard (1980) | 2988 | Drama |
| Bringing Out the Dead (1999) | 2976 | Drama |
| Body Shots (1999) | 2979 | Drama |
| Songs From the Second Floor (Sånger från andra... | 5515 | Drama |
| If These Walls Could Talk (1996) | 70990 | Drama |
| Radio Flyer (1992) | 7030 | Drama |
| Get on the Bus (1996) | 1054 | Drama |

Terlihat bahwa hasil rekomendasi dari _sample_ memiliki genre yang sama artinya hasil model yang dihasilkan cukup memuaskan

### Collaborative Filtering

Dalam proses pembuatan model _Collaborative Filtering_ yang memanfaatkan dataset film dan rating, penulis menggabungkan dua dataset utama, yaitu `movies.csv` dan `ratings.csv`. Langkah pertama dalam proses ini adalah melakukan _encoding_ terhadap `userId` dan `movieId`, yang menjadikan identitas unik pengguna dan film dalam format numerik yang dapat diproses oleh model. Selanjutnya, dilakukan proses _mapping_ untuk menyesuaikan data tersebut ke dalam dataset yang digunakan dan mengubah rating menjadi tipe data float, memastikan data siap untuk diproses oleh model.

Setelah persiapan data, dilakukan pembagian dataset menjadi dua bagian: 80% untuk _training_ dan 20% untuk validasi. Pembagian ini memungkinkan model untuk belajar dari sebagian besar data sambil memverifikasi kinerjanya terhadap sebagian data yang tidak digunakan selama proses pembelajaran, untuk menghindari _overfitting_ dan memastikan model dapat memprediksi dengan baik pada data baru.

Penulis kemudian mengimplementasikan _embedding_ untuk data pengguna dan film. Teknik _embedding_ ini memungkinkan representasi pengguna dan film dalam ruang vektor, di mana model dapat belajar preferensi dan karakteristik tersembuninya. Dengan melakukan operasi _dot product_ antara _embedding_ pengguna dan film dan menambahkan bias yang spesifik untuk setiap pengguna dan film, model dapat menghitung skor kesesuaian yang menunjukkan seberapa cocok film tersebut untuk pengguna tertentu.

Nilai kesesuaian ini diatur dalam rentang [0,1] menggunakan fungsi aktivasi _sigmoid_, menjadikan output model dapat diinterpretasikan sebagai probabilitas bahwa pengguna akan menyukai film tersebut. Hal ini sangat berguna dalam sistem rekomendasi karena memberikan dasar yang kuat untuk memilih film yang paling mungkin disukai oleh pengguna.

Penggunaan _Neural Network_ dalam proses ini sangat krusial karena memungkinkan model untuk mempelajari dan memahami pola kompleks dari interaksi pengguna dengan film. Dengan kemampuannya untuk menyesuaikan jutaan parameter selama proses pembelajaran, _Neural Network_ dapat menyesuaikan diri secara efektif dengan data dan memprediksi rating pengguna terhadap film dengan akurasi yang tinggi. Ini menghasilkan rekomendasi yang sangat personalisasi, berdasarkan pada preferensi pengguna yang dipelajari oleh model.

**Keunggulan**:

- Tidak membutuhkan detail atau atribut spesifik dari setiap item.
- Mampu memberikan rekomendasi meskipun menggunakan dataset yang tidak lengkap.
- Memiliki kelebihan dalam hal kecepatan dan kemampuan untuk ditingkatkan (skalabilitas).
- Dapat memberikan rekomendasi efektif bahkan ketika konten sulit untuk dianalisis.

**Keterbatasan**:

- Bergantung pada adanya data rating; sehingga, item baru tidak akan direkomendasikan oleh sistem sampai mendapatkan rating.

Berikut merupakan hasil rekomendasi untuk user 307

```
Rekomendasi untuk user 307

Movie with high ratings
- Army of Darkness (1993)
- American Psycho (2000)
- Reign Over Me (2007)
- Iron Man (2008)
- WALL·E (2008)

Top movies recommendation:
- Streetcar Named Desire, A (1951)
- Brazil (1985)
- 12 Angry Men (1957)
- Apocalypse Now (1979)
- Ran (1985)
- Grand Day Out with Wallace and Gromit, A (1989)
- Glory (1989)
- Seven Samurai (Shichinin no samurai) (1954)
- Guess Who's Coming to Dinner (1967)
- Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001)
```

# Evaluation

Metriks yang digunakan adalah _Mean Absolute Error_ (MAE) dan _Root Mean Squared Error_ (RMSE) pada _Collaborative Filtering_ dan _Precision_ dan _recall_ pada _Content Based Filtering_

### Content Based Filtering

Pada evaluasi model ini penulis menggunakan metrik _Precision_ dan _recall_.

Rumus _Precision_:
$$ \text{Precision} = \frac{\text{rekomendasi yang relevan}}{\text{item yang direkomendasikan}} $$

Rumus _Recall_:
$$ \text{Recall} = \frac{\text{rekomendasi yang relevan}}{\text{semua item relevan yang mungkin}} $$

_Sample_ yang digunakan adalah **WALL·E (2008)** dan film yang relevan yang dipilih antara lain, Titan A.E. (2000), Transformers: The Movie (1986), Chicken Little (2005), Toy Story 3 (2010), BURN-E (2008), Ratchet & Clank (2016), Meet the Robinsons (2007), dan Lilo & Stitch (2002). Dengan item yang direkomendasikan ada 10.

Hasilnya adalah

tabel 4. Metrik evaluasi model _Content Based Filtering_
| Metrik | Value |
|--------------|-------|
| Precision@10 | 0.5 |
| Recall@10 | 0.625 |

`Precision@10` mengukur persentase rekomendasi yang relevan dari 10 rekomendasi teratas yang diberikan kepada pengguna. Nilai 0.5 berarti bahwa dari 10 rekomendasi yang diberikan, hanya 5 yang relevan atau sesuai dengan apa yang diharapkan oleh pengguna.

Sedangkan `Recall@10` mengukur seberapa lengkap rekomendasi yang relevan berhasil diambil dari keseluruhan jumlah item yang relevan. Nilai 0.625 menunjukkan bahwa dari keseluruhan item yang relevan, sistem berhasil merekomendasikan 62.5% di antaranya dalam 10 rekomendasi teratas.

Kesamaan yang mendasari kedua metrik tersebut adalah fokus pada relevansi item yang direkomendasikan. Kedua metrik ini memberikan gambaran tentang efektivitas sistem dalam merekomendasikan item yang sesuai (_precision_) dan lengkap (_recall_) terhadap kebutuhan pengguna.

### Collaborative Filtering

metrik MAE (Mean Absolute Error) dan RMSE (Root Mean Square Error) yang sering digunakan untuk mengevaluasi model prediksi, terutama dalam sistem rekomendasi dan model regresi.

`Mean Absolute Error (MAE)`

MAE merupakan rata-rata dari nilai absolut dari kesalahan (error) antara prediksi dan nilai sebenarnya. MAE memberikan ide tentang seberapa besar kesalahan yang dibuat oleh model dalam prediksi, tanpa memperhatikan arah kesalahannya. Rumus untuk MAE adalah:

$$ \text{MAE} = \frac{1}{n} \sum\_{i=1}^{n} |y_i - \hat{y}\_i| $$

di mana:
$$n \text{ adalah jumlah sampel,}$$
$$y_i\text{ adalah nilai sebenarnya,}$$
$$\hat{y}_i\text{ adalah nilai prediksi.}$$

`Root Mean Square Error (RMSE)`:

RMSE merupakan akar kuadrat dari rata-rata kuadrat kesalahan. RMSE memberikan penekanan lebih pada kesalahan yang lebih besar, karena kesalahan tersebut dikuadratkan, yang membuat model sangat sensitif terhadap _outlier_. Rumus untuk RMSE adalah:

$$ \text{RMSE} = \sqrt{\frac{1}{n} \sum\_{i=1}^{n} (y_i - \hat{y}\_i)^2} $$

di mana:
$$n \text{ adalah jumlah sampel,}$$
$$y_i\text{ adalah nilai sebenarnya,}$$
$$\hat{y}_i\text{ adalah nilai prediksi.}$$
Kedua metrik ini sering digunakan bersama karena mereka memberikan perspektif yang berbeda terhadap kesalahan model. MAE memberikan gambaran umum tentang besarnya kesalahan tanpa mempertimbangkan arahnya, sedangkan RMSE memberikan penekanan lebih pada kesalahan besar, yang bisa sangat penting tergantung pada aplikasi model Anda.

#### Hasil

##### MAE

![MAE](https://raw.githubusercontent.com/mochavin/ML-system-recommender/main/images/mae.png)
Gambar 5. Plot MAE _Collaborative Filtering_

metriks MAE konvergen di sekitar 0.1300 untuk training dan 0.14250 untuk validasi

##### RMSE

![RMSE](https://raw.githubusercontent.com/mochavin/ML-system-recommender/main/images/rmse.png)
Gambar 6. Plot RMSE _Collaborative Filtering_

metriks RMSE konvergen di sekitar 0.1700 untuk training dan 0.1850 untuk validasi

Dalam 16 epoch metriks MAE dan RMSE hasilnya cukup memuaskan dengan tingkat error yang relatif kecil. _Nilai Mean Absolute Error_ (MAE) dan _Root Mean Square Error_ (RMSE) yang masing-masing berkisar pada 0.1300 dan 0.1700 untuk data latihan, serta 0.14250 dan 0.1850 untuk data validasi, menunjukkan bahwa model memiliki tingkat kesalahan yang relatif rendah dalam melakukan prediksi. Kedua metrik ini, MAE dan RMSE, memberikan _insight_ tentang seberapa jauh prediksi model menyimpang dari nilai sebenarnya, dengan nilai-nilai di bawah 0.2 (atau 20%) secara umum dianggap menunjukkan performa yang baik. Hal ini menunjukkan bahwa model cukup akurat dan konsisten dalam melakukan prediksi, baik pada data latihan maupun validasi, tanpa adanya indikasi _overfitting_ yang signifikan.

Berdasarkan latar belakang dan tujuan yang telah ditetapkan dalam proyek ini, implementasi dari _content-based filtering_ dan _collaborative filtering_ telah berhasil dilakukan dengan memanfaatkan proses pengolahan data yang tepat. Data yang telah diolah dengan baik memungkinkan pembangunan model sistem rekomendasi yang berkualitas dan akurat. Dengan penerapan kedua teknik tersebut, model berhasil mengembangkan sistem rekomendasi yang mampu menganalisis pola dan preferensi pengguna secara efektif, serta memberikan rekomendasi film yang relevan.

Hasil evaluasi menunjukkan bahwa model sistem rekomendasi yang dikembangkan berhasil mencapai tingkat error di bawah 20%, sesuai dengan salah satu tujuan utama proyek ini. Hal ini menandakan bahwa model _machine learning_ yang dikembangkan telah berhasil merekomendasikan film dengan tingkat akurasi yang tinggi, sesuai dengan preferensi pengguna. Kedua metode yang digunakan, yaitu _content-based filtering_ dan _collaborative filtering_, memiliki peranan penting dalam mencapai hasil ini, dengan masing-masing memberikan kontribusi dalam memahami preferensi pengguna dari sudut pandang yang berbeda.

> Kesimpulannya, proyek ini telah berhasil menyelesaikan permasalahan yang diangkat pada latar belakang. Proses pengolahan data yang baik dan pengembangan model _machine learning_ dengan pendekatan yang tepat telah memungkinkan pembuatan sistem rekomendasi yang tidak hanya berkualitas dan akurat, tetapi juga relevan dengan kebutuhan pengguna. Ini membuktikan efektivitas kombinasi antara _content-based filtering_ dan _collaborative filtering_ dalam membangun sistem rekomendasi yang mampu memenuhi tujuan proyek, yaitu memberikan rekomendasi film yang relevan dengan tingkat error di bawah 20%.

# REFERENCES

[1] Walek, B., & Fojtik, V. (2020). A hybrid recommender system for recommending relevant movies using an expert system. Expert Systems with Applications, 158, 113452.

[2] Ekstrand, M. D., Riedl, J. T., & Konstan, J. A. (2011). Collaborative filtering recommender systems. Foundations and Trends® in Human–Computer Interaction, 4(2), 81-173.

[3] Anwar, T., & Uma, V. (2021). Comparative study of recommender system approaches and movie recommendation using collaborative filtering. International Journal of System Assurance Engineering and Management, 12, 426-436.
