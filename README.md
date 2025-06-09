# Laporan Proyek Machine Learning - Daniela Natali Putri

## Project Overview

**Latar Belakang Proyek**

Seiring dengan perkembangan dunia digital saat ini, banyak sekali informasi dan hiburan yang beredar. Salah satunya adalah movie. Dengan begitu banyaknya movie yang diproduksi, calon penonton sering kali mengalami kesulitan dalam menentukan movie mana yang ingin ditonton. Proses pencarian movie bisa memakan waktu, dan movie yang akhirnya dipilih belum tentu sesuai dengan selera penonton setelah ditonton. Akibatnya, waktu yang dihabiskan menjadi lebih banyak. Selain itu, menonton movie melalui bioskop, platform streaming, atau media fisik seperti DVD juga memerlukan biaya. Jika movie yang ditonton ternyata tidak memuaskan, waktu dan biaya yang dikeluarkan pun akan terbuang sia-sia [1].

Mereka yang kesulitan memilih movie untuk ditonton sering kali mencari di aplikasi twitter atau mengunjungi situs seperti suggestmemovie.com yang memberikan rekomendasi movie kepada pengguna. Namun, dari berbagai solusi tersebut, banyak pengguna mengaku masih harus mencoba beberapa kali sebelum menemukan movie yang dianggap bagus.

Sistem rekomendasi adalah alat untuk berinteraksi dengan ruang informasi yang besar dan kompleks [2]. Oleh karena itu, penting untuk memiliki sistem rekomendasi yang dapat menyederhanakan proses ini dengan memberikan rekomendasi yang relevan berdasarkan kebutuhan dan preferensi pengguna. Dengan memanfaatkan data rating pengguna sebelumnya, sistem rekomendasi ini diharapkan dapat membantu pengguna menemukan movie yang paling sesuai dengan kebutuhan mereka.

**Pentingnya Proyek**

Proyek ini penting karena: 
- Peningkatan Pengalaman Pengguna: Membantu pengguna menemukan movie yang sesuai dengan preferensi mereka, meningkatkan kepuasan dan pengalaman pengguna.
- Efisiensi: Mengurangi waktu dan usaha yang dibutuhkan pengguna dalam mencari movie.
- Personalisasi: Memberikan rekomendasi yang dipersonalisasi berdasarkan data rating pengguna.

---
## Business Understanding

### Problem Statements
- Bagaimana kita bisa membantu pengguna mendapatkan movie dengan genre paling sesuai dengan preferensi mereka?
- Bagaimana kita bisa membantu pengguna menemukan movie yang mirip dengan movie yang pernah dirating sebelumnya?

### Goals
Menjelaskan tujuan proyek yang menjawab pernyataan masalah:
- Mengembangkan sistem rekomendasi yang dapat memberikan daftar movie terbaik berdasarkan genre atau tema dari movie yang disukai.
- Membangun sistem rekomendasi yang dapat memberikan daftar movie terbaik berdasarkan movie yang pernah dirating sebelumnya.

### Solution statements
- Menggunakan teknik **Content-Based Filtering** untuk mendapatkan movie dengan genre yang mirip dengan movie yang disukai.
- Menggunakan teknik **Collaborative Filtering** untuk mendapatkan rekomendasi movie berdasarkan movie yang pernah dirating.

## Data Understanding
Dataset ini dapat diunduh dari [kaggle](https://www.kaggle.com/datasets/aigamer/movie-lens-dataset). Dataset terbagi menjadi 4 yaitu links, movies, ratings, dan tags. Namun disini hanya menggunakan dataset movies dan ratings saja.

### Info Data movies.csv

|   # | Column   | Non-Null Count | Dtype   |
| --: | -------- | -------------- | ------- |
|   0 | movieId | 9742 non-null | int64   |
|   1 | title     | 9742 non-null | object  |
|   2 | genres    | 9742 non-null | object  |

Seperti yang terlihat di kolom **"movieId"** terdapat sampel sejumlah 9742, karena kolom **"movieId"** berisi id yang unik maka dapat disimpulkan bahwa panjang dataframe **movies.csv** adalah **9742**

Selanjutnya lihat contoh data dalam **movies.csv**
| movieId | title                   | genres                                    |
|---------|-------------------------|-------------------------------------------|
| 1       | Toy Story (1995)        | Adventure|Animation|Children|Comedy|Fantasy|
| 2       | Jumanji (1995)          | Adventure|Children|Fantasy              |
| 3       | Grumpier Old Men (1995) | Comedy|Romance                           |

Dalam kolom **"genres",** menggunakan pipe (|) sebagai pemisah. Hal ini akan membuat model machine learning lebih sulit dalam mengidentifikasi masing masing genre yang terdapat dalam suatu movie atau movie.

Karena hal tersebut, mengetahui genre apa saja yang ada dalam dataframe tersebut akan sulit. Dataframe ini perlu dibersihkan terlebih dahulu di tahap [Data Preparation](#data-preparation)

### Info Data rating.csv

Dataset ini mengandung 100836 entri dan 4 kolom

|   # | Column   | Dtype |
| --: | -------- | ----- |
|   0 | userId  | int64 |
|   1 | movieId | int64 |
|   2 | rating   | int64 |
|   3 | timestamp | int64 |


Variabel-variabel pada dataset adalah sebagai berikut:
- movies:
  - `movieId`: ID unik untuk setiap movie.
  - `title`: Judul movie.
  - `genres`: Genre dari movie.

- ratings:
  - `userId`: ID unik untuk setiap pengguna.
  - `movieId`: ID unik untuk setiap movie (mengacu pada movies.csv).
  - `rating`: Rating yang diberikan pengguna untuk movie tertentu (skala 0.5 - 5.0).
  - `timestamp`: Menunjukkan detik sejak tengah malam Waktu Universal Terkoordinasi (UTC) tanggal 1 Januari 1970.

---
**Exploratory Data Analysis (EDA)**
- Rating Plot
  - ![rating-plot](https://github.com/danielanputri/recommenderSystem/blob/main/images/rating-plot.png)
  - Rating Paling Umum adalah 4.0: Frekuensi tertinggi ada pada skor rating 4.0.
  - Rating Tinggi Mendominasi: Skor rating 3.0, 3.5, 4.0, 4.5, dan 5.0 secara kolektif memiliki frekuensi yang jauh lebih tinggi dibandingkan skor rendah.
  - Skor di bawah 2.5 (terutama 0.5, 1.0, 1.5) memiliki frekuensi yang sangat rendah, mengindikasikan pengguna jarang memberikan penilaian sangat negatif.
    
- Movie User Plot
  - ![user-plot](https://github.com/danielanputri/recommenderSystem/blob/main/images/user-plot.png)
  - Terdapat perbedaan yang sangat signifikan antara jumlah movie (sekitar 9600-9800) dengan jumlah pengguna (sekitar 600-700) dalam dataset.
  - Plot menunjukkan ketidakseimbangan yang besar, di mana jumlah item (movie) yang tersedia jauh mendominasi jumlah pengguna yang ada.

---
## Data Preparation
Tahap data preparation dipisah menjai 2 yaitu untuk dataframe movie dan dataframe rating.

### Movie Data Preparation
**Teknik Data Preparation**
- Konversi genre dari setiap movie menjadi list.
- Cek missing value pada dataframe
- Cek genre unik
- Menghapus rows genre yang tidak diperlukan.
- Mengubah list genre menjadi string.

#### 1. Konversi genre dari setiap movie menjadi list.
Pada tahap ini genre dari setiap movie pada dataframe **movies.csv** akan diubah menjadi bentuk array (list) dan menghapus pipe (|). Hal ini dilakukan untuk mempermudah akses ke genre di kolom "genre". Hasilnya sebagai berikut:

| movieId | title                   | genres                                           | genre_str                            |
|---------|-------------------------|--------------------------------------------------|--------------------------------------|
| 0       | Toy Story (1995)        | [Adventure, Animation, Children, Comedy, Fantasy]  | Adventure Animation Children Comedy Fantasy |
| 1       | Jumanji (1995)          | [Adventure, Children, Fantasy]                   | Adventure Children Fantasy           |
| 2       | Grumpier Old Men (1995) | [Comedy, Romance]                                | Comedy Romance                       |

#### 2. Cek missing value pada dataframe.
- Pada tahap ini tidak ada missing value pada dataframe.

#### 3. Cek genre unik.
- Tahap ini seharusnya dilakukan di bagian [Data Understanding](#data-understanding) namun karena kedua tahap diatas perlu dijalankan terlebih dahulu sebelum bisa mengidentifikasi tiap genre yang terdapat dalam dataframe movie.
- Genre yang didapat adalah:
```
Total # of genre:  20
List of all genre availabel:  ['Adventure' 'Animation' 'Children' 'Comedy' 'Fantasy' 'Romance' 'Drama'
 'Action' 'Crime' 'Thriller' 'Horror' 'Mystery' 'Sci-Fi' 'War' 'Musical'
 'Documentary' 'IMAX' 'Western' 'movie-Noir' '(no genres listed)']
```
#### 4. Drop baris yang tidak digunakan.
- Pada tahap ini baris (no genres listed) akan dihapus karena tidak memiliki fitur konten yang bisa dibandingkan dengan movie lain yang mana idak berguna dalam pendekatan **Content-Based Filtering**.

#### 5. Mengubah list genre menjadi string.
- Pada tahap ini list genre diubah lagi menjadi string dengan spasi sebagai pemisah dan dimasukkan ke kolom baru, sudah dilakukan pencegahan untuk genre 2 kata seperti "Toy Story" ketika diubah menjadi string maka spasi akan dihilangkan menjadi "ToyStory". Hal ini dilakukan untuk mempermudah **TF-IDF Vectorizer** dalam mendapatkan fitur genre.

### Rating Data Preparation
**Teknik Data Preparation**
- Cek missing value.
- Menghapus kolom yang tidak diperlukan
- Encoding userId dan movieId.
- Split menjadi data train dan validasi.

#### 1. Cek missing value pada dataframe.
- Pada tahap ini tidak ada missing value pada dataframe.

#### 2. Menghapus kolom yang tidak diperlukan
- Pada tahap ini, menghapus kolom timestamp karena tidak dibutuhkan. Hal ini bertujuan untuk pelatihan model yang lebih efektif.

#### 3. Encode userId dan movieId.
- Pada tahap ini dilakukan proses encoding pada kolom userId dan movieId dan dimasukkan ke kolom baru masing-masing. Hal ini dilakukan untuk merepresentasikan id user dan movie dalam format yang dapat di proses oleh model machine learning.

#### 4. Split menjadi data train dan validasi.
- Melakukan pemisahan pada dataframe menjadi train dan validasi dengan rasio 80:20, namun data diacak terlebih dahulu sebelum di pisah. Hal ini dilakukan supaya model dapat melakukan evaluasi pada data baru dan mencegah overfitting. Setelah diacak, fitur user dan movie dari dataframe rating diekstrasi. Lalu melakukan normalisasi min-max pada kolom 'rating' untuk mengubah nilainya ke dalam rentang 0 hingga 1.

#### 5. TF-IDF Vectorizer
Untuk merepresentasikan data kategori genre film ke dalam bentuk numerik yang dapat digunakan dalam perhitungan machine learning, dilakukan proses ekstraksi fitur berbasis teks menggunakan teknik TF-IDF Vectorizer (Term Frequency-Inverse Document Frequency). Teknik ini sangat umum digunakan dalam analisis teks untuk mengukur seberapa penting suatu kata (term) dalam suatu dokumen relatif terhadap kumpulan dokumen lainnya.

```
# Inisialisasi TF-IDF Vectorizer
tfv = TfidfVectorizer()

# Melatih TF-IDF berdasarkan genre film yang sudah digabung per baris (satu string per film)
tfv.fit(data['genre_str'])

# Menampilkan daftar fitur (kata unik dari seluruh genre)
tfv.get_feature_names_out()

# Transformasi teks menjadi matriks TF-IDF
tfidf_matrix = tfv.fit_transform(data['genre_str'])

# Ukuran matriks TF-IDF (baris = film, kolom = kata unik genre)
tfidf_matrix.shape
```
- Input: data`[genre_str]` adalah kolom yang berisi genre-genre film yang telah digabungkan menjadi satu string, misalnya "Action Adventure Comedy".
- Output: `tfidf_matrix` adalah matriks berdimensi (jumlah_film, jumlah_kata_unik_genre), di mana setiap sel menunjukkan skor TF-IDF dari suatu genre pada suatu film.

---
## Modeling

**Model Sistem Rekomendasi Content Based Filtering**

Content-Based Filtering menggunakan deskripsi dan fitur dari item itu sendiri untuk memberikan rekomendasi. Berikut adalah parameter untuk pendekatan ini.

Parameter yang Digunakan:
  - TF-IDF Vectorizer: Untuk mengubah deskripsi teks menjadi vektor numerik.
  - Cosine Similarity: Untuk menghitung kesamaan antara vektor item.

Teknik ini menggunakan model **TF-IDF Vectorizer** yang berada di bagian data preparation untuk mendapatkan informasi mengenai genre yang terdapat di setiap movie dan diubah menjadi fitur yang dapat diukur kemiripannya. Contoh hasilnya adalah sebagai berikut:
| Title                                          | Western | Comedy   | Sci     | Mystery | War | Documentary | Animation | Musical | IMAX | Noir |
|-----------------------------------------------|---------|----------|---------|---------|-----|-------------|-----------|---------|------|------|
| Naked (1993)                                   | 0.0     | 0.000000 | 0.000000| 0.0     | 0.0 | 0.0         | 0.0       | 0.0     | 0.0  | 0.0  |
| Police Academy 2: Their First Assignment (1985)| 0.0     | 0.533483 | 0.000000| 0.0     | 0.0 | 0.0         | 0.0       | 0.0     | 0.0  | 0.0  |
| Batman Returns (1992)                          | 0.0     | 0.000000 | 0.000000| 0.0     | 0.0 | 0.0         | 0.0       | 0.0     | 0.0  | 0.0  |
| Defending Your Life (1991)                     | 0.0     | 0.373037 | 0.000000| 0.0     | 0.0 | 0.0         | 0.0       | 0.0     | 0.0  | 0.0  |
| Doom (2005)                                    | 0.0     | 0.000000 | 0.522808| 0.0     | 0.0 | 0.0         | 0.0       | 0.0     | 0.0  | 0.0  |

Setiap baris merepresentasikan satu film, sedangkan setiap kolom mewakili bobot genre tertentu. Matriks ini kemudian digunakan sebagai dasar untuk mengukur kemiripan antar film.

Setelah representasi numerik dari genre diperoleh dalam bentuk matriks TF-IDF, tahap berikutnya adalah menghitung kemiripan antar film. Algoritma yang digunakan adalah **Cosine Similarity**, yaitu metode pengukuran kesamaan antar dua vektor berdasarkan sudut antar vektor tersebut.

Formula untuk **Cosine Similarity** adalah:  
$\displaystyle cos~(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$

Nilai cosine similarity berada pada rentang 0 hingga 1, di mana:
- Nilai 1 berarti kedua film memiliki genre yang identik atau sangat mirip,
- Nilai 0 berarti kedua film tidak memiliki kesamaan genre sama sekali.

Berikut merupakan hasil penerapan **Cosine Similarity** pada dataframe, yang menghasilkan output sebagai berikut:
| Title                                         | World's Fastest Indian, The (2005) | Philadelphia Experiment, The (1984) | Muppets From Space (1999) | Punisher, The (2004) | Breakfast Club, The (1985) | Invasion of the Body Snatchers (1956) | Believer, The (2001) | Rocket Singh: Salesman of the Year (2009) | Angst (1983) | Dragonslayer (1981) |
|----------------------------------------------|-------------------------------------|--------------------------------------|----------------------------|------------------------|-----------------------------|----------------------------------------|-----------------------|-------------------------------------------|--------------|----------------------|
| Sisterhood of the Traveling Pants, The (2005)| 0.446214                            | 0.529124                             | 0.226155                   | 0.000000               | 0.657734                    | 0.000000                               | 0.446214              | 0.657734                                  | 0.213968     | 0.426816             |
| Twelve Monkeys (a.k.a. 12 Monkeys) (1995)    | 0.000000                            | 0.563911                             | 0.000000                   | 0.217017               | 0.000000                    | 0.692438                               | 0.000000              | 0.000000                                  | 0.000000     | 0.000000             |
| Tampopo (1985)                                | 0.000000                            | 0.000000                             | 0.468011                   | 0.000000               | 0.734682                    | 0.000000                               | 0.000000              | 0.734682                                  | 0.000000     | 0.000000             |
| Coriolanus (2011)                              | 0.564262                            | 0.173810                             | 0.000000                   | 0.447470               | 0.382802                    | 0.346130                               | 0.564262              | 0.382802                                  | 0.270574     | 0.000000             |
| High and Low (Tengoku to jigoku) (1963)       | 0.195493                            | 0.060218                             | 0.000000                   | 0.368511               | 0.132625                    | 0.119919                               | 0.195493              | 0.132625                                  | 0.093742     | 0.000000             |
| Seeker: The Dark Is Rising, The (2007)        | 0.318216                            | 0.377343                             | 0.000000                   | 0.259190               | 0.215882                    | 0.000000                               | 0.318216              | 0.215882                                  | 0.152590     | 0.948018             |
| Hulk (2003)                                   | 0.000000                            | 0.857654                             | 0.000000                   | 0.237736               | 0.000000                    | 0.559491                               | 0.000000              | 0.000000                                  | 0.000000     | 0.494611             |
| Alien: Covenant (2017)                        | 0.000000                            | 0.543434                             | 0.000000                   | 0.423940               | 0.000000                    | 0.920367                               | 0.000000              | 0.000000                                  | 0.423513     | 0.194642             |
| Ride Along 2 (2016)                           | 0.000000                            | 0.000000                             | 0.276024                   | 0.443588               | 0.433301                    | 0.000000                               | 0.000000              | 0.433301                                  | 0.000000     | 0.401953             |
| Spellbound (1945)                             | 0.000000                            | 0.000000                             | 0.000000                   | 0.263009               | 0.000000                    | 0.203445                               | 0.000000              | 0.000000                                  | 0.000000     | 0.000000             |

Dari tabel di tersebut, dapat disimpulkan bahwa "Seeker: The Dark Is Rising, The (2007)" memiliki kemiripan sangat tinggi dengan "	Dragonslayer (1981)", ditunjukkan dengan nilai similarity sebesar 0.948018. Nilai-nilai inilah yang menjadi dasar dalam sistem rekomendasi: ketika seorang pengguna menyukai satu film, sistem akan merekomendasikan film lain yang memiliki nilai kemiripan tertinggi berdasarkan hasil cosine similarity.

Bagaimana Algoritma Bekerja:
- Content-Based Filtering menggunakan model dari item itu sendiri untuk memberikan rekomendasi. Algoritma ini bekerja dengan cara mengubah fitur deskriptif item (model) menjadi representasi numerik menggunakan TF-IDF Vectorizer. Kemudian, cosine similarity dihitung untuk menentukan seberapa mirip item-item tersebut berdasarkan vektor fitur mereka. Berdasarkan kemiripan ini, sistem dapat merekomendasikan item yang paling mirip dengan item yang sudah disukai pengguna.

**Top-N Recommendation Content Based Filtering**
Tabel tersebut adalah dataframe cosine similarity yang akan digunakan untuk mendapatkan top-N rekomendasi movie. Dalam kasus ini akan dicoba mendapatkan top-10 rekomendasi movie yang mirip dengan movie **"Avengers: Infinity War - Part I (2018)"**. Outputnya sebagai berikut
Data untuk uji coba:
| # | title | genres |
|--:|:------------------------------:|:--------------------------------:|
| 0 | Avengers: Infinity War - Part I (2018). | [Action, Adventure, Sci-Fi] |

Hasil rekomendasi:
| title                                               | genres                        |
|-----------------------------------------------------|-------------------------------|
| Rocketeer, The (1991)                               | [Action, Adventure, Sci-Fi]  |
| Ant-Man (2015)                                      | [Action, Adventure, Sci-Fi]  |
| Time Machine, The (2002)                            | [Action, Adventure, Sci-Fi]  |
| Iron Man (2008)                                     | [Action, Adventure, Sci-Fi]  |
| Sky Captain and the World of Tomorrow (2004)        | [Action, Adventure, Sci-Fi]  |
| Star Wars: Episode VI - Return of the Jedi (1983)   | [Action, Adventure, Sci-Fi]  |
| Justice League (2017)                               | [Action, Adventure, Sci-Fi]  |
| Farscape: The Peacekeeper Wars (2004)               | [Action, Adventure, Sci-Fi]  |
| Power/Rangers (2015)                                | [Action, Adventure, Sci-Fi]  |
| Black Panther (2017)                                | [Action, Adventure, Sci-Fi]  |

Berdasarkan hasil rekomendasi tersebut dapat dilihat bahwa movie yang direkomendasikan memiliki genre yang mirip dengan input movienya.

Hasil Content Based Filtering:
- ![result-content-based](https://github.com/danielanputri/movie-recommender/blob/main/images/result-content-based.png)

#### Kelebihan dan Kekurangan Content-Based Filtering

- Kelebihan:
  1. Tidak Perlu Data Pengguna Lain: Rekomendasi untuk seorang pengguna tidak bergantung pada data atau preferensi pengguna lain, hanya berdasarkan profil dan item yang disukainya sendiri.
  2. Mampu Merekomendasikan Item Baru: Selama item baru memiliki deskripsi fitur yang memadai, sistem dapat langsung merekomendasikannya tanpa perlu riwayat interaksi dari pengguna lain.
  3. Transparansi Rekomendasi: Alasan mengapa suatu item direkomendasikan bisa lebih mudah dijelaskan (misalnya, "karena Anda menyukai genre X, dan item ini juga bergenre X").

- Kekurangan:
  1. Keterbatasan Penemuan Hal Baru (Serendipity): Cenderung merekomendasikan item yang sangat mirip dengan yang sudah disukai pengguna, sehingga sulit menemukan variasi atau hal yang benar-benar baru.
  2. Sangat Bergantung pada Kualitas Fitur Item: Kualitas rekomendasi sangat dipengaruhi oleh seberapa baik dan lengkap fitur-fitur item diekstraksi dan direpresentasikan.
  3. Sulit Menangani Pengguna Baru (Cold Start User): Sistem kesulitan memberikan rekomendasi yang akurat untuk pengguna baru karena belum ada riwayat preferensi item untuk membangun profil pengguna.


**Model Sistem Rekomendasi Collaborative Filtering**

Collaborative Filtering menggunakan interaksi pengguna-item (rating) untuk memberikan rekomendasi. Berikut adalah parameter untuk pendekatan ini.

Proyek ini menggunakan model **RecommenderNet** yang dibuat dari kelas **Model** milik **Keras**.
Beberapa parameter yang Digunakan:
  - num_users: Jumlah pengguna (user) dalam dataset
  - num_movie: Jumlah film (movie) dalam dataset
  - embedding_size: `128`
  - loss = `tf.keras.losses.MeanSquaredError()`
  - optimizer = `keras.optimizers.Adam(learning_rate=0.001)`
  - learning_rate = `0.001`
  - metrics = [RootMeanSquaredError (rmse), MeanAbsoluteError (mae)]

Parameter Callbacks:
1. EarlyStoppingg
   - monitor = 'val_loss'
   - patience = 5
   - restore_best_weights = True
2. ModelCheckpoint
   - filepath = 'best_model.keras'
   - monitor = 'val_loss'
   - save_best_only = True
3. ReduceLROnPlateau
   - monitor = 'val_loss'
   - factor = 0.5
   - patience = 3
   - verbose = 1
   - min_lr = 1e-6

Bagaimana Algoritma Bekerja:
- Collaborative Filtering menggunakan interaksi pengguna-item (rating) untuk memberikan rekomendasi. Algoritma ini bekerja dengan cara memprediksi rating item yang belum diulas pengguna berdasarkan rating item yang mirip oleh pengguna lain. Model ini mempelajari pola preferensi pengguna dari data rating yang ada dan menggunakan pola tersebut untuk merekomendasikan item yang mungkin disukai pengguna.

**Top-N Recommendation Collaborative Filtering**
Pertama ambil dulu user secara acak dari dataframe rating. Kemudian tunjukkan movie yang telah dirating oleh user tersebut.

```
Showing recommendations for user: 610
========================================
Movie with high ratings from user
----------------------------------------
Heat (1995) : Action, Crime, Thriller
Down by Law (1986) : Comedy, Drama, Film-Noir
Suspiria (1977) : Horror
Hot Fuzz (2007) : Action, Comedy, Crime, Mystery
Avengers, The (2012) : Action, Adventure, Sci-Fi, IMAX
----------------------------------------
```
Kemudian akan diambil semua movie yang belum dilihat oleh user, lalu model akan melakukan prediksi berdasarkan movie dengan rating tinggi oleh user dan kemiripannya dengan user lain. Hasilnya akan mendapatkan rekomendasi sebagai berikut

```
Top 10 movie recommendations
----------------------------------------
It's a Wonderful Life (1946) : Children, Drama, Fantasy, Romance
Right Stuff, The (1983) : Drama
Boot, Das (Boat, The) (1981) : Action, Drama, War
Sting, The (1973) : Comedy, Crime
Manhattan (1979) : Comedy, Drama, Romance
Dead Poets Society (1989) : Drama
Touch of Evil (1958) : Crime, Film-Noir, Thriller
Chinatown (1974) : Crime, Film-Noir, Mystery, Thriller
Great Escape, The (1963) : Action, Adventure, Drama, War
Deer Hunter, The (1978) : Drama, War
```
Hasil Collaborative Filtering:
- ![result_collaborative](https://github.com/danielanputri/movie-recommender/blob/main/images/result-collaborative.png))

#### Kelebihan dan Kekurangan Content-Based Filtering

- Kelebihan:
  1. Menggunakan Data Pengguna: Memanfaatkan interaksi pengguna-item sehingga dapat merekomendasikan item yang tidak mirip tetapi disukai oleh pengguna dengan preferensi serupa.
  2. Menangani Data Besar: Dapat bekerja dengan data besar dan menemukan pola-pola kompleks.
 
- Kekurangan:
  1. Cold Start Problem: Kesulitan merekomendasikan item baru atau kepada pengguna baru yang belum memiliki cukup interaksi.

---
## Evaluation
Pada bagian ini, akan mengevaluasi model rekomendasi yang telah dibangun menggunakan metrik evaluasi yang tepat. Untuk model prediksi rating, kita akan menggunakan Mean Absolute Error (MAE) sebagai metrik evaluasi. Selain itu, akan mengevaluasi apakah proyek ini berhasil menjawab problem statement dan memberikan solusi yang diinginkan.


### Evaluasi Content-Based Filtering

Metrik evaluasi yang digunakan untuk **Content Based Filtering** adalah **Precision@K**.

Precision@K mengukur sejauh mana sistem memberikan rekomendasi yang relevan dalam K item teratas (top-K). Fokus dari metrik ini adalah menghindari item yang tidak relevan, sehingga cocok untuk sistem berbasis konten yang mengutamakan kesamaan karakteristik (seperti genre).

Formula dari Precision@K adalah:

Precision@K = $\displaystyle \frac{\text{Jumlah item yang relevan di top-K rekomendasi}}{\text{K}}$

Berikut analisa Precision@K untuk hasil rekomendasi **Content-Based Filtering**.

Data untuk uji coba
| # | title | genres |
|--:|:------------------------------:|:--------------------------------:|
| 0 | Avengers: Infinity War - Part I (2018). | [Action, Adventure, Sci-Fi] |

Hasil rekomendasi:
| title                                               | genres                        |
|-----------------------------------------------------|-------------------------------|
| Rocketeer, The (1991)                               | [Action, Adventure, Sci-Fi]  |
| Sky Captain and the World of Tomorrow (2004)        | [Action, Adventure, Sci-Fi]  |
| Star Wars: Episode VI - Return of the Jedi (1983)   | [Action, Adventure, Sci-Fi]  |
| Justice League (2017)                               | [Action, Adventure, Sci-Fi]  |
| Ant-Man (2015)                                      | [Action, Adventure, Sci-Fi]  |

Semua rekomendasi memiliki genre yang relevan (mengandung setidaknya satu genre yang sama dengan film input). Oleh karena itu, jumlah item relevan di top-5 adalah 5 dari 5 item yang direkomendasikan. Maka nilai dari Precision@K:

Precision@K = $\displaystyle \frac{\text{Jumlah item yang relevan di top-K rekomendasi}}{K} = \frac{5}{5} = 1.00 \text{ atau } 100%$

#### Evaluasi Collaborative Filtering**
Metrik evaluasi yang digunakan untuk **Collaborative Filtering** adalah **Mean Absolute Error (MAE)** dan **Root Mean Square Error (RMSE)**

MAE atau Mean Absolute Error diterapkan dengan cara mengukur rata-rata dari selisih absolut antara prediksi dan nilai asli (y_asli - y_prediksi).

MAE = $\displaystyle \sum\frac{|y_i - \hat{y}_i|}{n}$

Dimana:
- MAE = nilai Mean Absolute Error
- y = nilai aktual
- ŷ = nilai prediksi
- i = urutan data
- n = jumlah data

Root Mean Square Error (RMSE) adalah metrik yang digunakan untuk mengukur seberapa besar selisih (atau kesalahan) antara nilai yang diprediksi oleh sebuah model dengan nilai aktual. RMSE sangat populer karena memberikan bobot yang lebih besar pada kesalahan yang lebih besar.

Rumus untuk menghitung RMSE adalah sebagai berikut:

$$\text{RMSE} = \sqrt{\frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{n}}$$

**Keterangan:**

* $\text{RMSE}$ : Nilai *Root Mean Square Error*
* $n$ : Jumlah total data
* $y_i$ : Nilai aktual pada data ke-i
* $\hat{y}_i$ : Nilai prediksi pada data ke-i
* $\sum$ : Notasi sigma yang berarti penjumlahan semua selisih kuadrat dari i=1 hingga n

RMSE bernilai 0 jika prediksi model sempurna. Semakin kecil nilai RMSE, maka semakin baik performa model dalam melakukan prediksi.

Berikut plot MAE dari model:
- ![model_mae](https://github.com/danielanputri/movie-recommender/blob/main/images/model_mae.png)

Berikut plot RMSE dari model:
- ![model-rmse](https://github.com/danielanputri/movie-recommender/blob/main/images/model_rmse.png)

**Evaluasi Terhadap Business Understanding**
- Menjawab Problem Statement: Model yang dibuat berhasil menjawab problem statement dengan memberikan rekomendasi movie berdasarkan model yang ada. Pendekatan content-based filtering menggunakan model movie untuk memberikan rekomendasi yang relevan berdasarkan genre, sementara collaborative filtering memanfaatkan interaksi pengguna-item (rating) sebelumnya untuk menemukan pola preferensi pengguna.

- Mencapai Goals: Model content-based filtering dengan cosine similarity dan collaborative filtering dengan RecommenderNet berhasil mencapai tujuan untuk memberikan rekomendasi movie yang relevan.

---
## Kesimpulan
Dengan menggunakan kedua pendekatan ini, kita dapat membangun sistem rekomendasi yang lebih robust dan fleksibel. Content-Based Filtering cocok untuk memberikan rekomendasi berdasarkan fitur-fitur item itu sendiri, sementara Collaborative Filtering efektif dalam menemukan pola-pola preferensi pengguna dari data interaksi yang ada. Memahami kelebihan dan kekurangan masing-masing pendekatan membantu kita memilih metode yang paling sesuai dengan kebutuhan dan konteks spesifik dari sistem rekomendasi yang sedang dibangun. Namun proyek ini masih belum memberikan solusi untuk kasus **Cold Start**. Dimana user baru belum memiliki rating movie maupun jenis movie yang disukai.

---
## Referensi

[1] [Fajriansyah, M., Adikara, P. P., & Widodo, A. W. (2021). Sistem Rekomendasi movie Menggunakan Content Based Filtering. Jurnal Pengembangan Teknologi Informasi dan Ilmu Komputer, 5(6), 2188-2199.](https://j-ptiik.ub.ac.id/index.php/j-ptiik/article/view/9163)

[2] [R. Burke, A. Felfernig, and M. H. Göker, “Recommender Systems: An Overview”, AIMag, vol. 32, no. 3, pp. 13-18, Jun. 2011.](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2361) 
