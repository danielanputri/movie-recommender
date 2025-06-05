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

Dataset ini mengandung 100836 entri dan 3 kolom

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
  ![Rating Plot](https://github.com/danielanputri/recommenderSystem/blob/main/images/rating-plot.png)
  - Rating Paling Umum adalah 4.0: Frekuensi tertinggi ada pada skor rating 4.0.
  - Rating Tinggi Mendominasi: Skor rating 3.0, 3.5, 4.0, 4.5, dan 5.0 secara kolektif memiliki frekuensi yang jauh lebih tinggi dibandingkan skor rendah.
  - Skor di bawah 2.5 (terutama 0.5, 1.0, 1.5) memiliki frekuensi yang sangat rendah, mengindikasikan pengguna jarang memberikan penilaian sangat negatif.
    
- Movie User Plot
  ![Movie User Plot](https://github.com/danielanputri/recommenderSystem/blob/main/images/user-plot.png)
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
- Pada tahap ini genre dari setiap movie pada dataframe **movies.csv** akan diubah menjadi bentuk array (list) dan menghapus pipe (|). Hal ini dilakukan untuk mempermudah akses ke genre di kolom "genre". Hasilnya sebagai berikut
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

---
## Modeling
Pada tahap ini akan membahas dua pendekatan utama yang digunakan dalam membangun sistem rekomendasi: Content-Based Filtering dan Collaborative Filtering. Berikut adalah penjelasan lebih lanjut mengenai parameter yang digunakan, kelebihan, dan kekurangan dari masing-masing pendekatan, serta beberapa potongan kode yang relevan.

**Model Sistem Rekomendasi Content Based Filtering**

Content-Based Filtering menggunakan deskripsi dan fitur dari item itu sendiri untuk memberikan rekomendasi. Berikut adalah parameter untuk pendekatan ini.

Parameter yang Digunakan:
  - TF-IDF Vectorizer: Untuk mengubah deskripsi teks menjadi vektor numerik.
  - Cosine Similarity: Untuk menghitung kesamaan antara vektor item.

Formula untuk **Cosine Similarity** adalah:  
$\displaystyle cos~(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$

Teknik ini menggunakan model **TF-IDF Vectorizer** untuk mendapatkan informasi mengenai genre yang terdapat di setiap movie dan diubah menjadi fitur yang dapat diukur kemiripannya. Contohnya adalah sebagai berikut
| Title                                      | War | Drama    | Action | Documentary | Crime | Adventure | Fantasy  | Mystery | Fi       | Film |
|--------------------------------------------|-----|----------|--------|-------------|-------|-----------|----------|---------|----------|------|
| The Dark Tower (2017)                      | 0.0 | 0.000000 | 0.0    | 0.0         | 0.0   | 0.0       | 0.419398 | 0.0     | 0.392092 | 0.0  |
| National Lampoon's Van Wilder (2002)       | 0.0 | 0.000000 | 0.0    | 0.0         | 0.0   | 0.0       | 0.000000 | 0.0     | 0.000000 | 0.0  |
| Harold and Maude (1971)                    | 0.0 | 0.466216 | 0.0    | 0.0         | 0.0   | 0.0       | 0.000000 | 0.0     | 0.000000 | 0.0  |
| Mayor of the Sunset Strip (2003)           | 0.0 | 0.000000 | 0.0    | 1.0         | 0.0   | 0.0       | 0.000000 | 0.0     | 0.000000 | 0.0  |
| Seven Year Itch, The (1955)                | 0.0 | 0.000000 | 0.0    | 0.0         | 0.0   | 0.0       | 0.000000 | 0.0     | 0.000000 | 0.0  |

Pada baris dan kolom yang memiliki angka lebih dari 0 menunjukan genre yang ada pada movie tersebut.

Setelah itu **Cosine Similarity** akan diterapkan pada dataframe movie yang telah dibersihkan sehingga menghasilkan output sebagai berikut:
| Title                             | Sitter, The (2011) | Great Yokai War, The (2005) | Austin Powers: The Spy Who Shagged Me (1999) | Eros (2004) | Navy Seals (1990) | Glen or Glenda (1953) | Over the Top (1987) | Incredible Journey, The (1963) | All the Real Girls (2003) | Bliss (2012) |
|----------------------------------|---------------------|-----------------------------|----------------------------------------------|-------------|--------------------|-------------------------|----------------------|----------------------------------|----------------------------|--------------|
| They Call Me Trinity (1971)      | 0.359701            | 0.116405                    | 0.156174                                     | 0.000000    | 0.000000           | 0.000000                | 0.000000             | 0.000000                         | 0.000000                   | 0.000000     |
| Joe Gould's Secret (2000)        | 0.000000            | 0.000000                    | 0.000000                                     | 1.000000    | 0.000000           | 1.000000                | 0.559122             | 0.000000                         | 0.540111                   | 1.000000     |
| Robin and Marian (1976)          | 0.000000            | 0.339874                    | 0.455992                                     | 0.399120    | 0.349708           | 0.399120                | 0.223157             | 0.428926                         | 0.738959                   | 0.399120     |
| Portrait of a Lady, The (1996)   | 0.000000            | 0.000000                    | 0.000000                                     | 1.000000    | 0.000000           | 1.000000                | 0.559122             | 0.000000                         | 0.540111                   | 1.000000     |
| Stop-Loss (2008)                 | 0.000000            | 0.000000                    | 0.000000                                     | 0.391360    | 0.665323           | 0.391360                | 0.218818             | 0.000000                         | 0.211378                   | 0.391360     |
| Herbie Rides Again (1974)        | 0.317838            | 0.438499                    | 0.137998                                     | 0.000000    | 0.000000           | 0.000000                | 0.000000             | 0.462827                         | 0.384876                   | 0.000000     |
| Curly Sue (1991)                 | 0.734682            | 0.237754                    | 0.318983                                     | 0.678412    | 0.000000           | 0.678412                | 0.379315             | 0.000000                         | 0.366418                   | 0.678412     |
| House Party 3 (1994)             | 1.000000            | 0.323615                    | 0.434179                                     | 0.000000    | 0.000000           | 0.000000                | 0.000000             | 0.000000                         | 0.000000                   | 0.000000     |
| Sibling Rivalry (1990)           | 1.000000            | 0.323615                    | 0.434179                                     | 0.000000    | 0.000000           | 0.000000                | 0.000000             | 0.000000                         | 0.000000                   | 0.000000     |
| Assault on Precinct 13 (1976)    | 0.000000            | 0.000000                    | 0.423179                                     | 0.000000    | 0.324542           | 0.000000                | 0.590158             | 0.000000                         | 0.000000                   | 0.000000     |

Di tabel tersebut dapat dilihat kecocokan dari 1 movie dengan yang lain. Nilai-nilai pada tabel tersebut merepresentasikan persentase kecocokan antara kedua movie tersebut.

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
![result](https://github.com/danielanputri/recommenderSystem/blob/main/images/train%20vs%20test.png)

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
Showing recommendations for user: 419
========================================
Movie with high ratings from user
----------------------------------------
Pulp Fiction (1994) : Comedy, Crime, Drama, Thriller
Battle Royale 2: Requiem (Batoru rowaiaru II: Chinkonka) (2003) : Action, Drama, Thriller, War
Prestige, The (2006) : Drama, Mystery, Sci-Fi, Thriller
Paranormal Activity 2 (2010) : Horror, IMAX
Louis Theroux: Law & Disorder (2008) : Documentary
----------------------------------------
```
Kemudian akan diambil semua movie yang belum dilihat oleh user, lalu model akan melakukan prediksi berdasarkan movie dengan rating tinggi oleh user dan kemiripannya dengan user lain. Hasilnya akan mendapatkan rekomendasi sebagai berikut

```
Top 10 movie recommendations
----------------------------------------
Blade Runner (1982) : Action, Sci-Fi, Thriller
Life Is Beautiful (La Vita è bella) (1997) : Comedy, Drama, Romance, War
Runaway Train (1985) : Action, Adventure, Drama, Thriller
Big Chill, The (1983) : Comedy, Drama
Bug's Life, A (1998) : Adventure, Animation, Children, Comedy
Central Station (Central do Brasil) (1998) : Drama
Celebration, The (Festen) (1998) : Drama
Others, The (2001) : Drama, Horror, Mystery, Thriller
Superbad (2007) : Comedy
Everything Must Go (2010) : Comedy, Drama
```
Hasil Collaborative Filtering:
![result_collab](https://github.com/danielanputri/recommenderSystem/blob/main/images/train%20vs%20test.png)

#### Kelebihan dan Kekurangan Content-Based Filtering

- Kelebihan:
  1. Menggunakan Data Pengguna: Memanfaatkan interaksi pengguna-item sehingga dapat merekomendasikan item yang tidak mirip tetapi disukai oleh pengguna dengan preferensi serupa.
  2. Menangani Data Besar: Dapat bekerja dengan data besar dan menemukan pola-pola kompleks.
 
- Kekurangan:
  1. Cold Start Problem: Kesulitan merekomendasikan item baru atau kepada pengguna baru yang belum memiliki cukup interaksi.

---
## Evaluation
Pada bagian ini, akan mengevaluasi model rekomendasi yang telah dibangun menggunakan metrik evaluasi yang tepat. Untuk model prediksi rating, kita akan menggunakan Mean Absolute Error (MAE) sebagai metrik evaluasi. Selain itu, akan mengevaluasi apakah proyek ini berhasil menjawab problem statement dan memberikan solusi yang diinginkan.

**Metrik Evaluasi**

MAE atau Mean Absolute Error diterapkan dengan cara mengukur rata-rata dari selisih absolut antara prediksi dan nilai asli (y_asli - y_prediksi).

MAE = $\displaystyle \sum\frac{|y_i - \hat{y}_i|}{n}$

Dimana:
MAE = nilai Mean Absolute Error
y = nilai aktual
ŷ = nilai prediksi
i = urutan data
n = jumlah data

Berikut plot MAE dari model:
![Grafik train vs test](https://github.com/danielanputri/recommenderSystem/blob/main/images/train%20vs%20test.png)

Berikut plot MAE dari model:
![Grafik train vs test](https://github.com/danielanputri/recommenderSystem/blob/main/images/train%20vs%20test.png)

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
