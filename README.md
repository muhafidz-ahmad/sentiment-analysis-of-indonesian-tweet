# Project Overview
Model dideploy di [Streamlit](https://muhafidz-ahmad-sentiment-analysis-of-indon-streamlit-app-hwzuqv.streamlit.app/) untuk dapat dilakukan demo.

Twitter adalah sebuah sosial media dimana semua orang bisa membuat konten berupa tulisan dan membagikannya secara luas, dari konten edukatif hingga konten hiburan. Namun, tidak sedikit orang-orang membuat konten bersifat negatif terhadap sesuatu hal. Konten negatif tersebut dapat memicu kepada banyak hal negatif lainnya, bahkan bisa menular pada orang lain yang membaca konten tersebut.
* Diperlukannya sebuah filter yang dapat menyaring tweet yang memiliki konten yang negatif. Salah satu teknologi yang dapat digunakan untuk menyaring tweet tersebut adalah dengan machine learning.
* Telah banyak penelitian yang dilakukan dalam pembuatan model machine learning sentimen analysis ini. Beberapa diantaranya adalah:
	* Penelitian oleh Saif et al pada papernya yang berjudul *"Semantic sentiment analysis of twitter"* di International semantic web conference pada tahun 2012. Saif menggunakan algoritma Naive Bayes dalam penelitiannya untuk melakukan sentiment analysis.
	* Penelitian oleh Kouloumpis et al pada papernya yang berjudul *"Twitter sentiment analysis: The good the bad and the omg!,"* di ICWSM tahun 2011. Kouloumpis et al menggunakan AdaBoost Classifier dan dataset The hash-tagged and emoticon sebagai data training.
	* Dari banyaknya metode yang digunakan untuk sentiment analysis, SVM dan MNB memiliki nilai presisi yang paling baik, terlebih pada data dengan multiple features ([Alsaeedi et al, 2019](https://thesai.org/Publications/ViewPaper?Volume=10&Issue=2&Code=IJACSA&SerialNo=48)).
 
# Business Understanding
 
## Problem Statement
* Banyaknya tweet yang dikirim oleh pengguna twitter membuat isi konten twitter sulit dikendalikan.
* Konten yang dibuat oleh pengguna twitter memiliki konteks dan tujuan yang berbeda-beda.
 
## Goals
* Memahami sentiment pada tweet yang ditulis oleh pengguna twitter dengan melakukan analisis pada data tweet.
* Membuat model machine learning untuk mendeteksi sentiment pada tweet baru.
 
## Solution Statement
Salah satu teknik machine learning yang dapat digunakan adalah GRU, dimana GRU memiliki kemampuan yang sangat cocok untuk mengolah data teks karena mampu mengingat urutan pada suatu data.

# Data Understanding
Data yang digunakan untuk project ini adalah data dari [kaggle](https://www.kaggle.com/datasets/ilhamfp31/indonesian-abusive-and-hate-speech-twitter-text) yang berisi tweet bahasa Indonesia beserta labelny.
 
Dataset memiliki total 13169 tweet sebelum dilakukan data cleansing, dan menjadi 13023 tweet setelah data cleansing.
 
## Variabel-variabel pada dataset Tweet Entity Sentiment Analysis
Masing-masing data terdiri dari 4 kolom, yaitu:
1. Tweet : teks tweet Bahasa Indonesia
2. HS : label tweet termasuk hate speech (1) atau tidak (0)
3. Abusive : label tweet termasuk abusive (1) atau tidak (0)
4. HS_Individual : label tweet hate speech menyerang individu atau bukan
5. HS_Group : label tweet hate speech menyerang suatu grup/kelompok atau bukan
6. HS_Religion : label tweet hate speech menyerang suatu agama atau bukan
7. HS_Race : label tweet hate speech menyerang suatu ras atau bukan
8. HS_Physical : label tweet hate speech menyerang suatu fisik seseorang atau bukan
9. HS_Gender : label tweet hate speech menyerang suatu gender seseorang atau bukan
10. HS_Other : label tweet hate speech menyerang hal lainnya atau bukan
11. HS_Weak : label tweet hate speech bersifat implisit seperti menyindir
12. HS_Moderate : label tweet hate speech bersifat eksplisit tapi tidak kasar
13. HS_Strong : label tweet hate speech bersifat eksplisit dan kasar
 
## Kondisi Data
Dataset ini memiliki 146 data duplikat.
Baris yang memiliki data duplikat akan dihapus. Sehingga setelah dihapus, data training tersisa 13023 baris.
 
## Exploratory Data Analysis
![image](https://github.com/muhafidz-ahmad/sentiment-analysis-of-indonesian-tweet/assets/115754250/05b30fd7-f672-405d-9848-53c69dc6e180)

![image](https://github.com/muhafidz-ahmad/sentiment-analysis-of-indonesian-tweet/assets/115754250/5eeb6b81-f626-4ef0-a3c7-7c86e35dcdd1)

![image](https://github.com/muhafidz-ahmad/sentiment-analysis-of-indonesian-tweet/assets/115754250/45d6bedd-434a-4e42-8ca1-fafe376a5b2e)

![image](https://github.com/muhafidz-ahmad/sentiment-analysis-of-indonesian-tweet/assets/115754250/56a83c43-3f08-4203-8514-f25363c59ee7)

![image](https://github.com/muhafidz-ahmad/sentiment-analysis-of-indonesian-tweet/assets/115754250/f9472f99-22fb-4f27-a3f7-671172eb1506)

![image](https://github.com/muhafidz-ahmad/sentiment-analysis-of-indonesian-tweet/assets/115754250/3f2cbf7f-6be3-4810-9c95-96b13ecd9240)

![image](https://github.com/muhafidz-ahmad/sentiment-analysis-of-indonesian-tweet/assets/115754250/6903d46d-f747-46db-92c1-3a33d239efb2)

![image](https://github.com/muhafidz-ahmad/sentiment-analysis-of-indonesian-tweet/assets/115754250/03cdb899-d634-4dbc-9b4a-2f673b3611e7)
 
# Data Preparation
Pada bagian persiapan data, akan dilakukan Text Preprocessing dan Feature Engineering.
 
## Text Preprocessing
Text preprocessing dilakukan berdasarkan beberapa alasan. Pertama, untuk membuat data teks lebih mudah dipahami oleh model machine learning. Kedua, untuk efisiensi data ketika proses training. Karena pada tahap text preprocessing ini, akan memperbaiki dan mengurangi beberapa fitur yang tidak memiliki dampak yang besar pada makna teks.
 
Terdapat 3 teknik yang dilakukan pada text preprocessing di project ini, yaitu:
 
1. Case folding, membuat semua huruf menjadi huruf kecil. Case folding dilakukan untuk membuat kata-kata yang sama namun dengan dengan case yang berbeda menjadi satu makna yang sama, hal ini karena mesin memiliki sifat case sensitivity.
2. Remove unnecessary character, menghapus karakter atau kata-kata yang tidak penting. Seperti anotasi bawaan dari twitter.
3. Punctuation removal, menghapus semua tanda baca, seperti titik, koma, dan tanda baca lainnya. Puctuation removal dilakukan karena tanda baca bisa menjadi distraksi bagi mesin dalam memahami makna suatu kalimat.
4. Stopword removal, menghapus kata-kata yang termasuk stopword. Stopword removal juga dilakukan karena stopword tidak mempunyai makna, dan jika tidak dihapus hanya akan menanbah beban komputasi.
5. Mengganti kata-kata alay atau tidak baku menjadi baku, menggunakan dictionary yang telah disediakan di tautan dataset kaggle yang sama.
 
## Feature Engineering
Akan dibuat empat target atau label pada proyek ini, yaitu:
* Positive (Bukan Hate speech dan Tidak Abusive)
* Negative 1 (Hate speech)
* Negative 2 (Abusive)
* Very Negative (Hate speech dan Abusive)

![image](https://github.com/muhafidz-ahmad/sentiment-analysis-of-indonesian-tweet/assets/115754250/2372e85a-7869-488f-b5d3-76e101c2c020)
 
### Ubah label menjadi one hot encoding
Target akan diubah menjadi one-hot encoding agar lebih mudah dimasukan ke dalam model.
Untuk mengubah kolom sentiment menjadi one-hot encoding, digunakan fungsi *get_dummies()*.
 
Setelah penggunakan fungsi *get_dummies()* ini, kolom sentiment akan dipecah menjadi 4 kolom, karena kolom sentiment memiliki 4 nilai unik, yaitu Positive, Negative 1, Negative 2, dan Very Negative.
 
### Tokenizing
Tokenizing dilakukan untuk memecah kalimat tweet menjadi token-token atau kata-kata.
Pada model GRU, tokenizing dilakukan dengan menggunakan fungsi Tokenizer di tensorflow keras. Kemudian, tokenizer dilatih hanya pada data training.
 
# Modelling
GRU (Gated Recurrent Unit) adalah salah satu jenis arsitektur RNN (Recurrent Neural Network) yang dikembangkan oleh Cho et al. pada tahun 2014. GRU adalah pengembangan dari LSTM (Long Short-Term Memory) yang memiliki kemampuan untuk mengatasi masalah vanishing gradient pada LSTM. GRU memiliki dua gate yaitu reset gate dan update gate yang memungkinkan GRU untuk memilih informasi mana yang akan disimpan dan mana yang akan diabaikan. GRU juga memiliki kemampuan untuk mengatasi masalah overfitting pada data training.

![image](https://github.com/muhafidz-ahmad/sentiment-analysis-of-indonesian-tweet/assets/115754250/aa48093d-5650-4293-a24e-4ce9cce1ea10)
 
Kekurangan model ini bisa dibilang model yang sudah tertinggal, karena model paling canggih saat ini adalah sudah berbasis transformer, bukan berbasis RNN.
 
Pada project ini, akan digunakan arsitektur LSTM dengan tambahan Attention Mechanism.
Peningkatan performa dilakukan dengan hyperparameter tuning secara manual. Adapun hyperparameter yang memiliki nilai akurasi tertinggi project ini adalah:
* embedding_dim = 16
* dense_dim = 16
* dense_layers = 1
* dropout = 0.5
* lr = 0.001
* gru_dim = 32
* gru_layers = 1
* batch_size = 256
* epochs = 45
 
# Evaluation
Metrik evaluasi yang digunakan pada project ini adalah akurasi.
Akurasi merupakan metrik yang dapat mengukur kemampuan model klasifikasi dalam mengklasifikasikan dengan benar data pada keseluruhan dataset. Akurasi dapat dihitung dengan membagi jumlah prediksi yang benar dibagi dengan jumlah total data. 
 
$$akurasi = {TP + TN \\over TP + TN + FP + FN}.$$
 
Dimana:
* TP (True Positive), jumlah data positif yang diklasifikasikan dengan benar.
* TN (True Negative), jumlah data negatif yang diklasifikasikan dengan benar.
* FP (False Positive), jumlah data negatif yang salah diklasifikasikan sebagai positif (type 1 error).
* FN (False Negative), jumlah data positif yang salah diklasifikasikan sebagai negatif (type 2 error).
 
Model berhasil mendapatkan nilai evaluasi sebagai berikut:
* **Akurasi Training: 0.8374**
* **Akurasi Validasi: 0.7578**
* **Akurasi Testing: 0.7589**
 
Proses training berhenti pada epochs ke-18, karena dalam 5 epochs terakhir tidak ada penurunan loss validation yang besar.
 
# Conclusion
* Dari dataset yang digunakan, terdapat hingga 42,2% tweet yang berisi ujaran kebencian. Kata yang paling sering muncul dalam hate speech tersebut terdiri dari “Jokowi”, “cebong”, “komunis”, dan “ganti presiden”.
* 38,3% tweet yang berkonotasi kasar (abusive tweet) menampilkan beberapa kata kasar yang sering, seperti “g*blok”, “anj*ng”, dan “d*ngu”.
* Hate speech tweet dan abusive tweet paling tinggi berada pada kategori other dan individu, yang artinya banyak tweet yang menyerang personal orang lain.
* Hate speech individu memiliki korelasi yang kuat dengan hate speech weak (seperti menyindir). Sedangkan hate speech yang ditujukan pada suatu grup memiliki korelasi yang kuat dengan hate speech moderate (seperti mengancam, fitnah, provokasi).
* Model machine learning dapat mendeteksi 76% sentiment tweet dengan benar pada data test, yang sebetulnya masih perlu ditingkatkan lagi.

# Recommendation
* Memberikan edukasi yang lebih gencar kepada masyarakat mengenai dampak negatif dan ke-tidak-bermanfaatannya hate speech.
* Menerapkan model machine learning untuk menyaring hate speech tweet. Dengan model yang telah dibuat, setidaknya dapat menyaring 70% tweet yang ada.
* Terus meningkatkan performa model machine learning dengan hyperparameter tuning lebih lanjut dan memperkaya data hingga dapat diperoleh data yang seimbang.
* Memantau secara berkala model machine learning yang telah diterapkan, dan juga memantau tren dan perubahan dalam penggunaan bahasa dan perilaku netizen terkait hate speech.
