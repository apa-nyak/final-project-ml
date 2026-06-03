# Implementasi dan Analisis Komparatif MobileNetV3 dan DenseNet121 untuk Klasifikasi Penyakit Daun Padi 

Anggota Tim
Proyek ini dikerjakan oleh Kelompok 8 sebagai pemenuhan Ujian Akhir Semester Pembelajaran Mesin:

Maria Agatha Metri Tetrajati (24031554189)
Tara Tabriza Rachman (24031554107)
Ahmad Dhani Alfawwas (24031554096)

## Deskripsi Proyek
Proyek ini bertujuan untuk melakukan implementasi serta analisis komparatif beberapa arsitektur *deep learning*, yaitu **MobileNet (MobileNetV3)** dan kombinasi **CNN-MLP** dalam melakukan klasifikasi penyakit daun padi menggunakan citra digital.

Penelitian ini merupakan bentuk penerapan teknologi kecerdasan artifisial dalam mendukung transformasi digital sektor pangan, khususnya untuk meminimalisir risiko gagal panen melalui deteksi dini. Fokus utama penelitian adalah membandingkan performa model dari sisi akurasi, efisiensi komputasi, serta kemampuan generalisasi dalam mendeteksi berbagai jenis penyakit daun padi, guna menemukan arsitektur paling efisien yang berpotensi diadopsi pada perangkat dengan spesifikasi komputasi rendah (*low-resource devices*).

## Dataset
Dataset yang digunakan berasal dari repositori publik Kaggle:

**[RICE Leaf Diseases Dataset by Yusuf Murtaza](https://www.kaggle.com/datasets/yusufmurtaza01/rice-leaf-diseases)**

Dataset ini terdiri dari citra daun padi (`.jpg`) dan file anotasi *bounding box* dalam format YOLO (`.txt`). Terdapat **9 kelas kondisi tanaman** yang dievaluasi dalam proyek ini:

1. `Rice__BacterialLeafBlight`
2. `Rice__BrownSpot`
3. `Rice__Healthy`
4. `Rice__Hispa`
5. `Rice__LeafBlast`
6. `Rice__LeafScald`
7. `Rice__LeafSmut`
8. `Rice__NarrowBrownLeafSpot`
9. `Rice__NeckBlast`

## Metodologi
Proyek ini mengimplementasikan pendekatan **Transfer Learning** dan **Multi-Layer Perceptron (MLP)**:
1. **Ekstraksi Fitur:** Menggunakan *base model* Convolutional Neural Network (**MobileNetV3**) untuk mengekstrak peta fitur (*feature maps*) spasial dari citra masukan.
2. **Klasifikasi (CNN-MLP):** Mengganti *layer* klasifikasi bawaan model dengan arsitektur MLP (*Dense Layers*) kustom buatan sendiri untuk memetakan fitur yang diekstrak menjadi probabilitas 9 kelas penyakit.
3. **Evaluasi:** Mengukur performa komparatif menggunakan metrik *Accuracy*, *Precision*, *Recall*, *F1-Score*, dan menganalisis *Confusion Matrix*.

## Kendala
Dalam proses pra-pemrosesan data dan persiapan eksperimen awal, tim kami mengidentifikasi beberapa tantangan teknis utama beserta langkah mitigasi yang kami terapkan:

### 1. Inkompatibilitas Format Dataset (Object Detection ke Classification)
* **Kendala:** Dataset *RICE Leaf Diseases* yang tersedia di repositori publik dirancang untuk tugas *Object Detection* dengan anotasi target berbentuk koordinat *bounding box* YOLO (`.txt`). Sementara itu, arsitektur CNN-MLP yang kami bangun membutuhkan format dataset *ImageFolder* murni untuk tugas klasifikasi (*Image Classification*). Jika gambar utuh langsung digunakan, model berisiko mengalami bias akibat noise latar belakang (seperti tanah, air, atau area daun sehat).
* **Solusi:** Kami mengembangkan skrip Python kustom untuk melakukan ekstraksi. Skrip ini membaca berkas koordinat YOLO, mendenormalisasinya ke skala piksel asli, dan mengaplikasikan teknik **Adaptive Cropping** dengan perluasan bingkai (*expand ratio*) dinamis berdasarkan skala ukuran objek. Langkah ini berhasil mengisolasi titik-titik penyakit menjadi citra klasifikasi tunggal yang fokus pada karakteristik bercak daun.

### 2. Ketidakseimbangan Kelas yang Ekstrem (Extreme Class Imbalance)
* **Kendala:** Setelah proses pemotongan (*cropping*) selesai dan visualisasi distribusi label dijalankan, ditemukan ketimpangan jumlah sampel yang sangat masif antar kelas penyakit. Kelas `BrownSpot` sangat mendominasi dengan lebih dari 1.200 sampel citra, sedangkan kelas lain seperti `LeafSmut` dan `Hispa` memiliki jumlah sampel yang sangat minim. Kondisi ini berpotensi memicu *Accuracy Paradox*, di mana model cenderung bias memprediksi kelas mayoritas namun gagal mengenali kelas minoritas.
* **Solusi:** Untuk mencegah model mengalami *overfitting* pada kelas mayoritas, kami mengintegrasikan penanganan *imbalanced data* pada tahap pelatihan model. Strategi yang direncanakan meliputi penerapan **Class Weights** (pembobotan kelas) pada fungsi *loss* untuk memberikan penalti komputasi lebih besar saat model salah mengklasifikasikan kelas minoritas, serta opsi melakukan *Hybrid Resampling* berupa pembatasan acak (*undersampling*) pada kelas mayoritas dan ekspansi data via augmentasi kustom pada kelas minoritas.#

### 3. Data Validation yang terindikasi leakage atau hanya salinan
* **Kendala:** Setelah melakukan training dan dilakukan validation, ternyata hasil terlalu baik untuk percobaan pertama, seperti to good to be true hasilnya, padahal data validation yang kami pakai itu data validation yang ada di dataset kaggle, bukan hasil split manual, lalu juga pada pengerjaan sudah kami cek tidak ada leakage, baik dalam preprocessing maupun cropping karena kami melakukannya satu per satu
* **Solusi:** Kami mencoba menggunakan test eksternal untuk melihat apakah model memang sebaik itu atau tidak, nyatanya memang tidak sebaik itu, tapi test yang kami lakukan memang masih dalam sekala kecil dengan dataset hanya berjumlah 120, juga gambar yang berbeda karakteristiknya karna yang kami gunakan untuk tes adalah gambar dari lab bukan dari lapangan, maka dari itu untuk selanjutnya kami akan coba menggunakan tes set lain dengan karakteristik data yang mirip. Atau kami juga akan melakukan training model ulang menggunakan split manual dari data train saja dan tidak menggunakan data validation yang ada di dataset. Untuk hal ini kami sangat mengharapkan saran dari Bapak dosen, apakah hal yang kami lakukan sudah sesuai arahnya.
