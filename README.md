# Implementasi dan Analisis Komparatif MobileNetV3 dan DenseNet121 untuk Klasifikasi Penyakit Daun Padi 

Anggota Tim
Proyek ini dikerjakan oleh Kelompok 8 sebagai pemenuhan Ujian Akhir Semester Pembelajaran Mesin:

Maria Agatha Metri Tetrajati (24031554189)
Tara Tabriza Rachman (24031554107)
Ahmad Dhani Alfawwas (24031554096)

## Deskripsi Proyek
Proyek ini bertujuan untuk melakukan implementasi serta analisis komparatif beberapa arsitektur *deep learning*, yaitu **MobileNetV3** dan **DenseNet121** dalam melakukan klasifikasi penyakit daun padi menggunakan citra digital.

Penelitian ini merupakan bentuk penerapan teknologi kecerdasan artifisial dalam mendukung transformasi digital sektor pangan, khususnya untuk meminimalisir risiko gagal panen melalui deteksi dini penyakit tanaman. Fokus utama penelitian adalah membandingkan performa model dari sisi akurasi, efisiensi komputasi, serta kemampuan generalisasi dalam mendeteksi berbagai jenis penyakit daun padi guna menemukan arsitektur paling efektif dan efisien untuk implementasi pada perangkat dengan sumber daya komputasi terbatas (*low-resource devices*).

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

Namun, karena distribusi data yang sangat tidak seimbang, beberapa kelas minoritas kemudian digabungkan menjadi kategori OtherDisease untuk membantu stabilitas proses pelatihan model.

## Metodologi
Penelitian ini mengimplementasikan pendekatan *Transfer Learning* menggunakan dua arsitektur CNN modern yaitu MobileNetV3 dan DenseNet121.
1. **Adaptive Cropping berbasis YOLO**
Melakukan ekstraksi area penyakit berdasarkan koordinat bounding box YOLO sehingga setiap citra hanya merepresentasikan satu jenis penyakit.
2. **Image Enhancement**
Menggunakan metode CLAHE (Contrast Limited Adaptive Histogram Equalization) dan sharpening untuk meningkatkan kualitas visual citra penyakit daun padi.
3. **Data Augmentation dan Normalisasi**
Menerapkan horizontal flip, random rotation, color jitter, serta normalisasi ImageNet untuk meningkatkan kemampuan generalisasi model.
4. **Penanganan Imbalanced Dataset**
Menggunakan class weights pada fungsi loss untuk membantu model mempelajari kelas minoritas secara lebih seimbang.
5. **Pelatihan Model**
Mengimplementasikan MobileNetV3 dan DenseNet121 berbasis pretrained ImageNet menggunakan optimizer Adam dan CrossEntropyLoss.
6. **Evaluasi Model**
Mengukur performa model menggunakan Accuracy, Precision, Recall, F1-Score, serta Confusion Matrix pada pengujian internal dan eksternal.

## Hasil Penelitian
Hasil evaluasi internal menunjukkan bahwa kedua model memperoleh performa yang sangat tinggi dengan akurasi mendekati 99%. Hal ini menunjukkan bahwa proses preprocessing dan rekayasa fitur yang diterapkan mampu membantu model mempelajari karakteristik visual penyakit daun padi dengan sangat baik.

Namun, ketika dilakukan pengujian menggunakan dataset eksternal yang belum pernah digunakan sebelumnya, performa model mengalami penurunan yang cukup signifikan. Pada pengujian eksternal, MobileNetV3 memperoleh akurasi sekitar 31%, sedangkan DenseNet121 memperoleh akurasi sekitar 40%.

Hasil tersebut menunjukkan adanya perbedaan distribusi data (domain shift) antara dataset pelatihan dan dataset eksternal, sehingga kemampuan generalisasi model masih perlu ditingkatkan lebih lanjut. Meskipun demikian, DenseNet121 menunjukkan performa klasifikasi yang lebih baik pada data eksternal, sedangkan MobileNetV3 tetap unggul dari sisi efisiensi komputasi dan kecepatan inferensi sehingga lebih cocok diterapkan pada perangkat dengan sumber daya terbatas.

Penelitian ini juga menunjukkan bahwa akurasi tinggi pada validasi internal belum tentu mencerminkan performa yang sama pada data dunia nyata. Oleh karena itu, pengujian eksternal menjadi tahap penting dalam memastikan kemampuan generalisasi model deep learning.

## Kendala
Dalam proses pra-pemrosesan data dan persiapan eksperimen awal, tim kami mengidentifikasi beberapa tantangan teknis utama beserta langkah mitigasi yang kami terapkan:

### 1. Inkompatibilitas Format Dataset (Object Detection ke Classification)
* **Kendala:** Dataset awal menggunakan format *object detection* dengan anotasi YOLO, sedangkan model klasifikasi membutuhkan dataset berbentuk *ImageFolder*. Selain itu, satu gambar dapat mengandung beberapa objek penyakit sekaligus sehingga berpotensi menimbulkan noise latar belakang.
* **Solusi:** Dilakukan proses Adaptive Cropping berbasis anotasi YOLO untuk mengekstraksi area penyakit secara otomatis dan mengubah dataset menjadi dataset klasifikasi tunggal.

### 2. Ketidakseimbangan Kelas yang Ekstrem (Extreme Class Imbalance)
* **Kendala:** Setelah proses pemotongan (*cropping*) selesai dan visualisasi distribusi label dijalankan, ditemukan ketimpangan jumlah sampel yang sangat masif antar kelas penyakit. Kelas `BrownSpot` sangat mendominasi dengan lebih dari 1.200 sampel citra, sedangkan kelas lain seperti `LeafSmut` dan `Hispa` memiliki jumlah sampel yang sangat minim. Kondisi ini berpotensi memicu *Accuracy Paradox*, di mana model cenderung bias memprediksi kelas mayoritas namun gagal mengenali kelas minoritas.
* **Solusi:** Untuk mencegah model mengalami *overfitting* pada kelas mayoritas, kami mengintegrasikan penanganan *imbalanced data* pada tahap pelatihan model. Strategi yang direncanakan meliputi penerapan **Class Weights** (pembobotan kelas) pada fungsi *loss* sehingga model tetap memperhatikan kelas minoritas selama proses pelatihan berlangsung.

### 3. Data Validation yang terindikasi leakage atau hanya salinan
* **Kendala:** Pada pengujian internal, kedua model memperoleh performa yang sangat tinggi sejak tahap awal pelatihan sehingga menimbulkan dugaan adanya overfitting atau data leakage, terutama karena dataset dibentuk melalui proses cropping dari gambar sumber yang sama.
* **Solusi:** Dilakukan pengujian eksternal menggunakan dataset baru yang belum pernah digunakan sebelumnya untuk mengevaluasi kemampuan generalisasi model secara lebih realistis. Hasil pengujian eksternal menunjukkan bahwa performa model tidak setinggi hasil validasi internal sehingga memperkuat pentingnya evaluasi menggunakan data eksternal pada penelitian deep learning berbasis citra.
