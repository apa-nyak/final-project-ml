# Implekmentasi dan Analisis Komparatif MobileNet, DenseNet, dan kombinasi CNN-MLP untuk Klasifikasi Penyakit Daun Padi

Anggota Tim
Proyek ini dikerjakan oleh Kelompok Kelompok 8 sebagai pemenuhan Ujian Akhir Semester Pembelajaran Mesin:

Maria Agatha Metri Tetrajati (24031554189)
Tara Tabriza Rachman (24031554107)
Ahmad Dhani Alfawwas (24031554096)

## Deskripsi Proyek
Proyek ini bertujuan untuk melakukan implementasi serta analisis komparatif beberapa arsitektur deep learning, yaitu **MobileNet**, **DenseNet**, dan kombinasi **CNN-MLP** dalam melakukan klasifikasi penyakit daun padi menggunakan citra digital.
Penelitian ini merupakan bentuk penerapan teknologi kecerdasan artifisial dalam mendukung transformasi digital sektor pangan, khususnya untuk meminimalisir risiko gagal panen melalui deteksi dini. Fokus utama penelitian adalah membandingkan performa model dari sisi akurasi, efisiensi komputasi, serta kemampuan generalisasi dalam mendeteksi berbagai jenis penyakit daun padi, guna menemukan arsitektur paling efisien yang berpotensi diadopsi pada perangkat dengan komputasi rendah.

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
1. **Ekstraksi Fitur:** Menggunakan *base model* Convolutional Neural Network (MobileNet dan DenseNet) untuk mengekstrak peta fitur (*feature maps*) spasial dari citra masukan.
2. **Klasifikasi (CNN-MLP):** Mengganti *layer* klasifikasi bawaan model dengan arsitektur MLP (*Dense Layers*) buatan sendiri untuk memetakan fitur yang diekstrak menjadi probabilitas 9 kelas penyakit.
3. **Evaluasi:** Mengukur performa komparatif menggunakan metrik *Accuracy*, *Precision*, *Recall*, *F1-Score*, dan menganalisis *Confusion Matrix*.
