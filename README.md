# Asisten Labeling Ikan

Script Python untuk membantu memberi kandidat bounding box pada gambar ikan
yang belum dianotasi di assignment Roboflow. Deteksi memakai SAM 3 melalui
Roboflow, kemudian polygon dikonversi menjadi bounding box.

Hasil perlu diperiksa dan dikoreksi oleh labeler. Script melewati gambar
yang sudah berlabel dan tidak melakukan approval, Submit Review, atau
Add to Dataset otomatis.

## Persiapan

Gunakan Python 3.12, koneksi internet, dan credential API yang diizinkan
untuk workspace/project serta assignment yang dikerjakan.
API key dimasukkan melalui prompt tersembunyi; jangan ditulis di kode.

Download repository dengan **Code → Download ZIP**, ekstrak, lalu buka
terminal di folder yang berisi `asisten_labeling.py`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Tujuan default adalah workspace `rifkirozak35-gmail-com`, project
`sipandu-tambak`, dengan kelas hasil `ikan`. Untuk workspace/project lain,
ubah `WORKSPACE` dan `PROJECT` di bagian atas `asisten_labeling.py`.
Mengganti Job ID saja tidak mengganti workspace/project.

## Jalankan

Ganti `JOB_ID` dengan ID pada URL assignment milik pengguna,
yaitu bagian setelah `/annotate/job/`.

Periksa daftar assignment terlebih dahulu:

```powershell
.\.venv\Scripts\python.exe asisten_labeling.py job-plan --job-id JOB_ID
```

Uji satu gambar tanpa mengirim label:

```powershell
.\.venv\Scripts\python.exe asisten_labeling.py job-label --job-id JOB_ID --limit 1 --tile-size 1024 --confidence 0.4 --request-delay 10
```

Buka gambar review yang lokasinya dicetak. Untuk mengirim kandidat pada
gambar yang sama, ganti `IMAGE_ID` dengan ID dari hasil uji:

```powershell
.\.venv\Scripts\python.exe asisten_labeling.py job-label --job-id JOB_ID--image-id IMAGE_ID --limit 1 --tile-size 1024 --confidence 0.4 --request-delay 10 --send
```

Setelah uji satu gambar dinilai memadai, proses maksimal lima gambar kosong:

```powershell
.\.venv\Scripts\python.exe asisten_labeling.py job-label --job-id JOB_ID --limit 5 --tile-size 1024 --confidence 0.4 --request-delay 10 --send
```

`--limit` menghitung gambar, bukan jumlah bounding box. Dengan `--send`,
hasil kurang dari lima kotak ditahan untuk pemeriksaan dan batch berhenti.
Ambang jumlah tersebut bukan ukuran akurasi.

## Cache dan error

Foto dan hasil review dibuat di folder `hasil-asisten` pada komputer
pengguna. Potongan deteksi yang berhasil disimpan sebagai cache. Jika
deteksi berhenti, ulangi satu `IMAGE_ID` dengan parameter yang sama untuk
memakai cache tersebut.

- HTTP 429: pembatasan permintaan, dengan retry terbatas pada `job-label`.
- HTTP 502: Bad Gateway; versi ini berhenti dan belum mencoba ulang otomatis.
- HTTP 401/403: periksa credential dan izin akses dengan pengelola workspace.

Foto blur atau pantulan kuat bisa menghasilkan deteksi sedikit atau keliru.
SAM 3 tidak otomatis belajar dari koreksi anotasi pengguna.

Panduan lengkap ada dalam [BACA-DULU.txt](BACA-DULU.txt).
Paket share yang sudah disiapkan tersedia sebagai
[asisten-labeling-ikan-untuk-teman.zip](asisten-labeling-ikan-untuk-teman.zip).

Kode dan import modul diperiksa secara lokal menggunakan Python 3.12.
Instalasi dan akses API pada akun pengguna lain belum diuji.

Dokumentasi layanan yang digunakan:
[Roboflow Python SDK](https://roboflow.github.io/roboflow-python/core/project/)
dan [Authentication](https://docs.roboflow.com/reference/authentication/authentication.md).
