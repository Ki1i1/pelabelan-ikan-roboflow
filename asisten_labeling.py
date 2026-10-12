"""Asisten labeling dengan model siap pakai Roboflow, tanpa training sendiri.

Mulai dengan: python asisten_labeling.py preview --image foto.jpg
API key dimasukkan melalui input tersembunyi di terminal.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import getpass
import json
import math
import os
from pathlib import Path
import sys

WORKSPACE = 'rifkirozak35-gmail-com'
PROJECT = 'sipandu-tambak'
STAGE = 'membaca input'


class UserInputError(ValueError):
    """Pesan validasi lokal yang aman untuk ditampilkan."""


def normalize_key(raw, source):
    key = raw.strip()
    if not key:
        raise UserInputError('API key belum diisi.')
    if any(character.isspace() for character in key):
        raise UserInputError('Input key mengandung spasi/baris baru. Salin hanya nilai Private API Key.')
    # SDK 1.7.0 menganggap input tanpa huruf kecil sebagai dummy/onboarding.
    if not any(character.islower() for character in key):
        raise UserInputError(
            f'Input dari {source} dianggap key contoh oleh SDK Roboflow. '
            'Salin nilai Private API Key yang sebenarnya, bukan tanda penyamaran atau nama key.')
    return key


def safe_failure(error):
    """Tampilkan tipe/status saja; pesan mentah SDK dapat memuat URL credential."""
    from requests.exceptions import ConnectionError, Timeout, SSLError
    status = getattr(error, 'status_code', None)
    response = getattr(error, 'response', None)
    if status is None and response is not None:
        status = response.status_code
    hint = 'Penyebab belum teridentifikasi; kirim baris diagnostik ini saja.'
    if status == 401:
        hint = 'Autentikasi ditolak. Periksa nilai Private API Key.'
    elif status == 403:
        hint = 'Akses ditolak. Periksa izin key untuk workspace/proyek ini.'
    elif status == 429:
        hint = 'Batas permintaan tercapai; coba lagi nanti.'
    elif isinstance(error, SSLError):
        hint = 'Koneksi HTTPS gagal saat verifikasi sertifikat.'
    elif isinstance(error, Timeout):
        hint = 'Koneksi melewati batas waktu.'
    elif isinstance(error, ConnectionError):
        hint = 'Koneksi jaringan ke Roboflow gagal.'
    return f'Gagal pada tahap {STAGE}; tipe={type(error).__name__}; HTTP={status or "tidak diketahui"}. {hint}'


def available_model(models, model_id):
    if model_id == 'auto':
        candidates = [m for m in models if m.get('available') is True and
                      (str(m.get('id', '')).endswith('-boxes') or
                       m.get('outputFormat') in ('bbox', 'boxes', 'bounding-box'))]
        if not candidates:
            raise UserInputError('Tidak ada model bounding box yang tersedia. Kirim isi katalog-model.json (bukan API key) untuk diperiksa.')
        # Gunakan default akun bila kompatibel; selebihnya pilih biaya terendah.
        def priority(m):
            try:
                cost = Decimal(str(m.get('creditsPerImage')))
                if not cost.is_finite() or cost < 0:
                    cost = Decimal('Infinity')
            except InvalidOperation:
                cost = Decimal('Infinity')
            return (m.get('isDefault') is not True, cost, str(m.get('id')))
        return sorted(candidates, key=priority)[0]
    model = next((m for m in models if m.get('id') == model_id), None)
    if model is None or model.get('available') is not True:
        raise UserInputError('Model yang diminta tidak tersedia. Cek katalog-model.json atau hilangkan --model agar dipilih otomatis.')
    return model


def checked_cost(model, number, limit):
    """Jangan menjalankan batch bila biaya tidak diketahui atau di atas batas."""
    try:
        rate = Decimal(str(model['creditsPerImage']))
        ceiling = Decimal(str(limit))
    except (KeyError, InvalidOperation):
        raise UserInputError('Biaya/batas kredit tidak diketahui; batch tidak dijalankan.') from None
    if number < 1 or not rate.is_finite() or not ceiling.is_finite() or rate < 0 or ceiling < 0:
        raise UserInputError('Jumlah gambar atau batas kredit tidak valid.')
    total = rate * number
    if total > ceiling:
        raise UserInputError('Estimasi kredit melebihi batas; batch tidak dijalankan.')
    return total


def save_json(path, data, key):
    # Jangan menyimpan credential jika kebetulan muncul pada respons kesalahan.
    text = json.dumps(data, indent=2, ensure_ascii=False).replace(key, '[REDACTED]')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def main():
    global STAGE
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    job_plan = sub.add_parser('job-plan', help='Baca daftar gambar assignment; tidak mengubah anotasi')
    job_plan.add_argument('--job-id', required=True)
    job_plan.add_argument('--output', type=Path, default=Path(__file__).with_name('hasil-asisten') / 'job-plan.json')
    job_label = sub.add_parser('job-label', help='Deteksi ikan dan kirim bounding box ke gambar kosong dalam assignment')
    job_label.add_argument('--job-id', required=True)
    job_label.add_argument('--image-id', help='Batasi ke satu gambar tertentu dalam assignment')
    job_label.add_argument('--limit', type=int, default=1, help='Jumlah gambar kosong yang diproses; default 1')
    job_label.add_argument('--prompt', default='small fish')
    job_label.add_argument('--confidence', type=float, default=0.4,
                           help='Default 0.4 berdasarkan satu foto koreksi; masih perlu review')
    job_label.add_argument('--tile-size', type=int, default=1024, help='Default 1024 untuk ikan kecil; 0 = seluruh foto')
    job_label.add_argument('--tile-overlap', type=float, default=0.15)
    job_label.add_argument('--request-delay', type=float, default=5,
                           help='Jeda antar preview API dalam detik; default 5')
    job_label.add_argument('--max-retries', type=int, default=2,
                           help='Retry 429 per potongan; default 2, total cooldown satu proses maksimal 180 detik')
    job_label.add_argument('--min-boxes', type=int, default=5,
                           help='Tahan kiriman jika kandidat lebih sedikit; default 5 untuk foto ikan padat, bukan ukuran akurasi')
    job_label.add_argument('--send', action='store_true', help='Simpan kandidat ke gambar yang sama di Roboflow')
    job_label.add_argument('--output', type=Path, default=Path(__file__).with_name('hasil-asisten') / 'assignment')
    review = sub.add_parser('review', help='Ubah preview polygon lokal menjadi gambar bounding box dan XML kandidat; tanpa API key')
    review.add_argument('--image', type=Path, required=True)
    review.add_argument('--input', type=Path, default=Path(__file__).with_name('hasil-asisten') / 'preview.json')
    review.add_argument('--output', type=Path, default=Path(__file__).with_name('hasil-asisten') / 'review')
    review.add_argument('--min-confidence', type=float, default=0.0)
    sub.add_parser('models', help='Lihat model foundation yang tersedia dan tarifnya')
    preview = sub.add_parser('preview', help='Uji satu gambar tanpa mengubah anotasi')
    preview.add_argument('--image', type=Path, required=True)
    preview.add_argument('--model', default='auto', help='ID model atau auto (mengikuti katalog akun)')
    preview.add_argument('--class-name', default='ikan')
    preview.add_argument('--prompt', help='Deskripsi objek untuk model promptMap, misalnya small fish')
    preview.add_argument('--confidence', type=float, help='Ambang SAM 3 dari 0 sampai 1')
    preview.add_argument('--output', type=Path, default=Path(__file__).with_name('hasil-asisten') / 'preview.json')
    batch = sub.add_parser('batch', help='Rencana/jalankan Auto Label native pada batch Roboflow')
    batch.add_argument('--batch-id', required=True, help='ID batch upload, bukan ID annotation job')
    batch.add_argument('--model', default='auto')
    batch.add_argument('--class-name', default='ikan')
    batch.add_argument('--count', type=int, required=True)
    batch.add_argument('--max-credits', required=True, help='Batas kredit yang Anda izinkan')
    batch.add_argument('--run', action='store_true', help='Mulai job berbayar setelah rencana diperiksa')
    batch.add_argument('--output', type=Path, default=Path(__file__).with_name('hasil-asisten') / 'job.json')
    status = sub.add_parser('status', help='Cek job Auto Label yang sudah dimulai')
    status.add_argument('--job-id', required=True)
    args = parser.parse_args()
    if args.command == 'review':
        from review_preview import write_review
        if not args.image.is_file() or not args.input.is_file():
            raise UserInputError('Foto atau hasil preview lokal tidak ditemukan.')
        try:
            write_review(args.image, args.input, args.output, args.min_confidence)
        except (ValueError, KeyError, TypeError) as error:
            raise UserInputError('Format/koordinat preview tidak valid untuk review lokal.') from error
        return
    if args.command == 'preview' and not args.image.is_file():
        raise UserInputError('File gambar tidak ditemukan.')
    if args.command == 'preview' and args.confidence is not None and not 0 <= args.confidence <= 1:
        raise UserInputError('--confidence harus antara 0 dan 1.')
    if args.command == 'job-label' and (args.limit < 1 or not 0 <= args.confidence <= 1):
        raise UserInputError('--limit minimal 1 dan --confidence harus 0 sampai 1.')
    if args.command == 'job-label' and (args.tile_size != 0 and not 256 <= args.tile_size <= 2048 or not 0 <= args.tile_overlap < 0.8):
        raise UserInputError('--tile-size harus 0 atau 256–2048; --tile-overlap harus 0 sampai kurang dari 0.8.')
    if args.command == 'job-label' and (not math.isfinite(args.request_delay) or args.request_delay < 0):
        raise UserInputError('--request-delay harus angka hingga, minimal 0.')
    if args.command == 'job-label' and (not 0 <= args.max_retries <= 4 or args.min_boxes < 1):
        raise UserInputError('--max-retries harus 0–4; --min-boxes minimal 1.')

    environment_key = os.environ.get('ROBOFLOW_API_KEY')
    if environment_key:
        print('Menggunakan key dari environment ROBOFLOW_API_KEY.')
    key = normalize_key(environment_key or getpass.getpass('PRIVATE API key Roboflow (tersembunyi): '),
                        'ROBOFLOW_API_KEY' if environment_key else 'input terminal')
    from roboflow import Roboflow
    STAGE = 'validasi API key'
    client = Roboflow(api_key=key)
    STAGE = 'akses workspace'
    workspace = client.workspace(WORKSPACE)
    if args.command == 'job-plan':
        from job_assignment import scan_job
        STAGE = 'akses proyek'
        project = workspace.project(PROJECT)
        STAGE = 'membaca gambar assignment'
        result = scan_job(project, args.job_id)
        save_json(args.output, result, key)
        print('Daftar assignment:', result.get('image_count', 'format API perlu diperiksa'))
        print('Pemeriksaan lengkap:', result['complete'])
        print('Hasil pemeriksaan:', args.output.resolve())
        print('Belum ada deteksi atau penulisan label. Daftar ini dipakai untuk menghubungkan kotak ke gambar dalam job.')
        return
    if args.command == 'status':
        STAGE = 'status job'
        result = workspace.autolabel_job(args.job_id)
        print(json.dumps(result, ensure_ascii=False).replace(key, '[REDACTED]'))
        return

    if args.command == 'job-label':
        from label_assignment import label_job
        STAGE = 'katalog model SAM 3'
        available_model(workspace.autolabel_models().get('models', []), 'sam3-polygon')
        STAGE = 'akses proyek'
        project = workspace.project(PROJECT)
        def set_stage(value):
            global STAGE
            STAGE = value
        try:
            label_job(project, args, key, save_json, set_stage)
        except ValueError as error:
            # Jangan menampilkan pesan arbitrary SDK atau URL download.
            raise UserInputError(f'Proses assignment berhenti pada tahap {STAGE}. Periksa laporan lokal; label lama tidak ditimpa.') from error
        return

    STAGE = 'katalog model Auto Label'
    catalog = workspace.autolabel_models()
    models = catalog.get('models')
    if not isinstance(models, list):
        raise UserInputError('Katalog model tidak bisa dibaca.')
    # Simpan hanya metadata model yang aman untuk diagnosis, bukan respons mentah.
    fields = ('id', 'name', 'available', 'unavailableReason', 'creditsPerImage',
              'ontologyFormat', 'projectTypes', 'outputFormat', 'isDefault')
    safe_catalog = [{field: model[field] for field in fields if field in model}
                    for model in models]
    catalog_path = Path(__file__).with_name('katalog-model.json')
    save_json(catalog_path, {'models': safe_catalog}, key)
    print('Katalog model tersimpan:', catalog_path.resolve())
    if args.command == 'models':
        for model in models:
            print(f"{model.get('id')}: tersedia={model.get('available')}; kredit/gambar={model.get('creditsPerImage')}; format={model.get('ontologyFormat')}")
            if model.get('unavailableReason'):
                print('  Alasan:', str(model['unavailableReason']).replace(key, '[REDACTED]'))
        return
    model = available_model(models, args.model)
    args.model = model['id']
    print('Model dipilih:', args.model)
    is_box_model = (str(args.model).endswith('-boxes') or
                    model.get('outputFormat') in ('bbox', 'boxes', 'bounding-box'))
    if args.command == 'batch' and not is_box_model:
        raise UserInputError('Batch proyek ini memerlukan bounding box. Model segmentasi hanya diuji melalui preview; konversi kotak belum diverifikasi.')
    STAGE = 'akses proyek'
    project = workspace.project(PROJECT)
    # Gemini membaca nama kelas; kelas 'ikan' juga tetap cocok dengan proyek.
    # Model lain dapat memakai ontology prompt -> nama kelas melalui SDK.
    ontology = [args.class_name]
    if args.command == 'preview':
        options = {}
        if args.prompt:
            if model.get('ontologyFormat') != 'promptMap':
                raise UserInputError('Model ini hanya membaca nama kelas; --prompt tidak didukung.')
            ontology = {args.prompt: args.class_name}
        if args.confidence is not None:
            if not str(args.model).startswith('sam3-'):
                raise UserInputError('--confidence pada preview hanya didukung SAM 3.')
            options['confidence_threshold'] = args.confidence
        if not is_box_model:
            print('Model menghasilkan segmentasi. Preview mentah disimpan untuk memeriksa konversi bounding box.')
        print('Mengirim satu gambar ke Roboflow Auto Label Preview. Anotasi lama tetap tersimpan.')
        STAGE = 'preview Auto Label'
        result = project.autolabel_preview(args.model, str(args.image.resolve()), ontology=ontology, **options)
        save_json(args.output, result, key)
        if result.get('blockErrors'):
            raise UserInputError('Model mengembalikan blockErrors. Preview belum berhasil; periksa hasil lokal.')
        print('Ringkasan:', json.dumps(result.get('summary', {}), ensure_ascii=False).replace(key, '[REDACTED]'))
        print('Hasil preview:', args.output.resolve())
        print('Belum ada label yang dikirim ke editor. Periksa deteksi sebelum menjalankan batch.')
    elif args.command == 'batch':
        cost = checked_cost(model, args.count, args.max_credits)
        plan = {'batch_id': args.batch_id, 'model': args.model, 'num_images': args.count,
                'estimated_credits': str(cost), 'preserve_existing_annotations': True}
        print(json.dumps(plan, indent=2))
        if not args.run:
            print('Rencana saja. Setelah diperiksa, tambahkan --run untuk menjalankan job.')
            return
        STAGE = 'membuat job Auto Label'
        result = project.autolabel(args.batch_id, model=args.model, ontology=ontology,
                                   num_images=args.count, preserve_existing_annotations=True,
                                   run_nms=True)
        save_json(args.output, result, key)
        print('Job dibuat. Periksa kandidat kotak di Roboflow Annotate.')
        print('Hasil:', args.output.resolve())


if __name__ == '__main__':
    try:
        main()
    except UserInputError as error:
        # Pesan-pesan ini dibuat lokal, tidak mencetak URL credential dari SDK.
        print('Gagal:', str(error), file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        print(safe_failure(error), file=sys.stderr)
        sys.exit(1)
