"""SAM 3 -> bounding box -> gambar yang sudah ada dalam assignment Roboflow."""
from __future__ import annotations

import json
from pathlib import Path
import re
import time
from urllib.parse import urlparse

from PIL import Image
import requests

from autolabel_ikan import deduplicate, voc_xml
from job_assignment import scan_job, empty_annotation
from review_preview import polygon_boxes, write_review
from preview_requests import PreviewRequester, http_status


def download_original(details, target):
    url = details.get('urls', {}).get('original')
    if not isinstance(url, str) or urlparse(url).scheme != 'https':
        raise ValueError('URL HTTPS foto asli tidak tersedia dari API.')
    # URL asli hanya dipakai dalam memori, tidak disimpan dalam laporan/log.
    with requests.get(url, timeout=(30, 120), stream=True) as response:
        response.raise_for_status()
        with target.open('wb') as stream:
            for block in response.iter_content(1024 * 1024):
                stream.write(block)
    with Image.open(target) as image:
        if image.format not in ('JPEG', 'PNG', 'WEBP'):
            raise ValueError('Format foto asli belum didukung.')
        if image.getexif().get(274, 1) != 1:
            raise ValueError('Foto memiliki orientasi EXIF; koordinat perlu diverifikasi dahulu.')
        return image.size


def member_ids(project, job_id):
    listing = scan_job(project, job_id)
    if not listing.get('complete'):
        raise ValueError('Daftar gambar assignment belum lengkap.')
    return [image['image_id'] for image in listing['images']]


def send_boxes(project, job_id, image_id, xml_path):
    # Baca ulang setelah inference, supaya edit pengguna selama proses terjaga.
    if image_id not in member_ids(project, job_id):
        return {'status': 'skipped', 'reason': 'Gambar tidak lagi berada dalam assignment.'}
    current = project.image(image_id)
    if current.get('id') != image_id or not empty_annotation(current):
        return {'status': 'skipped', 'reason': 'Label sudah ada atau status kosong tidak dapat dipastikan.'}
    response = project.save_annotation(
        annotation_path=str(xml_path), image_id=image_id,
        is_prediction=True, annotation_overwrite=False, add_to_dataset=False,
    )
    saved = response[0] if isinstance(response, tuple) else response
    if not isinstance(saved, dict):
        raise ValueError('Respons penyimpanan anotasi belum dikenali.')
    if saved.get('warn'):
        return {'status': 'skipped', 'reason': 'Server tidak menulis; gambar mungkin sudah berlabel.'}
    if saved.get('success') is not True:
        raise ValueError('Server belum mengonfirmasi penyimpanan anotasi.')
    verified = project.image(image_id)
    if image_id not in member_ids(project, job_id):
        return {'status': 'sent_assignment_changed', 'server_confirmed': True,
                'note': 'Server menyimpan kandidat, tetapi gambar tidak ditemukan lagi di job asal. Proses dihentikan.'}
    visible = verified.get('annotation') is not None or bool(verified.get('labels'))
    return {'status': 'sent' if visible else 'sent_verification_pending', 'server_confirmed': True,
            'same_assignment': True, 'annotation_present': visible}


def label_job(project, args, key, save, stage):
    requester = PreviewRequester(getattr(args, 'request_delay', 5),
                                 max_retries=getattr(args,'max_retries',2))
    minimum_boxes = getattr(args,'min_boxes',5)
    stage('membaca assignment')
    image_ids = member_ids(project, args.job_id)
    if args.image_id:
        if args.image_id not in image_ids:
            raise ValueError('--image-id tidak berada dalam assignment ini.')
        image_ids = [args.image_id]
    report = {'job_id': args.job_id, 'model': 'sam3-polygon', 'prompt': args.prompt,
              'send': args.send, 'image_limit': args.limit, 'min_confidence': args.confidence,
              'tile_size': args.tile_size, 'tile_overlap': args.tile_overlap,
              'request_delay_seconds': requester.request_delay,
              'max_retries':requester.max_retries, 'max_retry_wait_seconds':requester.max_retry_wait,
              'min_boxes_before_send':minimum_boxes,
              'assignment_count': len(image_ids), 'images': []}
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / 'laporan.json'
    save(report_path, report, key)
    processed = 0
    for image_id in image_ids:
        if processed >= args.limit:
            break
        if not re.fullmatch(r'[A-Za-z0-9_-]+', image_id):
            raise ValueError('ID gambar tidak valid untuk folder lokal.')
        stage('memeriksa label lama')
        details = project.image(image_id)
        if details.get('id') != image_id:
            raise ValueError('Respons ID gambar tidak cocok.')
        if not empty_annotation(details):
            print('Lewati gambar berlabel/status tidak pasti:', image_id)
            report['images'].append({'image_id': image_id, 'status': 'skipped_existing_or_unknown'})
            save(report_path, report, key)
            continue
        record = {'image_id': image_id, 'name': details.get('name'), 'status': 'preparing'}
        report['images'].append(record)
        save(report_path, report, key)
        directory = args.output / image_id
        directory.mkdir(parents=True, exist_ok=True)
        image_path = directory / 'original.jpg'
        stage('mengambil foto asli assignment')
        size = download_original(details, image_path)
        print('Deteksi gambar kosong:', details.get('name', image_id), flush=True)
        stage('deteksi SAM 3')
        inference_started = time.monotonic()
        prior_stats = dict(requester.stats)
        try:
            if args.tile_size:
                from sam3_tiles import infer_tiles
                preview = infer_tiles(project, image_path, directory, args.prompt, args.confidence,
                                      args.tile_size, args.tile_overlap, save, key, stage,
                                      requester=requester)
            else:
                preview = requester.preview(project, image_path.resolve(), args.prompt, args.confidence)
        except Exception as error:
            record.update({'status': 'failed_inference', 'http_status': http_status(error),
                           'request_stats':{k:round(v-prior_stats[k],2) for k,v in requester.stats.items()},
                           'local_folder': str(directory.resolve()),
                           'resume_note': 'Ulangi image ID dengan parameter sama; cache potongan berhasil dipertahankan.'})
            save(report_path, report, key)
            raise
        preview_path = directory / 'preview.json'
        save(preview_path, preview, key)
        stage('konversi bounding box')
        boxes = polygon_boxes(preview, size, args.confidence)
        raw_count = len(preview['predictions'])
        confidence_count = len(boxes)
        # Selain confidence, hindari kotak yang menggabungkan kontur berjauhan.
        boxes = [box for box in boxes
                 if 'kontur_berjauhan_kotak_terlalu_luas' not in box['review_reasons']]
        contour_count = len(boxes)
        boxes = deduplicate(boxes, 0.5)
        record.update({'filter_counts':{'mapped_from_model':raw_count,
                                         'confidence_rejected':raw_count-confidence_count,
                                         'disjoint_rejected':confidence_count-contour_count,
                                         'duplicate_rejected':contour_count-len(boxes),
                                         'final_boxes':len(boxes)},
                       'inference_seconds':round(time.monotonic()-inference_started,2),
                       'request_stats':{k:round(v-prior_stats[k],2) for k,v in requester.stats.items()}})
        print(f'Ringkasan: {raw_count} kandidat gabungan; '
              f'{raw_count-confidence_count} confidence rendah, '
              f'{confidence_count-contour_count} kontur berjauhan, '
              f'{contour_count-len(boxes)} duplikat; {len(boxes)} kotak akhir.',flush=True)
        xml_path = directory / 'ikan.candidates.xml'
        xml_path.write_text(voc_xml(details.get('name') or 'original.jpg', size, boxes,
                                    one_based=True), encoding='utf-8')
        save(directory / 'boxes.json', {'image_id': image_id, 'width': size[0], 'height': size[1],
                                       'reviewed': False, 'detections': boxes}, key)
        write_review(image_path, preview_path, directory / 'review', args.confidence,
                     detections_override=boxes, voc_one_based=True)
        record.update({'status': 'prepared', 'boxes': len(boxes), 'local_folder': str(directory.resolve()),
                       'editor_url': f'https://app.roboflow.com/rifkirozak35-gmail-com/sipandu-tambak/images/{image_id}?jobStatus=assigned&annotationJob={args.job_id}'})
        processed += 1
        save(report_path, report, key)
        if not boxes:
            record['status'] = 'no_boxes'
            print('Tidak ada kotak yang lolos; tidak ada label dikirim ke Roboflow.')
            if args.send:
                save(report_path, report, key)
                print('Batch dihentikan agar hasil kosong diperiksa dahulu.')
                break
        elif args.send and len(boxes) < minimum_boxes:
            record['status'] = 'sparse_review_required'
            save(report_path, report, key)
            print(f'Hasil hanya {len(boxes)} kotak, di bawah ambang kiriman {minimum_boxes}. '
                  'Tidak dikirim. Periksa preview lokal; batch dihentikan.',flush=True)
            break
        elif args.send:
            stage('mengirim kandidat bounding box')
            # Simpan niat pengiriman sebelum request; laporan membantu diagnosis jika jaringan putus.
            record['status'] = 'sending'
            save(report_path, report, key)
            outcome = send_boxes(project, args.job_id, image_id, xml_path)
            record.update(outcome)
            save(report_path, report, key)
            print('Status pengiriman:', record['status'], '; kotak:', len(boxes))
            print('Buka editor:', record['editor_url'])
            if record['status'] in ('sent_assignment_changed', 'sent_verification_pending'):
                print('Pengiriman selanjutnya dihentikan sampai tampilan editor diperiksa.')
                break
        else:
            print('Kandidat siap; tanpa --send tidak ada penulisan label.')
            print('Buka gambar dalam editor:', record['editor_url'])
        save(report_path, report, key)
    print('Gambar kosong diproses:', processed)
    print('Laporan:', report_path.resolve())
    print('Kandidat tetap perlu dikoreksi. Tidak ada approval, Add to Dataset, atau Submit Review otomatis.')
    return report
