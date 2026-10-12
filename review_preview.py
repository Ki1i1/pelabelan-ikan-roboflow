"""Review lokal polygon Auto Label menjadi kandidat bounding box, tanpa API."""
from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from autolabel_ikan import voc_xml


def polygon_boxes(result, size, minimum=0.0):
    if result.get('blockErrors'):
        raise ValueError('Preview berisi blockErrors; tidak diekspor.')
    predictions = result.get('predictions')
    if not isinstance(predictions, list):
        raise ValueError('Format predictions belum didukung; perlu diperiksa.')
    width, height = size
    detections = []
    for index, prediction in enumerate(predictions, 1):
        if prediction.get('type') != 'polygon':
            raise ValueError('Review ini hanya mendukung respons polygon yang sudah diverifikasi.')
        confidence = float(prediction['confidence'])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Confidence tidak valid.')
        points = []
        component_areas = []
        # Satu prediksi dapat berisi beberapa kontur: tetap satu instance.
        for ring in prediction['vertices']:
            ring_points = []
            for point in ring:
                x, y = float(point['x']), float(point['y'])
                if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= width and 0 <= y <= height):
                    raise ValueError('Koordinat polygon tidak cocok dengan ukuran foto; tidak diskalakan secara otomatis.')
                points.append((x, y))
                ring_points.append((x, y))
            if ring_points:
                rx1, ry1 = map(min, zip(*ring_points))
                rx2, ry2 = map(max, zip(*ring_points))
                component_areas.append((rx2-rx1) * (ry2-ry1))
        if len(points) < 3:
            raise ValueError('Polygon tidak cukup titik.')
        x1, y1 = map(min, zip(*points))
        x2, y2 = map(max, zip(*points))
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Polygon menghasilkan kotak kosong.')
        if confidence >= minimum:
            reasons = []
            if confidence < 0.5:
                reasons.append('confidence_di_bawah_0.5')
            if (len(component_areas) > 1 and max(component_areas) > 0 and
                    (x2-x1) * (y2-y1) > 2 * max(component_areas)):
                reasons.append('kontur_berjauhan_kotak_terlalu_luas')
            detections.append({'candidate_id': index, 'class': prediction['class'],
                               'confidence': confidence, 'box': [x1, y1, x2, y2],
                               'review_reasons': reasons})
    return detections


def write_review(image_path, input_path, output, minimum=0.0, detections_override=None,
                 voc_one_based=False):
    if not 0 <= minimum <= 1:
        raise ValueError('Ambang confidence harus 0 sampai 1.')
    result = json.loads(input_path.read_text(encoding='utf-8'))
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    detections = polygon_boxes(result, image.size, minimum)
    if detections_override is not None:
        detections = detections_override
    output.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    payload = {'image': image_path.name, 'width': image.width, 'height': image.height,
               'model': result.get('model'), 'reviewed': False,
               'min_confidence': minimum, 'detections': detections}
    (output / f'{stem}.boxes.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    exportable = detections if detections_override is not None else [d for d in detections if not d['review_reasons']]
    (output / f'{stem}.candidates.xml').write_text(
        voc_xml(image_path.name, image.size, exportable, one_based=voc_one_based), encoding='utf-8')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=14)
    for detection in detections:
        color = '#00ed95' if detection['confidence'] >= 0.5 else '#ffbd38'
        if 'kontur_berjauhan_kotak_terlalu_luas' in detection['review_reasons']:
            color = '#ff5370'
        x1, y1, x2, y2 = detection['box']
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        draw.text((x1, max(0, y1-15)), str(detection['candidate_id']), font=font, fill=color,
                  stroke_width=1, stroke_fill='black')
    overlay_path = output / f'{stem}.boxes.jpg'
    image.save(overlay_path, quality=95)
    for row in range(3):
        for column in range(2):
            area = (column * image.width // 2, row * image.height // 3,
                    (column+1) * image.width // 2, (row+1) * image.height // 3)
            image.crop(area).save(output / f'{stem}.zoom-{row+1}-{column+1}.jpg', quality=95)
    (output / 'CATATAN-REVIEW.txt').write_text(
        'Semua kotak adalah kandidat AI yang BELUM diperiksa.\n'
        'Hijau: confidence >= 0.5; kuning: di bawah 0.5. Warna bukan bukti benar atau tajam.\n'
        'Merah: kotak gabungan jauh lebih luas dari komponen polygon terbesarnya.\n'
        'Satu polygon menjadi satu bounding box yang mencakup semua konturnya.\n'
        + ('XML kandidat berisi kotak terpilih dari job-label; ambang ada pada laporan.json.\n'
         if detections_override is not None else
         'XML kandidat mengecualikan confidence < 0.5 dan kotak merah; hasil tetap belum divalidasi manusia.\n')
        + 'Pembuatan gambar review tidak mengirim label. Untuk job-label, status pengiriman ada pada laporan.json.\n', encoding='utf-8')
    print(f'Kandidat bounding box: {len(detections)} (belum direview).')
    print(f'XML kandidat: {len(exportable)}; sisanya perlu pemeriksaan tambahan. Ini bukan hitungan ikan yang terverifikasi.')
    print('Gambar review:', overlay_path.resolve())
    print('Enam gambar zoom, JSON koordinat, dan XML kandidat tersimpan di:', output.resolve())
    return overlay_path
