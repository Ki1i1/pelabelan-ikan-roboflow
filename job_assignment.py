"""Pemeriksaan read-only gambar dalam assignment Roboflow."""
from __future__ import annotations


def shape(value, depth=0):
    """Skema tanpa nilai URL, credential, atau informasi pengguna."""
    if depth > 3:
        return type(value).__name__
    if isinstance(value, dict):
        return {key: shape(item, depth+1) for key, item in value.items()}
    if isinstance(value, list):
        return {'type': 'list', 'length': len(value), 'item': shape(value[0], depth+1) if value else None}
    return type(value).__name__


def page_items(page):
    if isinstance(page, list):
        return page
    if isinstance(page, dict):
        for key in ('images', 'imageIds', 'items', 'results'):
            if isinstance(page.get(key), list):
                return page[key]
        if isinstance(page.get('data'), (dict, list)):
            return page_items(page['data'])
    raise ValueError('Format daftar gambar job belum dikenali.')


def next_cursor(page):
    if not isinstance(page, dict):
        return None
    for node in (page, page.get('pagination'), page.get('pageInfo')):
        if isinstance(node, dict):
            if node.get('hasMore') is False or node.get('hasNextPage') is False:
                continue
            for key in ('nextCursor', 'next', 'nextPageToken', 'endCursor'):
                cursor = node.get(key)
                if isinstance(cursor, str) and cursor:
                    return cursor
            if node.get('hasMore') is True or node.get('hasNextPage') is True:
                raise ValueError('Ada halaman lanjutan tanpa cursor yang dikenali.')
    return None


def scan_job(project, job_id):
    images = {}
    cursor = None
    seen_cursors = set()
    first_schema = None
    for _ in range(100):
        page = project.get_annotation_job_images(job_id, limit=50, after=cursor)
        if first_schema is None:
            first_schema = shape(page)
        try:
            entries = page_items(page)
        except ValueError:
            return {'job_id': job_id, 'complete': False, 'response_schema': first_schema,
                    'note': 'Daftar gambar perlu disesuaikan dengan format respons API.'}
        for entry in entries:
            if isinstance(entry, str):
                record = {'image_id': entry}
            elif isinstance(entry, dict):
                identifier = entry.get('id') or entry.get('imageId') or entry.get('image_id')
                if not isinstance(identifier, str):
                    return {'job_id': job_id, 'complete': False, 'response_schema': first_schema,
                            'note': 'Field ID gambar belum dikenali.'}
                record = {'image_id': identifier}
                for field in ('name', 'filename', 'annotationStatus', 'status', 'annotationCount', 'width', 'height'):
                    value = entry.get(field)
                    if isinstance(value, (str, int, bool)):
                        record[field] = value
            else:
                raise ValueError('Entri gambar job tidak valid.')
            images[record['image_id']] = record
        cursor = next_cursor(page)
        if not cursor:
            break
        if cursor in seen_cursors:
            raise ValueError('Cursor berulang; daftar job tidak lengkap.')
        seen_cursors.add(cursor)
    else:
        raise ValueError('Batas halaman terlampaui; daftar job tidak lengkap.')
    result = {'job_id': job_id, 'complete': True, 'image_count': len(images),
              'images': list(images.values()), 'response_schema': first_schema,
              'annotation_writes': 0}
    if images:
        # Inspect one image's shape to implement annotation guards against actual API data.
        sample_id = next(iter(images))
        details = project.image(sample_id)
        sample = {'image_id': sample_id, 'response_schema': shape(details)}
        for field in ('name', 'filename', 'width', 'height', 'annotationStatus', 'status', 'annotationCount'):
            value = details.get(field)
            if isinstance(value, (str, int, bool)):
                sample[field] = value
        for field in ('annotations', 'labels'):
            value = details.get(field)
            if isinstance(value, list):
                sample[field + '_length'] = len(value)
        result['sample_image'] = sample
    return result


def empty_annotation(details):
    """True hanya untuk skema kosong yang telah diamati pada API akun ini."""
    if 'annotation' not in details or 'labels' not in details:
        return False
    if details['annotation'] is not None:
        return False
    if details['labels'] not in (None, [], {}):
        return False
    if details.get('annotations'):
        return False
    return True
