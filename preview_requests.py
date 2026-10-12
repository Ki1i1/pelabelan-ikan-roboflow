"""Pacing dan retry terbatas untuk deteksi preview; tidak menulis anotasi."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import time


def http_status(error):
    status = getattr(error, 'status_code', None)
    response = getattr(error, 'response', None)
    if status is None and response is not None:
        status = getattr(response, 'status_code', None)
    return status


def retry_after_seconds(error):
    # SDK saat ini biasanya hanya memberi status_code, tanpa headers.
    response = getattr(error, 'response', None)
    headers = getattr(response, 'headers', {}) if response is not None else {}
    raw = headers.get('Retry-After')
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        try:
            date = parsedate_to_datetime(raw)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            seconds = (date-datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0, seconds) if math.isfinite(seconds) else None


class PreviewRequester:
    def __init__(self, request_delay=5.0, max_retries=2, max_retry_wait=180):
        if (not math.isfinite(request_delay) or request_delay < 0 or max_retries < 0
                or not math.isfinite(max_retry_wait) or max_retry_wait < 0):
            raise ValueError('Jeda/retry tidak valid.')
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.max_retry_wait = max_retry_wait
        self.last_finished = None
        self.stats = {'api_calls':0, 'retries':0, 'api_seconds':0.0,
                      'pacing_wait_seconds':0.0, 'rate_limit_wait_seconds':0.0}

    @staticmethod
    def wait(seconds, progress=False):
        # Tampilkan kemajuan saat cooldown panjang dan mudah dihentikan Ctrl+C.
        while seconds > 0:
            chunk = min(30, seconds)
            time.sleep(chunk)
            seconds -= chunk
            if progress and seconds > 0:
                print(f'Menunggu batas API: {math.ceil(seconds)} detik lagi.', flush=True)

    def preview(self, project, image_path, prompt, confidence):
        for attempt in range(self.max_retries+1):
            if self.last_finished is not None:
                delay = self.request_delay-(time.monotonic()-self.last_finished)
                if delay > 0:
                    self.stats['pacing_wait_seconds'] += delay
                    self.wait(delay)
            started = time.monotonic()
            self.stats['api_calls'] += 1
            try:
                result = project.autolabel_preview('sam3-polygon', str(image_path),
                                                  ontology={prompt:'ikan'},
                                                  confidence_threshold=confidence)
            except Exception as error:
                self.last_finished = time.monotonic()
                self.stats['api_seconds'] += self.last_finished-started
                if http_status(error) != 429 or attempt == self.max_retries:
                    if http_status(error) == 429:
                        print('Batas API masih aktif setelah retry. Cache tetap tersimpan; '
                              'lanjutkan gambar ini nanti dengan parameter yang sama.', flush=True)
                    raise
                server_delay = retry_after_seconds(error)
                if server_delay is not None and server_delay > self.max_retry_wait:
                    print(f'Server meminta jeda {math.ceil(server_delay)} detik. '
                          'Proses berhenti; lanjutkan setelah jeda tersebut.', flush=True)
                    raise
                cooldown = max(min(60*2**attempt,300),server_delay or 0)
                if self.stats['rate_limit_wait_seconds'] + cooldown > self.max_retry_wait:
                    print('Batas total waktu tunggu 429 tercapai. Proses berhenti; '
                          'cache tersimpan untuk dilanjutkan nanti.', flush=True)
                    raise
                print(f'HTTP 429: tunggu {math.ceil(cooldown)} detik, lalu retry '
                      f'{attempt+1}/{self.max_retries} pada potongan yang sama.', flush=True)
                self.stats['retries'] += 1
                self.stats['rate_limit_wait_seconds'] += cooldown
                self.wait(cooldown, progress=True)
            else:
                self.last_finished = time.monotonic()
                self.stats['api_seconds'] += self.last_finished-started
                return result
