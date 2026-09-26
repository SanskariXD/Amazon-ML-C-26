"""Label-free views: retain Unicode and provide deterministic Latin transliteration."""
import re
import unicodedata
from anyascii import anyascii


def text(value):
    value = unicodedata.normalize('NFKC', value or '').casefold()
    return ' '.join(''.join(c if unicodedata.category(c)[0] in 'LMN' else ' ' for c in value).split())


def record(row):
    out = dict(row)
    out['name'] = text(row['business_name'])
    out['address'] = text(row['business_address'])
    out['latin_name'] = text(anyascii(out['name']))
    out['latin_address'] = text(anyascii(out['address']))
    return out


def numbers(value):
    return set(re.findall(r'\d+', value))
