"""Comparação das contagens com anotação manual por vídeo."""
import csv
import json
from pathlib import Path


def evaluate_counts(annotations_csv, results):
    """CSV: video,total. results: lista de dict com source,method,total."""
    with open(annotations_csv, newline='', encoding='utf-8') as handle:
        annotations = {row['video']: int(row['total'])
                       for row in csv.DictReader(handle)}
    if not annotations:
        raise ValueError('Arquivo de anotações vazio')
    reports = []
    for item in results:
        key = Path(item['source']).name
        if key not in annotations:
            continue
        expected = annotations[key]
        predicted = int(item['total'])
        error = abs(predicted - expected)
        reports.append({'video': key, 'method': item['method'],
                        'reference': expected, 'predicted': predicted,
                        'absolute_error': error,
                        'relative_error_pct': (100*error/expected if expected else None)})
    if not reports:
        raise ValueError('Nenhuma contagem corresponde aos nomes do CSV manual')
    return reports


def compare_summaries(paths):
    results = [json.loads(Path(p).read_text(encoding='utf-8')) for p in paths]
    if not results:
        raise ValueError('Informe ao menos um resumo')
    return results


def write_csv(rows, path):
    if not rows:
        raise ValueError('Sem resultados para exportar')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
