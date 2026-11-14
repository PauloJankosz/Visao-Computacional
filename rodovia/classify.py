"""Classificação aproximada por porte para baseline clássico.

Os limiares são em pixels e precisam ser calibrados por câmera. NÃO
identificam carro/moto/ônibus/caminhão; só dão uma categoria de porte.
"""


def classify_size(bbox, small=1200, large=4000):
    x1, y1, x2, y2 = bbox
    area = max(0, x2-x1) * max(0, y2-y1)
    if area < small:
        return 'porte_pequeno'
    if area < large:
        return 'porte_medio'
    return 'porte_grande'
