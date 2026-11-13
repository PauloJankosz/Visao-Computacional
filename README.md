# Análise de veículos em rodovia

Ferramenta que analisa o tráfego das câmeras ao vivo do DER-SP. Para cada veículo que passa, ela detecta, acompanha ao longo dos quadros, classifica o tipo e estima a velocidade.

Os mesmos vídeos são processados por duas abordagens, para comparação:

- **Clássica:** subtração de fundo (MOG2), morfologia, contornos e rastreamento por centroide. A classificação é feita pelo tamanho do veículo na vista de cima da pista.
- **Aprendizado profundo:** detector YOLO (classes carro, moto, ônibus e caminhão) com rastreador ByteTrack.

## Fonte dos dados

Câmeras do DER-SP, publicadas em `http://200.144.30.103:8084/`.

| Recurso | Endereço |
| --- | --- |
| Lista de câmeras (JSON) | `http://200.144.30.103:8084/api/cameras` |
| Vídeo de uma câmera (HLS) | `http://200.144.30.103:8084/hls/cam_<id>/stream.m3u8` |

O vídeo tem 704×480 pixels a 30 quadros por segundo e abre direto no OpenCV com `cv2.VideoCapture(url)`. O servidor é lento e às vezes fica fora do ar. Por isso o desenvolvimento e as medições são feitos sobre trechos gravados (`data/clips/`), e o modo ao vivo fica para a demonstração.


## Estrutura

```
main.py                  linha de comando (gravar, calibrar, processar, comparar)
config/                  câmeras escolhidas e pontos de calibração
rodovia/
  source.py              lista de câmeras e abertura do vídeo (ao vivo ou arquivo)
  recorder.py            gravação de trechos
  calibration.py         homografia imagem -> metros
  detectors/
    classic.py           subtração de fundo
    yolo.py              YOLO
  tracker.py             rastreamento por centroide (abordagem clássica)
  classify.py            classificação por tamanho (abordagem clássica)
  speed.py               estimativa de velocidade
  evaluate.py            comparação com a contagem manual
  visualize.py           desenho das caixas, trajetórias e velocidades
data/clips/              trechos gravados (não versionado)
data/annotations/        contagem manual usada na avaliação
results/                 vídeos processados, tabelas e figuras
```

## Instalação

```
pip install -r requirements.txt
```
