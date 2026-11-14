# Análise de veículos em rodovias — Trabalho Final de Visão Computacional

Implementação do esqueleto recebido do grupo (13/11/2026). Compara duas abordagens sobre **os mesmos vídeos**:

- **Clássica:** OpenCV MOG2 → binarização e morfologia → contornos → rastreador por centroide → contagem por cruzamento de linha. A classificação clássica indica apenas **porte aparente em pixels** (pequeno, médio, grande), não o tipo real do veículo.
- **Aprendizado profundo:** **YOLO11**, pesos pré-treinados nas classes COCO carro/moto/ônibus/caminhão → **ByteTrack** (IDs) → contagem pela mesma linha.

Opcional: estimativa de velocidade em km/h via homografia **calibrada em metros**. Sem calibração métrica, o sistema não exibe velocidade; mesmo com calibração, trata-se de estimativa experimental, não de medição válida para fiscalização.

## 1. Instalação (Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

A instalação do `ultralytics` traz PyTorch (volume de download depende do ambiente). Se quiser testar **somente MOG2** antes disso, basta instalar `numpy opencv-python`. A primeira execução de YOLO tentará baixar `yolo11n.pt`, exigindo Internet. A versão está restrita a YOLO11, como indicado na planilha da disciplina.

## 2. Teste local, sem dataset externo

```bash
python main.py demo --output data/clips/demo_sintetico.mp4
python main.py process --method classic --source data/clips/demo_sintetico.mp4 \
  --line 0 180 640 180 --min-area 180
```

Vídeo sintético com **3 objetos** atravessando horizontalmente a linha y=180; testa encadeamento dos módulos. **Não representa validação científica** de detecção de veículos nem substitui vídeos reais.

```bash
python main.py process --method yolo --source data/clips/demo_sintetico.mp4 \
  --line 0 180 640 180 --model yolo11n.pt
```

**Atenção:** YOLO11 COCO não foi treinado para os retângulos sintéticos; não espere três detecções nesse teste. Execute YOLO sobre um vídeo **real**.

## 3. Vídeo de rodovia

```bash
# Coloque video_real.mp4 em data/clips/
python main.py process --method classic --source data/clips/video_real.mp4 \
  --line 60 235 650 235 --min-area 350
python main.py process --method yolo --source data/clips/video_real.mp4 \
  --line 60 235 650 235 --model yolo11n.pt
# Ou compare ambos no mesmo vídeo, com mesmos parâmetros da linha
python main.py compare --source data/clips/video_real.mp4 --line 60 235 650 235
```

**A linha é EXEMPLO**, não foi determinada para nenhuma câmera real. Ajuste após observar os frames. Na abordagem clássica, `--min-area` e `--max-distance` também precisam ser ajustados por câmera. A linha possui orientação: `a_to_b` significa passar do lado negativo para o positivo no produto vetorial dos dois extremos. A opção padrão conta em ambas as direções, uma vez por ID.

Saídas em `results/`:

- `<video>_<method>.mp4` — vídeo com caixas, IDs, linha e contador;
- `<video>_<method>_events.csv` — cruzamentos, tempo, classe/porte e velocidade (se calibrado);
- `<video>_<method>_summary.json` — contagem total, contagem por categoria, desempenho de processamento e configuração de linha.

⚠️ **A classificação de classes não é comparável diretamente entre os métodos:** no método clássico são portes em pixels; no YOLO são classes COCO. Compare primeiro a **contagem total**, e só compare subclasses com rótulos manuais compatíveis.

## 4. Avaliação quantitativa com gabarito manual

Crie `data/annotations/manual.csv`, com ao menos:

```csv
video,total
video_real.mp4,47
```

Os números acima **são fictícios**, para mostrar o formato. Conte manualmente os cruzamentos de veículos na linha e no sentido escolhido. Compare:

```bash
python main.py evaluate --annotations data/annotations/manual.csv \
  --summaries results/video_real_classic_summary.json results/video_real_yolo_summary.json \
  --output results/evaluation.csv
```

O relatório apresenta erro absoluto e erro relativo. Não calcula precisão, recall nem IDF1 sem anotações individuais compatíveis; não atribua esses valores ao trabalho sem dataset anotado.

## 5. Calibrar homografia (opcional)

A homografia exige **pelo menos quatro pares de correspondências não colineares**: pontos de imagem `[x,y]` e suas posições reais no solo `[X,Y]`, **em metros**. Há um JSON puramente ilustrativo em `config/calibration.example.json` que **não pode ser usado como calibração real**.

```bash
python main.py calibrate --input config/minha_medicao_real.json --output config/calibration.json
python main.py process --method yolo --source data/clips/video_real.mp4 \
  --line 60 235 650 235 --calibration config/calibration.json
```

A câmera precisa ser razoavelmente fixa, o trecho da estrada aproximável por um plano, os pontos precisam ter medições confiáveis e a taxa de quadros deve ser conhecida. A velocidade usa o centro inferior da caixa delimitadora como aproximação do contato com o asfalto; erros de detecção/oclusão/calibração alteram o valor.

## 6. Fontes ao vivo (opcional, **não validadas**)

O README original indicava `http://200.144.30.103:8084/api/cameras` e URLs `http://200.144.30.103:8084/hls/cam_<id>/stream.m3u8`. **A disponibilidade e a autorização de uso dos endpoints não foram confirmadas aqui.** O projeto não depende deles para funcionar com MP4.

```bash
python main.py cameras
python main.py record --source 'http://200.144.30.103:8084/hls/cam_14/stream.m3u8' \
  --output data/clips/rodovia.mp4 --seconds 20
```

Se a câmera transmitir fotografias espaçadas, não será possível tratá-la como vídeo contínuo para tracking/velocidade. Verifique o stream e os termos do serviço antes da coleta.

## 7. Testes e limitações

```bash
python -m unittest discover -s tests -v
python -m compileall -q main.py rodovia tests
```

Implementações de tracking podem trocar IDs sob oclusões; cada ID só é contabilizado uma vez. Trocas de ID podem criar falsos cruzamentos adicionais ou perdas. MOG2 exige câmera fixa e tende a errar com mudanças de iluminação, sombras e congestionamento. ByteTrack depende das detecções YOLO. A precisão real depende de anotação manual e análise de casos de falha. Os resultados do vídeo sintético são apenas testes de integração.

**Licenças e fontes**: consulte a licença da dependência Ultralytics e dos pesos antes de redistribuir/comercializar. Documentação oficial: https://docs.ultralytics.com/modes/track .
