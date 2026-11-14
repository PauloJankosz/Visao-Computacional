# Registro de validação do protótipo — 14/11/2026

## Executado neste ambiente

- Sintaxe de todos os módulos (`python -m compileall`): **aprovada**.
- 9 testes automatizados com unittest: **9 passaram**.
- Geração de vídeo sintético com três objetos cruzando uma linha: **aprovada**.
- Pipeline clássico MOG2, rastreamento por centroide e contagem: **3 cruzamentos detectados / 3 planejados** no vídeo sintético.
- Exportação de vídeo de saída MP4, eventos CSV e resumo JSON: **aprovada**.
- Comparação com anotação manual sintética (`manual_example.csv`): erro absoluto **0** neste caso artificial.
- Adaptador YOLO + ByteTrack: chamadas e conversão de saída verificadas com objetos simulados no teste unitário (stub). **Não equivale a rodar YOLO com pesos reais**.

## Ainda não validado

- Inferência de YOLO11/ByteTrack com pesos reais: biblioteca `ultralytics` não instalada e rede indisponível no ambiente de validação.
- Desempenho de contagem em vídeos de rodovias reais: depende de obter MP4 e contagem manual.
- Estimativa métrica de velocidade: depende de calibração em metros e validação contra referência confiável.
- Disponibilidade dos endpoints HLS do DER-SP: não confirmada.
- Precisão da classificação dos veículos: não avaliada; o método clássico apenas estima porte em pixels.

## Dados necessários para fechar o experimento

1. Pelo menos dois ou três vídeos **contínuos** de uma câmera fixa, com resolução, FPS e autorização de uso.
2. Uma linha de contagem definida por dois pontos de pixel em cada câmera.
3. Anotação manual do total de veículos que cruzam essa linha (por sentido, se aplicável).
4. Para velocidade em km/h: pares de pontos de imagem com posições **medidas** no plano da pista.
5. Enunciado/rubrica de avaliação da disciplina, se houver outros requisitos de entrega.

Nenhuma métrica de precisão ou velocidade real deve ser divulgada com base apenas nos testes sintéticos.
