"""CLI do projeto de análise comparativa de veículos em rodovias."""
import argparse
import json
from pathlib import Path


def parser_builder():
    p = argparse.ArgumentParser(description='Análise de veículos: MOG2 ou YOLO + ByteTrack')
    sub = p.add_subparsers(dest='action', required=True)
    sub.add_parser('cameras', help='Consulta catálogo de câmeras DER (depende do serviço externo)')
    rec = sub.add_parser('record', help='Grava trecho de stream ou arquivo')
    rec.add_argument('--source', required=True)
    rec.add_argument('--output', default='data/clips/gravacao.mp4')
    rec.add_argument('--seconds', type=float, default=20)
    demo = sub.add_parser('demo', help='Gera vídeo artificial para testar o pipeline')
    demo.add_argument('--output', default='data/clips/demo_sintetico.mp4')
    pro = sub.add_parser('process', help='Processa vídeo e exporta eventos, vídeo e métricas')
    pro.add_argument('--source', required=True, help='arquivo .mp4 ou URL HLS')
    pro.add_argument('--method', choices=['classic','yolo'], required=True)
    pro.add_argument('--output-dir', default='results')
    pro.add_argument('--model', default='yolo11n.pt', help='pesos YOLOv11 COCO; auto download se necessário')
    pro.add_argument('--line', nargs=4, type=int, metavar=('X1','Y1','X2','Y2'))
    pro.add_argument('--direction', choices=['both','a_to_b','b_to_a'],default='both')
    pro.add_argument('--calibration', help='JSON com correspondências imagem-metros')
    pro.add_argument('--min-area', type=int, default=250)
    pro.add_argument('--max-distance', type=float, default=70)
    pro.add_argument('--conf', type=float, default=0.25)
    pro.add_argument('--max-frames', type=int)
    pro.add_argument('--device', help='cpu, 0, etc. (YOLO)')
    pro.add_argument('--no-video', action='store_true')
    cal = sub.add_parser('calibrate', help='Calcula e salva homografia de um JSON de pontos')
    cal.add_argument('--input', required=True)
    cal.add_argument('--output', default='config/calibration.json')
    ev = sub.add_parser('evaluate', help='Compara contagens ao CSV manual (video,total)')
    ev.add_argument('--annotations', required=True)
    ev.add_argument('--summaries', nargs='+', required=True)
    ev.add_argument('--output', default='results/evaluation.csv')
    comp = sub.add_parser('compare', help='Executa os dois métodos no mesmo vídeo')
    comp.add_argument('--source', required=True)
    comp.add_argument('--output-dir', default='results')
    comp.add_argument('--line', nargs=4, type=int)
    comp.add_argument('--model', default='yolo11n.pt')
    comp.add_argument('--calibration')
    comp.add_argument('--max-frames', type=int)
    comp.add_argument('--no-video', action='store_true')
    return p


def main(args=None):
    opts = parser_builder().parse_args(args)
    if opts.action == 'cameras':
        from rodovia.source import list_cameras
        print(json.dumps(list_cameras(),ensure_ascii=False,indent=2))
    elif opts.action == 'record':
        from rodovia.recorder import record
        print(f'Frames gravados: {record(opts.source,opts.output,opts.seconds)}')
    elif opts.action == 'demo':
        from rodovia.demo import generate_demo
        print(json.dumps(generate_demo(opts.output),indent=2,ensure_ascii=False))
    elif opts.action == 'calibrate':
        from rodovia.calibration import Calibration
        data = json.loads(Path(opts.input).read_text(encoding='utf-8'))
        cal = Calibration.from_correspondences(data['image_points'],data['world_points_m'])
        cal.save(opts.output)
        print(f'Homografia salva em {opts.output}')
    elif opts.action == 'evaluate':
        from rodovia.evaluate import compare_summaries,evaluate_counts,write_csv
        results = evaluate_counts(opts.annotations,compare_summaries(opts.summaries))
        write_csv(results,opts.output)
        print(json.dumps(results,ensure_ascii=False,indent=2))
    else:
        from rodovia.processor import process
        kwargs = dict(source=opts.source, output_dir=opts.output_dir, line=opts.line,
                      model=opts.model, calibration_path=opts.calibration,
                      max_frames=opts.max_frames, no_video=opts.no_video)
        if opts.action == 'process':
            kwargs.update(method=opts.method,direction=opts.direction,
                          min_area=opts.min_area,max_distance=opts.max_distance,
                          conf=opts.conf,device=opts.device)
            print(json.dumps(process(**kwargs),ensure_ascii=False,indent=2))
        else:
            summaries = [process(method=m, **kwargs) for m in ('classic','yolo')]
            print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
