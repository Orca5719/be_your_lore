"""WorldCheck：本地设定建库、单次查询与连续查询。"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent


def positive(value):
    try:
        number=int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('必须为正整数') from error
    if number<1:
        raise argparse.ArgumentTypeError('必须为正整数')
    return number


def token_budget(value):
    number=positive(value)
    if number<3 or number>512:
        raise argparse.ArgumentTypeError('token 预算必须在 3 到 512 之间')
    return number


def make_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for name,description in [('index','读取资料并建立索引'),('search','输入新剧情，检索一次'),('interactive','模型加载一次，连续查询'),('inspect','查看索引或文字的编码过程')]:
        command=commands.add_parser(name,help=description)
        command.add_argument('--index',type=Path,default=ROOT/'data/index',help='索引目录，默认使用项目 data/index')
        command.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
        if name!='inspect':
            command.add_argument('--max-tokens',type=token_budget,default=512,help='必须与建库预算一致，默认 512')
        command.add_argument('--json',action='store_true',help='输出 JSON；连续查询时每次一行')
        command.add_argument('--download',action='store_true',help='允许下载公开模型；默认只读取本地缓存')
        if name=='index':
            command.add_argument('--lore',type=Path,default=ROOT/'lore',help='Markdown/TXT 资料目录')
            command.add_argument('--batch-size',type=positive,default=16)
        elif name in {'search','interactive'}:
            command.add_argument('--k',type=positive,default=5)
        if name=='search':
            command.add_argument('query',help='一句或一小段新剧情')
        if name=='inspect':
            command.add_argument('text',nargs='?',help='不带文本查看索引；带文本查看编码过程')
            command.add_argument('--mode',choices=['passage','query'],default='passage',help='正文模式不加提示；query 模式加检索提示')
    benchmark=commands.add_parser('benchmark',help='比较 CPU/GPU、逐条/批量及 FP32/FP16')
    benchmark.add_argument('--index',type=Path,default=ROOT/'data/index')
    benchmark.add_argument('--reports',type=Path,default=ROOT/'reports')
    benchmark.add_argument('--repeats',type=positive,default=3)
    benchmark.add_argument('--warmups',type=positive,default=1)
    benchmark.add_argument('--threads',type=positive,default=4,help='CPU intra-op 线程数，默认 4')
    benchmark.add_argument('--json',action='store_true')
    return parser


def print_results(query,results,json_output):
    from retrieval import NOTICE
    if json_output:
        print(json.dumps({'query':query,'notice':NOTICE,'results':results},ensure_ascii=False),flush=True)
        return
    print('查询:',query)
    for rank,result in enumerate(results,1):
        print(f"\n[{rank}] score={result['score']:.6f} | {result['file']}:{result['start_line']}-{result['end_line']}")
        print('标题:',' > '.join(result['heading_path']))
        print(result['text'])
    print('\n'+NOTICE,flush=True)


def interactive(retriever,k,json_output):
    # 提示和错误去 stderr，不混入 JSON 结果。非终端输入不打印逐行提示。
    if sys.stdin.isatty():
        print('模型与索引已加载。输入剧情后回车；/quit 或 /exit 退出，Ctrl+C 也可退出。',file=sys.stderr)
    while True:
        if sys.stdin.isatty():
            print('剧情> ',end='',file=sys.stderr,flush=True)
        line=sys.stdin.readline()
        if not line:
            return 0
        query=line.strip()
        if query.lower() in {'/quit','/exit'}:
            return 0
        try:
            results=retriever.search(query,k)
        except ValueError as error:
            print(f'错误：{error}',file=sys.stderr,flush=True)
            continue
        print_results(query,results,json_output)


def inspect_index(directory):
    import numpy as np
    from index_store import load_index
    vectors,metadata=load_index(directory)
    norms=np.linalg.norm(vectors,axis=1)
    return {'type':'index','index':str(Path(directory).resolve()),'shape':list(vectors.shape),
            'storage_dtype':str(vectors.dtype),'config':metadata['config'],
            'source_files':sorted(metadata.get('sources',{})),
            'norm_range':[float(norms.min()),float(norms.max())]}


def print_inspection(report,json_output):
    if json_output:
        print(json.dumps(report,ensure_ascii=False))
        return
    if report['type']=='index':
        print('索引目录:',report['index'])
        print('向量矩阵:',report['shape'],report['storage_dtype'])
        print('向量长度范围:',report['norm_range'])
        print('资料文件:',', '.join(report['source_files']))
        print('建库配置:',json.dumps(report['config'],ensure_ascii=False,indent=2))
        return
    print('编码模式:',report['mode'],'设备:',report['device'],'类型:',report['dtype'])
    print('实际编码输入:',report['encoding_text'])
    for name in ['tokens','input_ids','attention_mask','input_shape','hidden_shape','cls_shape','cls_head','normalized_head','raw_norm','normalized_norm']:
        print(name+':',report[name])
    print('形状：[B,L] -> [B,L,D] -> CLS [B,D]。这里 B=1。')
    print('向量前 8 维仅展示数字，单个维度不等于一个可命名的设定属性。')


def main(argv=None):
    parser=make_parser()
    args=parser.parse_args(argv)
    if args.command=='search' and not args.query.strip():
        parser.error('查询不能为空')
    if args.command=='inspect' and args.text is not None and not args.text.strip():
        parser.error('检查文本不能为空')
    if args.command=='inspect' and args.text is None and args.mode=='query':
        parser.error('query 编码模式必须提供检查文本')
    try:
        if args.command=='benchmark':
            from benchmarking import run_benchmark
            run_benchmark(args.index,args.reports,args.repeats,args.warmups,args.threads,args.json)
            return 0
        if args.command=='inspect' and args.text is None:
            print_inspection(inspect_index(args.index),args.json)
            return 0
        # 参数验证完成后才导入并加载大型依赖。
        from encoder import Encoder
        from index_store import build_index,load_index
        from retrieval import Retriever
        encoder=Encoder(device=args.device,offline=not args.download)
        if args.command=='inspect':
            print_inspection(encoder.inspect_text(args.text,args.mode),args.json)
            return 0
        if args.command=='index':
            path=build_index(args.lore,args.index,encoder,max_tokens=args.max_tokens,batch_size=args.batch_size)
            vectors,metadata=load_index(args.index)
            summary={'version':path.name,'index':str(args.index.resolve()),'shape':list(vectors.shape),'device':encoder.device,'source_files':len(metadata['sources'])}
            if args.json:
                print(json.dumps(summary,ensure_ascii=False))
            else:
                print(f"建库完成：{vectors.shape[0]} 个片段，{vectors.shape[1]} 维，设备 {encoder.device}")
                print('索引目录:',args.index.resolve())
            return 0
        retriever=Retriever(args.index,encoder,max_tokens=args.max_tokens)
        if args.command=='search':
            print_results(args.query,retriever.search(args.query,args.k),args.json)
            return 0
        return interactive(retriever,args.k,args.json)
    except (ValueError,OSError) as error:
        print(f'错误：{error}',file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('\n已退出。',file=sys.stderr)
        return 0


if __name__=='__main__':
    raise SystemExit(main())
