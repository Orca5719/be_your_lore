"""Benchmark 5: offline audit and explicit user-run 5B experiments."""
import argparse
from datetime import datetime
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description='Benchmark 5 audit and short-wire experiment')
    sub=parser.add_subparsers(dest='command',required=True)
    audit=sub.add_parser('audit')
    audit.add_argument('--source-root',type=Path,required=True)
    audit.add_argument('--output',type=Path)
    replay=sub.add_parser('replay-c')
    replay.add_argument('--result-dir',type=Path,required=True)
    replay.add_argument('--output',type=Path)
    for name in ('validate-b','run-b','validate-c','run-c','validate-d','run-d'):
        cmd=sub.add_parser(name)
        cmd.set_defaults(experiment={'b':'5B','c':'5C1','d':'5D0'}[name[-1]])
        cmd.add_argument('--source-root',type=Path,default=Path('C:/Users/Xhang/Desktop/project/worldcheck'))
        cmd.add_argument('--asset-root',type=Path,default=Path('C:/Users/Xhang/Desktop/project/worldcheck'))
        cmd.add_argument('--phase',choices=('screen','full'),default='screen')
        cmd.add_argument('--order',choices=('forward','reverse'),default='forward')
        if name.startswith('run-'):cmd.add_argument('--output',type=Path)
    for name in ('summary-b','score-b','run-downstream-b','summary-c','score-c','run-downstream-c','summary-d','score-d','run-downstream-d'):
        cmd=sub.add_parser(name)
        cmd.add_argument('--result-dir',type=Path,required=True)
        if not name.startswith('summary-'):
            from .runner import ARMS,C1_ARMS,D0_ARMS
            cmd.add_argument('--arms',nargs='+',choices={'b':ARMS,'c':C1_ARMS,'d':D0_ARMS}[name[-1]])
    args=parser.parse_args(argv)
    try:
        if args.command=='audit':
            from .diagnostics import audit_archives
            output=args.output or Path(__file__).parent/'reports'/('audit_5A_'+datetime.now().strftime('%Y%m%dT%H%M%S'))
            audit_archives(args.source_root,output)
            print('SUMMARY='+str((output/'audit_summary.md').resolve()))
        elif args.command=='replay-c':
            from .repair import replay_archive
            output=args.output or Path(__file__).parent/'reports'/('replay_5C1_'+datetime.now().strftime('%Y%m%dT%H%M%S'))
            replay_archive(args.result_dir,output)
        elif args.command.startswith('validate-'):
            from .experiment import prepare
            from .resources import check_resources
            cases,identity=prepare(args.source_root,args.asset_root,args.phase,args.order,experiment=args.experiment)
            resource=check_resources(Path(__file__).parent/'reports',initial=True)
            print(f'VALIDATED {len(cases)} stories; four arms; shared local cache; no model loaded')
            print('Shared assets: '+identity['config']['asset_root'])
            print('Disk check: '+str(resource))
        elif args.command in ('run-b','run-c','run-d'):
            from .experiment import run_experiment
            return run_experiment(args)
        elif args.command.startswith('summary-'):
            from .experiment import summarize
            summarize(args.result_dir)
        elif args.command.startswith('score-'):
            from .quality import score
            score(args.result_dir,args.arms)
        else:
            from .quality import run_downstream
            return run_downstream(args)
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:
        parser.exit(2,f'错误：{exc}\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
