"""用真实缓存模型检查 CLI 建库、JSON 和连续查询。"""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT=Path(__file__).resolve().parents[1]

def run(*args,input_text=None):
    result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'worldcheck.py'),*args],input=input_text,capture_output=True,text=True,encoding='utf-8',timeout=180)
    assert result.returncode==0,(result.returncode,result.stderr)
    return result

with TemporaryDirectory() as directory:
    index=Path(directory)/'index'
    built=run('index','--index',str(index),'--json')
    summary=json.loads(built.stdout)
    assert summary['shape'][0]>=40 and summary['shape'][1]==512
    one=run('search','雷感觉左胸里的亚巴顿开始躁动。','--index',str(index),'--json','--k','5')
    single=json.loads(one.stdout)
    assert len(single['results'])==5
    assert any('左侧心脏寄宿亚巴顿' in r['text'] for r in single['results'])
    session=run('interactive','--index',str(index),'--json','--k','5',input_text='雷感觉左胸里的亚巴顿开始躁动。\n\n德尔塔装甲靠什么供电？\n/quit\n不应该执行这条查询\n')
    lines=[json.loads(line) for line in session.stdout.splitlines()]
    assert len(lines)==2,'会话退出后仍继续查询，或错误污染了 JSON'
    assert lines[0]==single,'单次查询与连续查询结果不同'
    assert '查询不能为空' in session.stderr
    assert '德尔塔' in lines[1]['results'][0]['text']
    eof=run('interactive','--index',str(index),'--json',input_text='')
    assert eof.stdout=='','EOF 应干净退出'
    report={'index_summary':summary,'single_search':single,'interactive_results':lines,'empty_query_error':session.stderr.strip(),'eof_exit_code':eof.returncode,'checks':'real index, JSON parse, two queries, recovery after blank input, /quit, EOF: passed'}
    (ROOT/'reports/cli_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('真实 CLI 检查通过：建库、单次 JSON、两次连续查询、空输入恢复、/quit 和 EOF。')

