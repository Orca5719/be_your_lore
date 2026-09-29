"""Real entry integration check in an isolated temporary lore/index/store."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import entry
from encoder import Encoder
from entries import EntryStore
from index_store import build_index,load_index
from retrieval import Retriever

ROOT = Path(__file__).resolve().parents[1]
original = (ROOT/'data/index/CURRENT').read_bytes()
with tempfile.TemporaryDirectory(prefix='worldcheck-entry-') as tmp:
    directory = Path(tmp)
    lore = directory/'lore'
    lore.mkdir()
    for source in (ROOT/'lore').glob('*.md'):
        if source.name != entry.GENERATED:
            shutil.copy2(source,lore/source.name)
    encoder = Encoder(device='cpu')
    index = directory/'index'
    build_index(lore,index,encoder)
    output = io.StringIO()
    previous_stdin = sys.stdin
    sys.stdin = io.StringIO('确认保存\n')
    try:
        with contextlib.redirect_stdout(output):
            code = entry.main(['雷能控制时间。','--time','第九话','--store',str(directory/'entries.json'),'--lore',str(lore),'--index',str(index),'--device','cuda','--json'])
    finally:
        sys.stdin = previous_stdin
    assert code == 0, output.getvalue()
    preview,saved = [json.loads(line) for line in output.getvalue().splitlines()]
    assert preview['status']=='pending_confirmation' and saved['status']=='saved' and saved['index_status']=='ready'
    assert saved['entry']['event_time']=='第九话'
    assert preview['classification']['entity']=='雷' and preview['classification']['category']=='能力'
    assert preview['existing']==[] and '生理结构' in preview['other_categories']
    matches = Retriever(index,encoder).search('雷能控制时间。')
    assert any(item['file']==entry.GENERATED and '雷能控制时间。' in item['text'] for item in matches)
    _,metadata = load_index(index)
    chunks = [x for x in metadata['chunks'] if x['file'] != entry.GENERATED]
    after = EntryStore(directory/'entries.json').preview('雷','能力','新能力',chunks)
    assert len(after['existing'])==1
    report = dict(passed=True,preview=preview,saved=saved,reloaded_existing=after['existing'],retrieved=matches,original_index_unchanged=(ROOT/'data/index/CURRENT').read_bytes()==original)
    assert report['original_index_unchanged']
    (ROOT/'reports/entry_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('真实录入确认/保存/重建/检索通过；原资料索引未改变。')

