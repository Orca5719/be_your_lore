"""Generalized real ingestion in isolated data: multi items and broad rules."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import world_entry
from encoder import Encoder
from index_store import build_index,load_index
from world_records import WorldStore
from retrieval import Retriever
ROOT=Path(__file__).resolve().parents[1]
source='第九话，雷加入晨星议会，并获得控制时间的能力。'
original=(ROOT/'data/index/CURRENT').read_bytes()
with tempfile.TemporaryDirectory(prefix='worldcheck-general-') as tmp:
    directory=Path(tmp)
    lore=directory/'lore'
    lore.mkdir()
    for name in ['characters.md','angels.md','history.md','technology.md']:
        shutil.copy2(ROOT/'lore'/name,lore/name)
    session={'encoder':Encoder(device='cpu')}
    index=directory/'index'
    build_index(lore,index,session['encoder'])
    options=['--lore',str(lore),'--index',str(index),'--store',str(directory/'world_records.json'),'--legacy-store',str(directory/'legacy.json'),'--device','cuda','--json']
    output=io.StringIO()
    with contextlib.redirect_stdout(output):
        code=world_entry.main([source]+options+['--preview-only'],session)
    preview=json.loads(output.getvalue())
    (ROOT/'reports/general_preview_debug.json').write_text(json.dumps(preview,ensure_ascii=False,indent=2),encoding='utf-8')
    assert code==0, preview
    kinds={item['proposed']['kind'] for item in preview['items']}
    assert {'entity','timepoint','relation','attribute'}<=kinds,kinds
    assert all(item['review']['verdict']=='不确定' for item in preview['items'])
    assert not (directory/'world_records.json').exists()
    # Confirm the exact validated preview once; no extra generation or changing proposals.
    store=WorldStore(directory/'world_records.json')
    saved=store.commit(preview)
    store.export(lore/world_entry.GENERATED)
    build_index(lore,index,session['encoder'])
    results=Retriever(index,session['encoder']).search('雷控制时间的能力。')
    assert any(row['file']==world_entry.GENERATED and '控制时间' in row['text'] for row in results)
    assert any(record['kind']=='attribute' and record['valid_from']=='第九话' for record in saved)
    second='黎明城位于北方。灵能只能在月光下使用。'
    output=io.StringIO()
    with contextlib.redirect_stdout(output):
        code=world_entry.main([second]+options+['--preview-only'],session)
    broad=json.loads(output.getvalue())
    (ROOT/'reports/general_broad_debug.json').write_text(json.dumps(broad,ensure_ascii=False,indent=2),encoding='utf-8')
    assert code==0,broad
    assert any(item['proposed']['kind']=='entity' and item['proposed']['entity']=='黎明城' for item in broad['items'])
    assert any(item['proposed']['kind']=='rule' for item in broad['items'])
    report=dict(passed=True,preview=preview,broad_preview=broad,saved=saved,retrieved=results,original_index_unchanged=(ROOT/'data/index/CURRENT').read_bytes()==original,note='隔离开发验证，真实模型输出；当前资料未添加测试组织/能力/规则。')
    assert report['original_index_unchanged']
    (ROOT/'reports/general_entry_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('真实通用预览/多条保存/时间范围/索引查回与地点规则验证通过。')

