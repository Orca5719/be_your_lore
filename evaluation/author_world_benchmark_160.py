"""Author the benchmark before any model run; evidence anchors are exact source lines."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
initial=json.loads((ROOT/'evaluation/world_benchmark_batch01_draft.json').read_text(encoding='utf-8'))
source={f:(ROOT/'lore'/f).read_text(encoding='utf-8-sig').splitlines() for f in initial['fixture_hashes']}
anchors={
'heart':('characters.md','雷拥有两颗心脏。'), 'pain':('characters.md','雷曾在大地震期间失去意识。'), 'warm':('characters.md','拉古艾尔施展治疗时，雷的右胸'), 'secret':('characters.md','雷隐藏自己是双宿主'), 'armor':('characters.md','德尔塔使用动力外骨骼'), 'public':('characters.md','德尔塔以装甲行动者'), 'cool':('characters.md','德尔塔的装甲连续高功率'), 'arm':('characters.md','月城失去的是左臂'), 'surgery':('characters.md','月城在2064年'), 'father':('characters.md','雷的父亲不知道'), 'lia':('characters.md','莉娅是城区档案馆'), 'blind':('characters.md','监察官在爆炸后'),
'ab':('angels.md','亚巴顿寄宿在雷'), 'early':('angels.md','亚巴顿在大地震时'), 'memory':('angels.md','雷借用亚巴顿'), 'rag':('angels.md','拉古艾尔寄宿在雷'), 'heal':('angels.md','拉古艾尔可以加速'), 'contract':('angels.md','宿主与天使建立契约'), 'leave':('angels.md','寄宿天使离开宿主'), 'watch':('angels.md','守望者通过契约'), 'seal':('angels.md','地下神殿的封印'), 'guise':('angels.md','天使可以隐藏羽翼'), 'resonate':('angels.md','雷的两颗心脏同时'), 'church':('angels.md','旧教堂的静默结界'),
'quake':('history.md','2063年大地震'), 'lost':('history.md','月城在大地震救援'), 'install':('history.md','地震一年后'), 'news':('history.md','第一话广场新闻屏幕'), 'meet':('history.md','雷与德尔塔在第二话'), 'disclose':('history.md','雷在第三话'), 'sword':('history.md','雷在第四话'), 'broken':('history.md','雷的长剑在第八话'), 'depart':('history.md','雷的父亲在第五话'), 'shelter':('history.md','雷与德尔塔在第六话'), 'outage':('history.md','地震前一年'), 'open':('history.md','城区档案馆在2065年'),
'battery':('technology.md','德尔塔的外骨骼采用'), 'heat':('technology.md','外骨骼推进器产生'), 'neural':('technology.md','月城的机械左臂通过'), 'touch':('technology.md','月城的机械左臂能反馈'), 'radio':('technology.md','金属屏蔽室'), 'tunnel':('technology.md','旧城区地下隧道'), 'train':('technology.md','城区列车每天'), 'solar':('technology.md','城区太阳能板'), 'scan':('technology.md','普通心脏扫描'), 'paper':('technology.md','档案馆保留地震前'), 'gun':('technology.md','电磁枪发射'), 'maintenance':('technology.md','机械义肢每月')}
def ref(name):
    f,q=anchors[name];matches=[i for i,l in enumerate(source[f],1) if q in l];assert len(matches)==1,name
    i=matches[0];return dict(file=f,start_line=i,end_line=i,quote=source[f][i-1])
def cl(text,verdict,keys=(),reason='',subject=(),context=(),kinds=('attribute','relation','rule','event')):
    return dict(input_quote=text,expected_verdict=verdict,allowed_kinds=list(kinds),evidence=[ref(k) for k in keys],rationale=reason or ('对应证据直接支持。' if verdict=='一致' else '与对应证据在同一对象及条件下不相容。' if verdict=='矛盾' else '现有证据没有建立或否定此项，未提及不等于否定。'),subject_any=list(subject),context_terms=list(context))
cases=initial['cases']
for c in cases:
    c['annotation_status']='author_self_checked_not_independently_reviewed'
    c['required_structures']=[]
    for claim in c['claims']:
        claim['subject_any']=[];claim['context_terms']=[]
    if c['id']=='wb001':c['claims'][0].update(subject_any=['德尔塔'],context_terms=['第三话','之后'])
    if c['id']=='wb007':c['claims'][0].update(input_quote='雷没有在广场',context_terms=['第一话','播放德尔塔救援新闻'])
    if c['id']=='wb012':c['claims'][0]['input_quote']='莉娅是医院的医生'
    if c['id']=='wb004':c['claims'][0]['context_terms']=['网络数据库损坏']
    if c['id']=='wb006':c['claims'][0]['subject_any']=['雷的父亲']
    if c['id']=='wb012':c['claims'][0]['subject_any']=['莉娅']
    if c['id']=='wb014':c['claims'][0]['subject_any']=['雷的父亲']
    if c['id']=='wb018':c['claims'][0]['context_terms']=['赤砂王国']
    if c['id']=='wb019':c['claims'][0]['context_terms']=['第27章']
    if c['group']=='unrelated':
        names={'wb017':['艾琳','海雾学院'],'wb018':['赤砂王国'],'wb019':['泽恩','琥珀岛','风帆公会'],'wb020':['苏州']}[c['id']]
        c['required_structures']=[dict(kind='entity',entity=n) for n in names]
        if c['id']=='wb019':c['required_structures'].append(dict(kind='timepoint',entity='第27章'))
def add(group,dimension,family,claims,structures=()):
    text='。'.join(c['input_quote'] for c in claims)+'。'
    verdict='矛盾' if any(c['expected_verdict']=='矛盾' for c in claims) else '不确定' if any(c['expected_verdict']=='不确定' for c in claims) else '一致'
    cases.append(dict(id='wb'+str(len(cases)+1).zfill(3),group=group,primary_dimension=dimension,family_id=family,text=text,expected_overall=verdict,claims=claims,required_structures=list(structures),annotation_status='author_self_checked_not_independently_reviewed',note='测试设定，不代表正式canon；预期在模型运行前由实现者编写。'))
# 28 supported standalone inputs.
correct=[
('身份','dual_host','雷拥有两颗心脏',['heart']),('人物关系','ab_host','亚巴顿寄宿在雷的左侧心脏',['ab','heart']),('空间','raguel_warm_position','拉古艾尔施展治疗时雷的右胸会变暖',['warm','rag']),('因果','ab_memory_cost','雷借用亚巴顿的力量后会短暂失去部分记忆',['memory']),('时间','angel_schedule','亚巴顿的降临比天使议会计划早了三年',['early']),('世界规则','healing_scope','拉古艾尔可以加速伤口愈合',['heal']),('世界规则','limb_regeneration','拉古艾尔不能再生完整肢体',['heal']),('人物关系','contract_payment','宿主建立天使契约时必须自愿付出重要记忆',['contract']),('世界规则','contract_lifespan','天使契约不会自动延长宿主寿命',['contract']),('时间','angel_departure_duration','寄宿天使离开宿主体内最多三十秒',['leave']),('世界规则','seal_hosts','解除地下神殿封印需要两名宿主同时触碰石门',['seal']),('世界规则','guise_detection','天使的外表伪装不能消除契约精神波动',['guise']),('因果','church_detection','旧教堂静默结界能降低守望者发现宿主的概率',['church']),('人物关系','resonance_partners','雷双心脏共鸣时两名寄宿天使都能听到对方的声音',['resonate']),('物理规则','armor_power','德尔塔的装甲依靠独立电池供电',['armor']),('时间','walking_duration','德尔塔的外骨骼正常行走可运行两小时',['battery']),('物理规则','armor_cooling','德尔塔装甲连续高功率运行十分钟后必须冷却',['cool','heat']),('物理规则','cooling_thrust','德尔塔的装甲冷却期间无法使用推进器',['cool']),('物理规则','pressure_feedback','月城的机械左臂能反馈压力',['touch']),('物理规则','neural_calibration','月城机械左臂的神经接口需要定期校准',['neural']),('物理规则','radio_blocking','金属屏蔽室可以阻挡普通无线电信号',['radio']),('世界规则','mental_radio','金属屏蔽室不能屏蔽天使契约的精神波动',['radio']),('空间','underground_network','旧城区地下隧道没有移动网络覆盖',['tunnel']),('时间','train_schedule','城区列车每天早晨六点开始运行',['train']),('因果','night_storage','城区夜间照明依靠白天充电的储能电池',['solar']),('物理规则','scan_visibility','普通心脏扫描无法显示寄宿天使的形体',['scan']),('身份','archive_backup','档案馆保留地震前的纸质备份',['paper']),('物理规则','gun_ammunition','电磁枪发射的是金属弹丸',['gun'])]
for dim,family,text,refs in correct:add('consistent',dim,family,[cl(text,'一致',refs)])
# Eight supported mixed inputs; both clauses must be checked.
mixed_good=[
('空间','heart_sides',('雷的左侧心脏寄宿亚巴顿',['heart']),('雷的右侧心脏寄宿拉古艾尔',['heart'])),('身份','arm_sides',('月城的左臂是机械臂',['arm']),('月城的右臂是自然肢体',['arm'])),('人物知识','news_vs_meeting',('雷在第一话新闻中见过德尔塔的装甲形象',['news','meet']),('雷与德尔塔在第二话第一次当面交谈',['meet'])),('时间','quake_then_install',('摧毁旧城区的大地震发生于2063年',['quake']),('月城的机械左臂安装于2064年',['surgery'])),('身份','lia_archive',('莉娅是城区档案馆管理员',['lia']),('城区档案馆在2065年开放地震救援档案',['open'])),('世界规则','tracking_vs_shelter',('守望者通过契约精神波动追踪宿主',['watch']),('旧教堂静默结界可以削弱宿主精神波动',['church'])),('物理规则','armor_battery_cool',('德尔塔装甲使用可更换独立电池',['battery']),('德尔塔装甲连续高功率运行十分钟必须停止散热',['heat'])),('身份','blind_navigation',('监察官在爆炸后双眼失明',['blind']),('监察官依靠听觉和手杖定位',['blind']))]
for dim,family,a,b in mixed_good:add('consistent',dim,family,[cl(a[0],'一致',a[1]),cl(b[0],'一致',b[1])])
# Nine additional conflicts per dimension. First two in each include a separate true fact.
bad={
'时间':[
('arm_install_year','月城在2063年已经安装了机械左臂',['surgery']),('quake_year','摧毁旧城区并让雷昏迷的那次大地震发生在2064年',['quake']),('first_face_to_face','雷与德尔塔第一次当面交谈发生在第三话',['meet']),('disclosure_chapter','雷说明双心脏秘密的既有告知事件发生在第二话而非第三话',['disclose']),('sword_acquisition','雷从旧教堂获得长剑的既有事件发生在第八话',['sword']),('sword_breakage','雷的长剑断裂的既有交战事件发生在第四话',['broken']),('father_departure','雷的父亲乘列车离城的既有事件发生在第六话',['depart']),('angel_schedule','亚巴顿的降临比天使议会计划晚了三年',['early']),('angel_departure_duration','寄宿天使离体可以稳定维持三分钟',['leave'])],
'人物知识':[
('public_true_name','现有公众已经知道德尔塔装甲下的真实姓名',['public']),('host_secrecy','除了德尔塔与两位寄宿天使之外所有人也都知道雷的双心脏秘密',['secret']),('delta_identity_disclosure','第三话雷告知宿主身份之后德尔塔仍不知道雷是宿主',['disclose']),('father_host_knowledge','雷的父亲知道儿子的胸痛与拉古艾尔寄宿有关',['father']),('father_host_knowledge','雷的父亲明确认出了寄宿在儿子体内的拉古艾尔',['father']),('news_vs_meeting','雷在第二话当面交谈之前从未见过德尔塔的装甲形象',['meet','news']),('host_secrecy','现有普通公众都已经知道雷是双宿主',['secret']),('delta_identity_disclosure','德尔塔在第三话听完身份说明后仍不知道雷拥有两颗心脏',['disclose']),('public_codename','公众连德尔塔的代号都不知道',['public'])],
'空间':[
('heart_sides','亚巴顿的既定寄宿位置是雷的右侧心脏',['ab','heart']),('heart_sides','拉古艾尔的既定寄宿位置是雷的左侧心脏',['rag','heart']),('research_station_location','北方研究站远离城区列车终点站',['train']),('news_square_presence','第一话雷观看德尔塔救援新闻时身处北方研究站而非广场',['news']),('father_destination','第五话既有离城事件中雷的父亲前往的是南方研究站而非北方研究站',['depart']),('neural_arm_side','月城的机械左臂无需改造接口就能直接装在右臂上',['neural']),('seal_contact_location','两名宿主无需触碰地下神殿石门也能按既定仪式解除封印',['seal']),('heart_sides','雷的两名寄宿天使都住在同一颗右侧心脏中',['heart','ab','rag']),('raguel_warm_position','拉古艾尔治疗引起的既定发热位置是雷的左胸而非右胸',['warm','rag'])],
'人物关系':[
('ab_host','亚巴顿在现有寄宿关系中的宿主是德尔塔',['ab']),('raguel_host','拉古艾尔在现有寄宿关系中的宿主是月城',['rag']),('sword_holder','第四话从旧教堂获得长剑的持有人是德尔塔而不是雷',['sword']),('delta_equipment','德尔塔的既定行动装备是月城的机械左臂而不是动力外骨骼',['armor']),('contract_payment','宿主契约里的重要记忆由天使付出而不是宿主',['contract']),('resonance_partners','雷双心脏共鸣时能听到伙伴声音的寄宿天使只有亚巴顿而不包括拉古艾尔',['resonate','heart']),('tracking_target','守望者追踪的是普通无线电信号而非宿主契约精神波动',['watch']),('delta_equipment','德尔塔并不使用任何动力外骨骼',['armor']),('host_membership','雷体内的两位寄宿天使是亚巴顿和守望者',['heart','rag'])],
'物理规则':[
('solar_night','城区太阳能板在夜间仍能发电',['solar']),('gun_recharging','电磁枪不必充电就能连续无限射击',['gun']),('armor_cooling','德尔塔现有装甲可连续高功率运行二十分钟而无需冷却',['cool','heat']),('pressure_feedback','月城的机械左臂连压力也无法反馈',['touch']),('cooling_thrust','德尔塔的现有装甲冷却期间仍能使用推进器',['cool']),('walking_duration','德尔塔外骨骼的独立电池在正常行走时可以持续供电两天',['battery']),('propulsion_consumption','德尔塔外骨骼的高功率推进完全不会消耗电池电量',['battery']),('radio_blocking','金属屏蔽室无法阻挡任何普通无线电信号',['radio']),('neural_control','月城现有机械左臂的运动指令通过普通无线电而非神经接口传递',['neural'])],
'世界规则':[
('healing_resurrection','拉古艾尔现有的治疗力量可以让已经死亡的人复活',['heal']),('limb_regeneration','拉古艾尔现有的治疗力量可以再生完整肢体',['heal']),('angel_departure_duration','寄宿天使离开宿主后可以无限期维持凝聚形体',['leave']),('seal_hosts','一名宿主独自触碰石门就能解除地下神殿封印',['seal']),('contract_consent','宿主不需要自愿也能按既定规则建立天使契约',['contract']),('contract_lifespan','建立天使契约会自动延长宿主寿命',['contract']),('mental_radio','普通无线电屏蔽可以阻止守望者对契约波动的感知',['watch','radio']),('guise_detection','天使隐藏羽翼和改变外表就能消除契约精神波动',['guise']),('resonance_trigger','雷只要普通心跳加速就必定进入双心脏共鸣',['resonate'])],
'因果':[
('memory_cost_strength','雷借用亚巴顿的力量越强造成的记忆缺失就越轻',['memory']),('ab_memory_cost','雷借用亚巴顿力量后从来不会失去任何记忆',['memory']),('raguel_ab_location','拉古艾尔施展治疗会让亚巴顿转移到右侧心脏',['rag']),('mental_radio','屏蔽普通无线电就会让守望者无法感知宿主精神波动',['watch','radio']),('guise_detection','天使伪装外表会使契约的精神波动消失',['guise']),('night_storage','城区夜间照明依靠太阳能板当夜发电而非白天充电的储能电池',['solar']),('maintenance_effect','忽略机械义肢接口维护会保证连接更稳定并杜绝运动延迟',['maintenance']),('seal_force','单个天使只要加大蛮力就能打开地下神殿石门',['seal']),('church_detection','旧教堂的静默结界会增强精神波动并提高守望者发现宿主的概率',['church'])],
'身份':[
('delta_public_identity','德尔塔的装甲行动者身份从未出现在公共新闻中',['public']),('dual_host','雷在现有生理设定中只拥有一颗心脏',['heart']),('arm_sides','月城在现有状态下两条手臂都是自然肢体',['arm']),('blind_state','监察官在既有爆炸事件之后并未失明',['blind']),('blind_recovery','监察官已经拥有恢复视力的能力',['blind']),('lia_profession','莉娅并不是城区档案馆管理员',['lia']),('heart_sides','拉古艾尔是雷左心脏里的那名寄宿天使',['heart','rag']),('heart_sides','亚巴顿是雷右心脏里的那名寄宿天使',['heart','ab']),('dual_host','雷在现有身份设定中并不是双宿主',['secret','heart'])]}
true_by_dim={'时间':('月城在2064年安装机械左臂',['surgery']),'人物知识':('公众知道德尔塔的代号',['public']),'空间':('月城的右臂始终是自然肢体',['arm']),'人物关系':('雷有两位寄宿天使',['heart']),'物理规则':('电磁枪发射金属弹丸',['gun']),'世界规则':('拉古艾尔可以加速伤口愈合',['heal']),'因果':('旧教堂的静默结界可以削弱精神波动',['church']),'身份':('莉娅负责纸质历史档案整理',['lia'])}
for dim,entries in bad.items():
    assert len(entries)==9
    for i,(family,text,keys) in enumerate(entries):
        claims=[cl(text,'矛盾',keys)]
        if i<2:
            true,refs=true_by_dim[dim];claims.insert(0,cl(true,'一致',refs))
        add('contradiction',dim,family,claims)
# Eight unsupported standalone inputs and eight known+unknown mixed inputs.
unknown=[('人物知识','ray_mind_reading','雷能够直接读懂陌生人的思想',()),('时间','future_arm_upgrade','月城在2067年为机械左臂增加了温度传感器',('surgery','touch')),('人物关系','unknown_engagement','德尔塔已经与莉娅订婚',()),('身份','unknown_father_name','雷的父亲名叫维克多',('father',)),('物理规则','unknown_armor_water','德尔塔的装甲能够变形成潜水艇',('armor',)),('因果','quake_cause_unknown','摧毁旧城区的大地震是发电厂事故造成的',('quake','outage')),('物理规则','unknown_arm_smell','月城的机械左臂能感知气味',('touch',)),('时间','unknown_sword_repair','第九话雷将断裂的长剑送到研究站修复',('broken',))]
for dim,family,text,refs in unknown:add('insufficient',dim,family,[cl(text,'不确定',refs)])
mixed_unknown=[('物理规则','unknown_temperature_power',('雷拥有两颗心脏',['heart']),'雷可以操控周围气温'),('身份','unknown_lia_study',('莉娅是城区档案馆管理员',['lia']),'莉娅曾经在海外留学'),('人物知识','unknown_moon_telepathy',('月城安装了机械左臂',['arm']),'月城能用机械左臂读取他人思想'),('人物知识','unknown_delta_birthday',('德尔塔使用动力外骨骼',['armor']),'德尔塔知道莉娅的生日'),('身份','unknown_father_language',('雷的父亲不知道拉古艾尔寄宿在儿子体内',['father']),'雷的父亲精通古天使语言'),('因果','unknown_watcher_origin',('守望者追踪契约精神波动',['watch']),'守望者由北方研究站制造'),('物理规则','unknown_battery_material',('城区太阳能板夜间不能发电',['solar']),'城区储能电池采用液态金属制造'),('因果','unknown_ab_summoning',('亚巴顿比计划早降临三年',['early']),'亚巴顿提前降临是雷主动召唤造成的')]
for dim,family,known,novel in mixed_unknown:add('insufficient',dim,family,[cl(known[0],'一致',known[1]),cl(novel,'不确定')])
unrelated=[
('卡拉是黄昏群岛的祭司',['卡拉','黄昏群岛']),('蓝雾城所有街道都由巨龙铺设',['蓝雾城','巨龙']),('第42章托林在云海建立了羽帆公会',['托林','云海','羽帆公会']),('伊莎的左眼里住着名为烛鸟的精灵',['伊莎','烛鸟']),('翡翠帝国的国王由飞鱼投票选出',['翡翠帝国','飞鱼']),('玻璃星球上的居民以音乐代替食物',['玻璃星球']),('远帆公司的董事长叫安德鲁',['远帆公司','安德鲁']),('在雾钟世界所有影子都会在正午开口说话',['雾钟世界']),('凯文于2088年在月亮城成为市长',['凯文','月亮城']),('黑沙海盗团只使用木制飞船',['黑沙海盗团']),('露西的宠物能将石头变成糖果',['露西']),('东方航线的飞船通过折叠梦境航行',['东方航线']),('石眠族的孩子出生时就会七种语言',['石眠族']),('钟摆城的居民每天将时间存入银行',['钟摆城']),('小说中的莉娜嫁给了阿尔诺',['莉娜','阿尔诺']),('玛丽在巴黎经营一家面包店',['玛丽','巴黎'])]
for i,(text,names) in enumerate(unrelated):
    structures=[dict(kind='entity',entity=n) for n in names]
    for t in ['第42章','2088年']:
        if t in text:structures.append(dict(kind='timepoint',entity=t))
    add('unrelated','完全无关','unrelated_'+str(i+5).zfill(2),[cl(text,'不确定',reason='原文人物/地点/规则与本地资料没有确立联系，不使用模型外部记忆判断真伪。')],structures)
assert len(cases)==160,len(cases)
assert len({c['text'] for c in cases})==160
assert len({c['id'] for c in cases})==160
from collections import Counter
assert Counter(c['group'] for c in cases)==dict(consistent=40,contradiction=80,insufficient=20,unrelated=20)
assert Counter(c['primary_dimension'] for c in cases if c['group']=='contradiction')=={k:10 for k in bad}
for c in cases:
    for claim in c['claims']:
        assert claim['input_quote'] in c['text'],c['id']
        assert all(t in c['text'] for t in claim.get('context_terms',[])),c['id']
        assert all(any(s in c['text'] for s in claim['subject_any']) for _ in [0]) if claim.get('subject_any') else True
        for e in claim['evidence']:assert e['quote'] in '\n'.join(source[e['file']][e['start_line']-1:e['end_line']])
    for s in c['required_structures']:assert s['entity'] in c['text']
# Explicit tags distinguish inherited draft and familiar semantic families from strict blind testing.
prior=json.loads((ROOT/'evaluation/llm_baseline_v1.json').read_text(encoding='utf-8'))['cases']
prior_families={'heart_sides','ab_host','raguel_host','lia_profession','dual_host','solar_night','healing_resurrection','arm_install_year','unknown_arm_smell','unknown_armor_water','future_arm_upgrade'}
for c in cases:
    c['origin']='approved_draft_20' if int(c['id'][2:])<=20 else 'expanded_before_run'
    c['prior_development_semantic_overlap']=c['family_id'] in prior_families or any(c['text']==r['text'] for r in prior)
    c['evidence_families']=sorted({e['file']+':'+str(e['start_line']) for clm in c['claims'] for e in clm['evidence']})
result=dict(schema_version=1,name='worldcheck-world-benchmark-160-v1',status='frozen_before_model_run',k=5,fixture_hashes=initial['fixture_hashes'],annotation_policy='运行前实现者人工标注并自查，无独立专家审核。沿用开发资料、含同源变体及开发语义，非严格盲测。无关输入不确定。8维度可重叠。',group_targets=dict(consistent=40,contradiction=80,insufficient=20,unrelated=20),cases=cases)
(ROOT/'evaluation/world_benchmark_160_v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
md=['# 世界观benchmark：160条冻结标注','',result['annotation_policy'],'', '分布：40一致、80矛盾（8类各10）、20相关未知、20完全无关；含32条混合输入。测试设定，不代表正式canon。','']
for c in cases:
    md += [f"## {c['id']} — {c['group']} / {c['primary_dimension']} / {c['expected_overall']}",'',c['text'],'',f"family：`{c['family_id']}`；开发语义重叠标记：{c['prior_development_semantic_overlap']}",'']
    for claim in c['claims']:
        md += [f"- **{claim['expected_verdict']}**：{claim['input_quote']}", '  - 理由：'+claim['rationale']]
        if claim.get('subject_any'):md += ['  - 主体允许名称：'+' / '.join(claim['subject_any'])]
        if claim.get('context_terms'):md += ['  - 必须保留的上下文：'+' / '.join(claim['context_terms'])]
        for e in claim['evidence']:md += [f"  - 原文 `{e['file']}:{e['start_line']}`：{e['quote']}"]
        if not claim['evidence']:md += ['  - 无直接证据；不依据外部常识强判。']
    if c['required_structures']:md += ['','新增结构期望：'+json.dumps(c['required_structures'],ensure_ascii=False)]
    md+=['']
(ROOT/'evaluation/world_benchmark_160_v1.md').write_text('\n'.join(md),encoding='utf-8')
print(json.dumps(dict(total=len(cases),claims=sum(len(c['claims']) for c in cases),groups=dict(Counter(c['group'] for c in cases)),conflict_dimensions=dict(Counter(c['primary_dimension'] for c in cases if c['group']=='contradiction')),mixed=sum(len(c['claims'])>1 for c in cases),families=len({c['family_id'] for c in cases})),ensure_ascii=False))
