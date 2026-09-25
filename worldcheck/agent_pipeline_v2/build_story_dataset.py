"""Build the deterministic 24-story draft dataset for Task 2."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "evaluation" / "story_benchmark_pilot_4_v1.json"
OUT = ROOT / "evaluation" / "story_benchmark_24_v2.json"
INDEX = ROOT / "data" / "index"


def lore_ids():
    version = (INDEX / "CURRENT").read_text(encoding="utf-8").strip()
    metadata = json.loads((INDEX / "versions" / version / "metadata.json").read_text(encoding="utf-8"))
    rows = metadata["chunks"]
    result = {}
    for row in rows:
        text = row["text"]
        for key in (
            "两颗心脏", "左侧心脏里", "右侧心脏里", "身份", "动力外骨骼", "十分钟",
            "公众知道", 
            "机械左臂", "2064年", "2063年大地震", "第一次当面", "第三话", "第五话",
            "静默结界", "纸质备份", "无线电", "单个天使", "三十秒", "长剑",
            "不能让已经死亡", "失去部分记忆", "城区列车", "夜间不能发电",
        ):
            if key in text and key not in result:
                result[key] = row["id"]
    return result


def convert_pilot(data):
    from agent_pipeline_v2.story_dataset import upgrade_legacy_story_dataset
    return upgrade_legacy_story_dataset(data)


def make_story(sentences):
    prefix = "故事发生在城区的一间安静房间里。人物先观察周围环境，再按照当时的情况继续行动。"
    suffix = "随后房间里的光线慢慢移动，风吹过窗帘，桌上的纸张仍留在原处。人物确认没有新的动静，便继续等待后续安排。"
    filler = "他把杯子移到桌子中央，整理衣袖，检查门窗，又在椅子旁停了一会儿。"
    story = prefix + "".join(sentences) + filler + suffix
    while len(story) < 205:
        story += "他安静地观察四周。"
    return story


def fact(sentence, subject, predicate, object_, verdict, dimension, evidence, fact_id):
    return {
        "id": fact_id,
        "subject": subject,
        "predicate": predicate,
        "object": object_,
        "normalized_fact": subject + predicate + (object_ or ""),
        "type": "character_attribute" if dimension in {"physical_rule", "identity", "character_relation"} else "world_event",
        "dimension": dimension,
        "sentence": sentence,
        "expected_verdict": verdict,
        "evidence": evidence,
        "reportable": verdict == "矛盾",
    }


def build_case(case_id, title, group, facts, ignored_text="他把杯子移到桌子中央，"):
    sentences = [item["sentence"] for item in facts]
    story = make_story(sentences)
    gold = []
    cursor = 0
    for item in facts:
        start = story.index(item["sentence"], cursor)
        end = start + len(item["sentence"])
        cursor = end
        evidence = item["evidence"]
        gold.append({
            "id": item["id"],
            "subject": item["subject"],
            "predicate": item["predicate"],
            "object": item["object"],
            "normalized_fact": item["normalized_fact"],
            "type": item["type"],
            "dimension": item["dimension"],
            "source_anchors": [{"text": item["sentence"], "start": start, "end": end}],
            "context_anchors": [],
            "expected_verdict": item["expected_verdict"],
            "relevant_lore_ids": evidence,
            "minimum_evidence_sets": [evidence] if item["expected_verdict"] != "不确定" else [],
            "reportable": item["reportable"],
        })
    ignored_start = story.index(ignored_text)
    ignored = [{
        "anchor": {"text": ignored_text, "start": ignored_start, "end": ignored_start + len(ignored_text)},
        "normalized_proposition": "人物移动杯子",
        "reason": "routine",
    }]
    findings = []
    for item in gold:
        if item["expected_verdict"] == "矛盾":
            findings.append({
                "id": "F" + item["id"].lstrip("G"),
                "gold_fact_ids": [item["id"]],
                "expected_type": "conflict",
                "dimension": item["dimension"],
                "minimum_evidence_sets": [item["relevant_lore_ids"]],
                "merge": False,
            })
    return {
        "id": case_id,
        "title": title,
        "group": group,
        "dimensions": sorted({item["dimension"] for item in gold}),
        "story": story,
        "character_count": len(story),
        "gold_facts": gold,
        "ignored_events": ignored,
        "non_events": [],
        "gold_findings": findings,
        "annotation_status": "assistant_authored_draft_pending_human_review",
    }


def main():
    ids = lore_ids()
    pilot = convert_pilot(json.loads(PILOT.read_text(encoding="utf-8-sig")))
    cases = pilot["cases"]
    # Existing pilot is retained verbatim after the in-memory schema migration.
    # Five additional zero-conflict cases: 9 supported and 11 uncertain facts.
    supported = [
        ("雷拥有两颗心脏。", "雷", "拥有", "两颗心脏", "一致", "physical_rule", [ids["两颗心脏"]]),
        ("亚巴顿寄宿在雷的左侧心脏里。", "亚巴顿", "寄宿", "雷的左侧心脏", "一致", "physical_rule", [ids["左侧心脏里"]]),
        ("拉古艾尔寄宿在雷的右侧心脏里。", "拉古艾尔", "寄宿", "雷的右侧心脏", "一致", "physical_rule", [ids["右侧心脏里"]]),
        ("月城在2064年完成机械左臂安装。", "月城", "安装时间", "2064年", "一致", "time", [ids["2064年"]]),
        ("雷与德尔塔在第二话第一次当面交谈。", "雷", "首次见面", "德尔塔第二话", "一致", "character_relation", [ids["第一次当面"]]),
        ("雷在第三话向德尔塔说明双心脏秘密。", "雷", "告知身份", "德尔塔知道双宿主身份", "一致", "character_knowledge", [ids["第三话"]]),
        ("雷的父亲在第五话乘列车离开城区。", "雷的父亲", "离开", "第五话北方研究站", "一致", "time", [ids["第五话"]]),
        ("雷与德尔塔在第六话躲进旧教堂。", "雷", "躲进", "旧教堂", "一致", "space", [ids["静默结界"]]),
        ("档案馆在2065年开放地震救援档案。", "档案馆", "开放", "2065年救援档案", "一致", "time", [ids["纸质备份"]]),
    ]
    uncertain = [
        ("雷拥有读取他人梦境的能力。", "雷", "拥有", "读取梦境的能力", "不确定", "world_rule", []),
        ("月城可以让机械左臂变成液体。", "月城", "拥有", "液态机械臂能力", "不确定", "physical_rule", []),
        ("德尔塔能够让装甲永久不需要电池。", "德尔塔", "拥有", "无限能源装甲", "不确定", "physical_rule", []),
        ("莉娅知道亚巴顿的完整计划。", "莉娅", "知道", "亚巴顿计划", "不确定", "character_knowledge", []),
        ("雷在地下隧道遇见了守望者。", "雷", "遇见", "地下隧道守望者", "不确定", "space", []),
        ("旧教堂封印可以由一个人独自打开。", "单个宿主", "打开", "旧教堂封印", "不确定", "world_rule", []),
        ("亚巴顿的力量让雷获得永久不死之身。", "亚巴顿", "赋予", "永久不死", "不确定", "causality", []),
        ("雷和德尔塔在第一话已经正式见面。", "雷", "见面", "第一话德尔塔", "不确定", "character_relation", []),
        ("月城在地震前已经使用机械左臂。", "月城", "使用", "地震前机械左臂", "不确定", "time", []),
        ("监察官恢复了爆炸前的视力。", "监察官", "恢复", "视力", "不确定", "physical_rule", []),
        ("守望者无法感知静默结界中的宿主。", "守望者", "无法感知", "静默结界宿主", "不确定", "world_rule", []),
    ]
    for i in range(5):
        items = []
        for j in range(2):
            item = supported[(i * 2 + j) % len(supported)] if i < 2 and j == 0 else uncertain[(i * 2 + j) % len(uncertain)]
            items.append(fact(*item, fact_id=f"G{j+1}"))
        if i == 0:
            items.append(fact(*supported[2], fact_id="G3"))
        elif i in (1, 2, 3):
            items.append(fact(*supported[i + 2], fact_id="G3"))
        cases.append(build_case(f"SL-{5+i:03d}", f"新增无冲突样例{i+1}", "zero_conflict", items))

    # Seven one-conflict cases: one supported, one contradiction, one uncertain, plus selected variants.
    contradictions = [
        ("公众知道德尔塔的真实姓名。", "公众", "知道", "德尔塔的真实姓名", "矛盾", "identity", [ids["公众知道"]]),
        ("月城在2063年已经安装机械左臂。", "月城", "安装时间", "2063年", "矛盾", "time", [ids["2064年"]]),
        ("雷的父亲知道拉古艾尔寄宿在雷体内。", "雷的父亲", "知道", "拉古艾尔寄宿", "矛盾", "character_knowledge", [ids["身份"]]),
        ("月城的右臂是机械臂。", "月城", "拥有", "机械右臂", "矛盾", "space", [ids["机械左臂"]]),
        ("雷能让已经死亡的人复活。", "雷", "拥有", "复活能力", "矛盾", "world_rule", [ids["不能让已经死亡"]]),
        ("雷借用亚巴顿力量后记忆不会受影响。", "雷", "借用后", "记忆不变", "矛盾", "causality", [ids["失去部分记忆"]]),
        ("雷和德尔塔在第一话已经当面交谈。", "雷", "当面交谈", "第一话", "矛盾", "character_relation", [ids["第一次当面"]]),
    ]
    for i, contradiction in enumerate(contradictions):
        facts = [fact(*supported[(i + 3) % len(supported)], fact_id="G1"), fact(*contradiction, fact_id="G2"), fact(*uncertain[(i + 3) % len(uncertain)], fact_id="G3")]
        cases.append(build_case(f"SL-{10+i:03d}", f"单冲突样例{i+1}", "one_conflict", facts))

    # Eight multi-conflict cases, two distinct contradictions each, plus one support and one uncertainty.
    multi_pairs = [
        (contradictions[0], contradictions[1]), (contradictions[2], contradictions[3]),
        (contradictions[4], contradictions[5]), (contradictions[6], contradictions[0]),
        (contradictions[1], contradictions[2]), (contradictions[3], contradictions[4]),
        (contradictions[5], contradictions[6]), (contradictions[0], contradictions[4]),
    ]
    for i, pair in enumerate(multi_pairs):
        facts = [
            fact(*pair[0], fact_id="G1"),
            fact(*pair[1], fact_id="G2"),
            fact(*supported[(i + 5) % len(supported)], fact_id="G3"),
            fact(*uncertain[(i + 5) % len(uncertain)], fact_id="G4"),
        ]
        cases.append(build_case(f"SL-{17+i:03d}", f"多冲突样例{i+1}", "multi_conflict", facts))

    pilot["version"] = "v2-draft-24"
    pilot["schema_version"] = "agent-pipeline-v2-story-dataset-v2"
    pilot["name"] = "Story-Level Consistency Benchmark"
    pilot["target_distribution"] = {"total": 24, "zero_conflict": 8, "one_conflict": 8, "multi_conflict": 8}
    pilot["notice"] = "测试故事和测试设定，不代表正式canon；24篇均为助手草案，尚未完成用户语义审核。"
    pilot["cases"] = cases
    OUT.write_text(json.dumps(pilot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} cases={len(cases)}")


if __name__ == "__main__":
    main()
