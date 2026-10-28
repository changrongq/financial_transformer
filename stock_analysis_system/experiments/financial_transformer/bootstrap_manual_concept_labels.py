from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


INDUSTRY_CONCEPT_MAP: dict[str, list[str]] = {
    "银行": ["银行", "高股息"],
    "保险": ["保险", "高股息"],
    "证券": ["券商", "非银金融"],
    "多元金融": ["多元金融", "非银金融"],
    "石油开采": ["油气", "能源"],
    "石油加工": ["炼化", "能源"],
    "化纤": ["炼化", "化工"],
    "煤炭开采": ["煤炭", "高股息"],
    "电信运营": ["运营商", "中字头"],
    "电气设备": ["电力设备", "新能源"],
    "电器仪表": ["工业自动化", "智能制造"],
    "新型电力": ["电力", "公用事业"],
    "水力发电": ["水电", "高股息"],
    "火力发电": ["火电", "高股息"],
    "半导体": ["半导体", "国产替代"],
    "通信设备": ["通信设备", "AI硬件"],
    "元器件": ["消费电子", "电子元件"],
    "IT设备": ["安防", "计算机设备"],
    "软件服务": ["软件", "AI应用"],
    "汽车整车": ["汽车整车", "新能源车"],
    "汽车配件": ["汽车零部件", "新能源车"],
    "白酒": ["白酒", "消费"],
    "食品": ["食品饮料", "消费"],
    "家用电器": ["家电", "消费"],
    "生物制药": ["创新药", "医药"],
    "化学制药": ["化学制药", "医药"],
    "医疗保健": ["医疗器械", "医药"],
    "农业综合": ["养殖", "农业"],
    "农药化肥": ["化肥", "农业"],
    "化工原料": ["化工", "周期"],
    "黄金": ["黄金", "资源品"],
    "铜": ["铜", "有色金属"],
    "铝": ["铝", "有色金属"],
    "小金属": ["稀有金属", "资源品"],
    "普钢": ["钢铁", "周期"],
    "玻璃": ["建材", "周期"],
    "水泥": ["建材", "周期"],
    "工程机械": ["工程机械", "基建"],
    "专用机械": ["高端装备", "智能制造"],
    "运输设备": ["轨交装备", "中字头"],
    "建筑工程": ["基建", "中字头"],
    "船舶": ["造船", "军工"],
    "航空": ["航空装备", "军工"],
    "空运": ["航空运输", "消费复苏"],
    "水运": ["航运", "周期"],
    "港口": ["港口", "高股息"],
    "铁路": ["铁路", "中字头"],
    "仓储物流": ["物流", "消费复苏"],
    "互联网": ["互联网", "AI应用"],
    "旅游服务": ["旅游消费", "消费复苏"],
    "乳制品": ["乳制品", "消费"],
    "软饮料": ["饮料", "消费"],
    "综合类": ["综合", "资产重组"],
}


STOCK_OVERRIDE_MAP: dict[str, list[str]] = {
    "300750": ["锂电", "储能", "新能源车"],
    "002594": ["新能源车", "锂电", "智能驾驶"],
    "601138": ["AI硬件", "服务器", "消费电子"],
    "300308": ["光模块", "CPO", "AI硬件"],
    "300502": ["光模块", "CPO", "AI硬件"],
    "300394": ["光模块", "CPO", "AI硬件"],
    "601869": ["光通信", "光模块", "AI硬件"],
    "688981": ["晶圆代工", "半导体", "国产替代"],
    "688041": ["国产算力", "CPU", "半导体"],
    "688256": ["AI芯片", "国产算力", "半导体"],
    "688795": ["GPU", "国产算力", "半导体"],
    "688802": ["GPU", "国产算力", "半导体"],
    "002371": ["半导体设备", "国产替代", "半导体"],
    "688012": ["半导体设备", "国产替代", "半导体"],
    "603986": ["存储芯片", "半导体", "国产替代"],
    "688347": ["晶圆代工", "半导体", "国产替代"],
    "000333": ["家电", "机器人", "智能制造"],
    "600406": ["电网设备", "特高压", "电力设备"],
    "300274": ["光伏", "储能", "新能源"],
    "603993": ["钼", "资源品", "新能源上游"],
    "601899": ["铜", "黄金", "资源品"],
    "000792": ["盐湖提锂", "钾肥", "资源品"],
    "002475": ["消费电子", "AI硬件", "连接器"],
    "002384": ["消费电子", "PCB", "AI硬件"],
    "300476": ["PCB", "AI硬件", "消费电子"],
    "002415": ["安防", "AI视觉", "计算机设备"],
    "300059": ["互联网金融", "券商", "非银金融"],
    "600150": ["造船", "军工", "海洋装备"],
    "601919": ["航运", "集运", "周期"],
    "601816": ["铁路", "中字头", "公用事业"],
    "300124": ["工业自动化", "机器人", "智能制造"],
    "002352": ["物流", "冷链物流", "消费复苏"],
    "002602": ["游戏", "互联网", "AI应用"],
    "000617": ["AMC", "多元金融", "国企改革"],
    "000988": ["激光设备", "智能制造", "光通信"],
    "002353": ["油服", "页岩油气", "能源"],
    "002837": ["液冷", "数据中心", "AI硬件"],
    "601888": ["免税", "旅游消费", "消费复苏"],
}


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _as_clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _seed_concepts(stock_code: str, industry_name: str) -> tuple[str, str]:
    concepts: list[str] = []
    notes: list[str] = []
    if industry_name in INDUSTRY_CONCEPT_MAP:
        concepts.extend(INDUSTRY_CONCEPT_MAP[industry_name])
        notes.append("industry_rule")
    if stock_code in STOCK_OVERRIDE_MAP:
        concepts = STOCK_OVERRIDE_MAP[stock_code] + concepts
        notes.append("stock_override")
    return "|".join(_dedupe_keep_order(concepts)), "|".join(_dedupe_keep_order(notes))


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap seed manual concept labels from industry and stock override rules")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--overwrite-nonempty", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve() if args.output else input_path

    frame = pd.read_csv(input_path, dtype={"stock_code": str}, encoding="utf-8-sig")
    frame["stock_code"] = frame["stock_code"].astype(str).str.strip().str.zfill(6)
    frame["industry_name"] = frame.get("industry_name", pd.Series([""] * len(frame))).fillna("").astype(str).str.strip()
    if "manual_concept_names" not in frame.columns:
        frame["manual_concept_names"] = ""
    if "manual_concept_ids" not in frame.columns:
        frame["manual_concept_ids"] = ""
    if "notes" not in frame.columns:
        frame["notes"] = ""
    frame["manual_concept_names"] = frame["manual_concept_names"].astype("object")
    frame["manual_concept_ids"] = frame["manual_concept_ids"].astype("object")
    frame["notes"] = frame["notes"].astype("object")

    seeded_count = 0
    for index, row in frame.iterrows():
        existing = _as_clean_text(row.get("manual_concept_names", ""))
        if existing and not args.overwrite_nonempty:
            continue
        seeded_concepts, seeded_notes = _seed_concepts(str(row["stock_code"]), str(row.get("industry_name", "")))
        if not seeded_concepts:
            continue
        frame.at[index, "manual_concept_names"] = seeded_concepts
        if not _as_clean_text(row.get("manual_concept_ids", "")):
            frame.at[index, "manual_concept_ids"] = ""
        notes = _dedupe_keep_order([_as_clean_text(row.get("notes", "")), seeded_notes, "seed_v1"])
        frame.at[index, "notes"] = "|".join(notes)
        seeded_count += 1

    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"seeded {seeded_count} rows -> {output_path}")


if __name__ == "__main__":
    main()