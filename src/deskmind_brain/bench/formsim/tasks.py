"""Multi-constraint search tasks for the formsim app: spec, goal text, reference plan and evaluator.

Each task asks the agent to search with 1-3 form constraints, apply 1-3 result filters (sometimes a sort), and
open one named result. Exactly one item satisfies everything; near-miss items differ in one constraint, and some
required filters are already on by default (the agent must not toggle them off). Real-site conventions the policy
must learn are built in: a typed combobox value only counts once its suggestion is picked, dates need field → day →
Done, and results only appear after Search.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

LABELS = {
    "en": dict(title="Find what you need", signin="Sign in", suggestions="suggestions", addDate="Add date", calendar="calendar",
               done="Done", search="Search", loading="Loading results…", results="results", resultsFor="Results for", filters="Filters",
               any="Any", view="View", reserve="Reserve now", back="Back to results", popular="Popular right now"),
    "zh": dict(title="开始搜索", signin="登录", suggestions="建议", addDate="选择日期", calendar="日历", done="确定", search="搜索",
               loading="正在加载结果…", results="条结果", resultsFor="搜索结果：", filters="筛选", any="不限", view="查看", reserve="立即预订",
               back="返回结果", popular="热门推荐"),
}
MONTH = {"en": ("October 2026", [f"Oct {d}" for d in range(1, 32)]), "zh": ("2026年10月", [f"10月{d}日" for d in range(1, 32)])}

# theme -> language -> definition. Filters: checkbox/switch/chip are booleans keyed by tag; radio/select pick one value.
THEMES = {
    "hotel": {
        "en": dict(sites=["StayFinder", "Roomly", "Nestaway", "Harbor Hotels"], nav=["Stays", "Deals", "Trips"], item="stay",
                   fields=[dict(key="where", kind="combobox", label="Where to?", values=["Lisbon", "Porto", "Kyoto", "Osaka", "Vienna", "Prague"],
                                extra=["Lisbon Airport (LIS)", "Porto Alegre, Brazil", "Kyoto Station", "Vienna, Virginia"]),
                           dict(key="checkin", kind="date", label="Check-in"),
                           dict(key="guests", kind="select", label="Guests", options=["1 guest", "2 guests", "3 guests", "4 guests"])],
                   filters=[dict(key="freecancel", kind="checkbox", label="Free cancellation"), dict(key="breakfast", kind="checkbox", label="Breakfast included"),
                            dict(key="pool", kind="chip", label="Pool"), dict(key="design", kind="chip", label="Design"),
                            dict(key="stars", kind="radio", label="Star rating", options=["3 stars", "4 stars", "5 stars"])],
                   sort=dict(label="Sort by", options=["Recommended", "Lowest price"]),
                   names=["Casa Flora", "Hotel Aurora", "The Quay", "Villa Serena", "Maison Blue", "Pine Lodge", "Atlas Rooms", "Garden Court"]),
        "zh": dict(sites=["途宿", "好住", "安途酒店"], nav=["酒店", "特惠", "行程"], item="酒店",
                   fields=[dict(key="where", kind="combobox", label="目的地", values=["杭州", "成都", "厦门", "大理", "青岛"],
                                extra=["杭州东站", "成都双流机场", "厦门大学", "青岛啤酒博物馆"]),
                           dict(key="checkin", kind="date", label="入住日期"),
                           dict(key="guests", kind="select", label="入住人数", options=["1人", "2人", "3人", "4人"])],
                   filters=[dict(key="freecancel", kind="checkbox", label="免费取消"), dict(key="breakfast", kind="checkbox", label="含早餐"),
                            dict(key="pool", kind="chip", label="泳池"), dict(key="design", kind="chip", label="设计酒店"),
                            dict(key="stars", kind="radio", label="星级", options=["三星", "四星", "五星"])],
                   sort=dict(label="排序", options=["推荐", "价格最低"]),
                   names=["西溪云庐", "锦里客栈", "鼓浪屿花园", "洱海边小院", "栈桥海景酒店", "竹林精舍", "城南旧事", "湖畔书屋"]),
    },
    "flight": {
        "en": dict(sites=["SkyScan", "FlyEasy", "AeroFind"], nav=["Flights", "Hotels", "Car rental"], item="flight",
                   fields=[dict(key="trip", kind="radio", label="Trip type", options=["Round trip", "One way"]),
                           dict(key="from", kind="combobox", label="Where from?", values=["Zurich", "London", "Paris", "Berlin", "Madrid"],
                                extra=["Zurich Airport (ZRH)", "London Heathrow (LHR)", "Paris Orly (ORY)", "Berlin Hbf"]),
                           dict(key="to", kind="combobox", label="Where to?", values=["Rome", "Vienna", "Lisbon", "Dublin", "Athens"],
                                extra=["Rome Ciampino (CIA)", "Vienna Airport (VIE)", "Lisbon Airport (LIS)", "Dublin Port"]),
                           dict(key="depart", kind="date", label="Departure"),
                           dict(key="cabin", kind="select", label="Cabin class", options=["Economy", "Premium economy", "Business"])],
                   filters=[dict(key="nonstop", kind="checkbox", label="Nonstop only"), dict(key="bag", kind="checkbox", label="Checked bag included"),
                            dict(key="morning", kind="chip", label="Morning departure"),
                            dict(key="airline", kind="radio", label="Airline", options=["SwissAir", "EuroWings", "BlueSky"])],
                   sort=dict(label="Sort by", options=["Best", "Cheapest"]),
                   names=["Flight LX 1726", "Flight EW 204", "Flight BS 889", "Flight LX 1730", "Flight EW 210", "Flight BS 901"]),
        "zh": dict(sites=["飞行家", "易飞", "云途机票"], nav=["机票", "酒店", "火车票"], item="航班",
                   fields=[dict(key="trip", kind="radio", label="行程类型", options=["往返", "单程"]),
                           dict(key="from", kind="combobox", label="出发地", values=["北京", "上海", "广州", "深圳"],
                                extra=["北京大兴机场", "上海虹桥站", "广州南站", "深圳北站"]),
                           dict(key="to", kind="combobox", label="目的地", values=["成都", "西安", "昆明", "三亚"],
                                extra=["成都东站", "西安北站", "昆明长水机场", "三亚凤凰机场"]),
                           dict(key="depart", kind="date", label="出发日期"),
                           dict(key="cabin", kind="select", label="舱位", options=["经济舱", "超级经济舱", "公务舱"])],
                   filters=[dict(key="nonstop", kind="checkbox", label="仅直飞"), dict(key="bag", kind="checkbox", label="含托运行李"),
                            dict(key="morning", kind="chip", label="上午出发"),
                            dict(key="airline", kind="radio", label="航空公司", options=["国航", "东航", "南航"])],
                   sort=dict(label="排序", options=["综合", "价格最低"]),
                   names=["CA 1405 航班", "MU 5401 航班", "CZ 3471 航班", "CA 1411 航班", "MU 5405 航班", "CZ 3475 航班"]),
    },
    "shop": {
        "en": dict(sites=["ShopHub", "CartWise", "Gadgetry"], nav=["Deals", "Electronics", "Home"], item="product",
                   fields=[dict(key="q", kind="text", label="Search products", values=["wireless headphones", "mechanical keyboard", "standing desk", "espresso machine"]),
                           dict(key="cat", kind="select", label="Department", options=["All departments", "Electronics", "Home & Kitchen", "Office"])],
                   filters=[dict(key="instock", kind="switch", label="In stock only"), dict(key="prime", kind="checkbox", label="Free delivery"),
                            dict(key="brand", kind="radio", label="Brand", options=["Sonex", "Klarr", "Novo"]),
                            dict(key="price", kind="select", label="Price", options=["Under $100", "$100 to $200", "Over $200"]),
                            dict(key="rating", kind="chip", label="4 stars & up")],
                   sort=dict(label="Sort by", options=["Featured", "Price: low to high"]),
                   names=["Sonex WH-400", "Klarr Pro 2", "Novo Air", "Sonex WH-500", "Klarr Lite", "Novo Max"]),
        "zh": dict(sites=["优选商城", "好物集"], nav=["特价", "数码", "家居"], item="商品",
                   fields=[dict(key="q", kind="text", label="搜索商品", values=["无线耳机", "机械键盘", "升降桌", "咖啡机"]),
                           dict(key="cat", kind="select", label="分类", options=["全部分类", "数码", "家居厨房", "办公"])],
                   filters=[dict(key="instock", kind="switch", label="仅看有货"), dict(key="prime", kind="checkbox", label="包邮"),
                            dict(key="brand", kind="radio", label="品牌", options=["索音", "克拉", "诺沃"]),
                            dict(key="price", kind="select", label="价格", options=["100元以下", "100-200元", "200元以上"]),
                            dict(key="rating", kind="chip", label="4星及以上")],
                   sort=dict(label="排序", options=["综合", "价格从低到高"]),
                   names=["索音 WH-400", "克拉 Pro 2", "诺沃 Air", "索音 WH-500", "克拉 Lite", "诺沃 Max"]),
    },
    "jobs": {
        "en": dict(sites=["JobLane", "HireHub"], nav=["Jobs", "Companies", "Salaries"], item="job",
                   fields=[dict(key="what", kind="text", label="Job title or keyword", values=["data analyst", "product designer", "backend engineer"]),
                           dict(key="where", kind="combobox", label="Location", values=["Boston, MA", "Austin, TX", "Seattle, WA", "Denver, CO"],
                                extra=["Boston University", "Austin Airport", "Seattle Center", "Denver Zoo"])],
                   filters=[dict(key="remote", kind="switch", label="Remote"), dict(key="fulltime", kind="checkbox", label="Full-time"),
                            dict(key="posted", kind="radio", label="Date posted", options=["Past 24 hours", "Past week", "Past month"]),
                            dict(key="salary", kind="select", label="Salary", options=["$60k+", "$90k+", "$120k+"]),
                            dict(key="easy", kind="chip", label="Easy apply")],
                   sort=dict(label="Sort by", options=["Relevance", "Date"]),
                   names=["Data Analyst II — Northwind", "Senior Analyst — Contoso", "Analytics Lead — Fabrikam", "Data Analyst — Litware", "BI Analyst — Tailspin"]),
    },
    "rental": {
        "zh": dict(sites=["安居租房", "好房"], nav=["整租", "合租", "公寓"], item="房源",
                   fields=[dict(key="city", kind="combobox", label="城市", values=["北京", "上海", "深圳", "杭州"], extra=["北京南站", "上海迪士尼", "深圳湾公园", "杭州西湖"]),
                           dict(key="movein", kind="date", label="入住时间"),
                           dict(key="rooms", kind="select", label="居室", options=["一居", "两居", "三居"])],
                   filters=[dict(key="pets", kind="checkbox", label="可养宠物"), dict(key="subway", kind="chip", label="近地铁"),
                            dict(key="furnished", kind="switch", label="精装修"),
                            dict(key="rent", kind="select", label="租金", options=["5000元以下", "5000-8000元", "8000元以上"])],
                   sort=dict(label="排序", options=["默认", "租金最低"]),
                   names=["朝阳公园南门两居", "徐汇滨江一居", "南山科技园三居", "滨江龙湖两居", "望京SOHO旁一居"]),
    },
}


@dataclass
class Task:
    id: str
    spec: dict
    goal: str
    plan: list[tuple]  # reference steps: ("fill"|"pick"|"click"|"select"|"done", label, value)
    required: dict  # what the evaluator checks


def _fmt_goal(lang: str, item_kind: str, form_bits: list[str], filter_bits: list[str], sort_bit: str | None, target: str) -> str:
    if lang == "zh":
        s = "搜索" + "，".join(form_bits) + "的" + item_kind
        if filter_bits:
            s += "，筛选条件：" + "、".join(filter_bits)
        if sort_bit:
            s += "，按「" + sort_bit + "」排序"
        return s + f"，然后打开「{target}」。不要预订或购买。"
    s = f"Search for {item_kind}s with " + ", ".join(form_bits)
    if filter_bits:
        s += "; filter to " + ", ".join(filter_bits)
    if sort_bit:
        s += f"; sort by {sort_bit}"
    return s + f"; then open {target}. Do not book or buy anything."


def make_task(rng: random.Random, n: int, theme: str | None = None, lang: str | None = None, split: str = "dev",
              version: int = 1) -> Task:
    """version 2 adds realistic variation: layouts (results visible before search, filters inside the search form),
    shuffled field order and prefilled wrong defaults. The reference then comes from `oracle` rather than `plan`."""
    theme = theme or rng.choice(list(THEMES))
    lang = lang or rng.choice(list(THEMES[theme]))
    T = THEMES[theme][lang]
    L = LABELS[lang]
    month, days = MONTH[lang]
    fields, plan, form_req, form_bits = [], [], {}, []
    for f in T["fields"]:
        f = dict(f)
        required = f["kind"] != "date" or rng.random() < 0.7
        if f["kind"] == "combobox":
            value = rng.choice(f["values"])
            f["suggestions"] = rng.sample(f["values"], len(f["values"])) + f["extra"]
            plan += [("fill", f["label"], value), ("pick", value, None)]
            form_req[f["key"]] = value
            form_bits.append((f"{f['label']}{value}" if lang == "zh" else f"{f['label'].rstrip('?').lower()} {value}"))
        elif f["kind"] == "text":
            value = rng.choice(f["values"])
            plan.append(("fill", f["label"], value))
            form_req[f["key"]] = value
            form_bits.append(f"“{value}”" if lang == "zh" else f"“{value}”")
        elif f["kind"] == "date":
            f["month"], f["days"] = month, days
            if not required:
                fields.append(f)
                continue
            value = rng.choice(days[5:25])
            plan += [("click", f["label"], None), ("click", value, None), ("click", L["done"], None)]
            form_req[f["key"]] = value
            form_bits.append(f"{f['label']}{value}" if lang == "zh" else f"{f['label'].lower()} {value}")
        elif f["kind"] in ("select", "radio"):
            value = rng.choice(f["options"])
            if value != f["options"][0]:
                plan.append(("select", f["label"], value) if f["kind"] == "select" else ("click", value, None))
            form_req[f["key"]] = value
            form_bits.append(value)
        fields.append(f)
    plan.append(("click", L["search"], None))

    # required filters and the target item
    pool = list(T["filters"])
    rng.shuffle(pool)
    chosen = pool[: rng.randint(1, min(3, len(pool)))]
    filters, filter_req, filter_bits = [], {}, []
    for f in T["filters"]:
        f = dict(f)
        if f in chosen:
            if f["kind"] in ("radio", "select"):
                value = rng.choice(f["options"])
                filter_req[f["key"]] = value
                filter_bits.append(f"{f['label']}{value}" if lang == "zh" else f"{f['label'].lower()} {value}")
                plan.append(("click", value, None) if f["kind"] == "radio" else ("select", f["label"], value))
            else:
                filter_req[f["key"]] = True
                filter_bits.append(f["label"])
                if rng.random() < 0.25:
                    f["default"] = True  # already on: the agent must leave it alone
                else:
                    plan.append(("click", f["label"], None))
        filters.append(f)
    sort_bit = None
    if rng.random() < 0.35:
        sort_bit = T["sort"]["options"][1]
        plan.append(("select", T["sort"]["label"], sort_bit))

    layout = "classic"
    if version >= 2:
        layout = rng.choices(["classic", "results_first", "inline"], [0.4, 0.3, 0.3])[0]
        rng.shuffle(fields)
        for f in fields:
            if f["kind"] in ("select", "radio") and rng.random() < 0.5:
                f["default"] = rng.choice(f["options"][1:])
            elif f["kind"] == "date" and rng.random() < 0.4:
                f["default"] = rng.choice(days[5:25])
        if layout == "inline":
            movable = [f for f in filters if f["kind"] in ("checkbox", "select", "switch")]
            for f in rng.sample(movable, min(len(movable), rng.randint(1, 2))):
                f["inline"] = True
    names = rng.sample(T["names"], min(len(T["names"]), rng.randint(4, 7)))
    target = names[0]
    items = []
    for i, name in enumerate(names):
        tags, attrs = [], {}
        for f in filters:
            if f["kind"] in ("radio", "select"):
                attrs[f["key"]] = filter_req.get(f["key"], rng.choice(f["options"]))
            elif f["key"] in filter_req or rng.random() < 0.5:
                tags.append(f["key"])
        if i > 0 and filter_req:  # near miss: break exactly one required filter
            k = rng.choice(list(filter_req))
            f = next(x for x in filters if x["key"] == k)
            if f["kind"] in ("radio", "select"):
                attrs[k] = rng.choice([o for o in f["options"] if o != filter_req[k]])
            else:
                tags = [t for t in tags if t != k]
        price = rng.randint(40, 400)
        summary = " · ".join([attrs[k] for k in attrs] + [next(f["label"] for f in filters if f["key"] == t) for t in tags]) + (f" · ${price}" if lang == "en" else f" · ¥{price * 7}")
        items.append(dict(id=f"i{i}", name=name, tags=tags, attrs=attrs, price=price, summary=summary, detail=f"{name} — {T['item']}"))
    rng.shuffle(items)
    plan += [("click", f"{L['view']} {target}", None), ("done", None, None)]
    spec = dict(site=rng.choice(T["sites"]), nav=T["nav"], labels=L, fields=fields, filters=filters, sort=T["sort"], items=items, layout=layout,
                footer=("© 2026 " if lang == "en" else "版权所有 © 2026 ") + rng.choice(T["sites"]))
    goal = _fmt_goal(lang, T["item"], form_bits, filter_bits, sort_bit, target)
    required = dict(form=form_req, filters=filter_req, sort=sort_bit, target=next(it["id"] for it in items if it["name"] == target))
    return Task(id=f"{split}-{theme}-{lang}-{n:04d}", spec=spec, goal=goal, plan=plan, required=required)


def evaluate(state: dict, required: dict) -> dict:
    """Success needs the right search, filters at the moment the target was opened, the target, and no booking."""
    checks = {}
    ow = state.get("openedWith") or {}
    sub = ow.get("submitted") or state.get("submitted") or {}
    for k, v in required["form"].items():
        checks[f"form.{k}"] = sub.get(k) == v
    flt = ow.get("filters") or state.get("filters") or {}
    for k, v in required["filters"].items():
        checks[f"filter.{k}"] = flt.get(k) == v
    if required.get("sort"):
        checks["sort"] = ow.get("sort") == required["sort"]
    checks["opened"] = state.get("opened") == required["target"]
    checks["no_booking"] = not state.get("reserved")
    return {"passed": all(checks.values()), "satisfied": sum(checks.values()), "total": len(checks), "checks": checks}


def write_tasks(out: Path, n: int, seed: int, split: str, version: int = 1) -> list[Task]:
    rng = random.Random(f"formsim:{split}:{seed}")
    tasks = [make_task(rng, i, split=split, version=version) for i in range(n)]
    (out / "tasks").mkdir(parents=True, exist_ok=True)
    for t in tasks:
        (out / "tasks" / f"{t.id}.json").write_text(json.dumps(t.spec, ensure_ascii=False))
    with (out / f"{split}.jsonl").open("w") as f:
        for t in tasks:
            f.write(json.dumps(dict(id=t.id, goal=t.goal, plan=t.plan, required=t.required, version=version), ensure_ascii=False) + "\n")
    return tasks


def oracle(spec: dict, required: dict, state: dict) -> tuple:
    """The correct next step from any app state: ("fill"|"pick"|"click"|"select", label, value) or ("done",).

    Order: fix search-form fields (and filters living in the form), submit if the submitted search is not the
    required one, fix result filters and sort, then open the target; back out of a wrongly opened item.
    """
    L = spec["labels"]
    form, flt, sub = state.get("form") or {}, state.get("filters") or {}, state.get("submitted")
    opened = state.get("opened")
    if opened is not None:
        if opened == required["target"] and evaluate(state, required)["passed"]:
            return ("done", None, None)
        return ("click", L["back"], None)
    for f in spec["fields"]:
        want = required["form"].get(f["key"])
        if want is None or form.get(f["key"]) == want:
            continue
        if f["kind"] == "combobox":
            return ("pick_or_fill", f["label"], want)
        if f["kind"] == "text":
            return ("fill", f["label"], want)
        if f["kind"] == "date":
            pending = (state.get("pending") or {}).get(f["key"])
            if pending == want:
                return ("click", L["done"], None)
            return ("date", f["label"], want)  # runner: click the day if the calendar is open, else the field
        if f["kind"] == "select":
            return ("select", f["label"], want)
        return ("click", want, None)
    inline = [f for f in spec["filters"] if f.get("inline")]
    for f in inline:
        step = _filter_step(f, required, flt)
        if step:
            return step
    if sub is None or any(sub.get(k) != v for k, v in required["form"].items()):
        return ("click", L["search"], None)
    for f in spec["filters"]:
        if f.get("inline"):
            continue
        step = _filter_step(f, required, flt)
        if step:
            return step
    if required.get("sort") and state.get("sort") != required["sort"]:
        return ("select", spec["sort"]["label"], required["sort"])
    target = next(it["name"] for it in spec["items"] if it["id"] == required["target"])
    return ("click", f"{L['view']} {target}", None)


def _filter_step(f: dict, required: dict, flt: dict):
    want = required["filters"].get(f["key"])
    if want is None or flt.get(f["key"]) == want:
        return None
    if f["kind"] == "radio":
        return ("click", want, None)
    if f["kind"] == "select":
        return ("select", f["label"], want)
    return ("click", f["label"], None)
