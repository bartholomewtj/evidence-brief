"""Hand check: Bart judges the items the reference models split on.

    python tools/jev_eval/run.py handcheck build    # writes evals/jev/handcheck.md + cards.json
    python tools/jev_eval/run.py handcheck apply    # reads evals/jev/hand_answers.txt

Answers file: one `ID: answer` per line (answer = the option word or letter).
`apply` writes hand_relevance.json / hand_design.json / hand_claims.json, which
every `score` command folds into the gold labels.
"""

from __future__ import annotations

import os
import re

from common import OUT, load_draft, load_json, pool_paths, save_json, short, slug

MD = os.path.join(OUT, "handcheck.md")
CARDS = os.path.join(OUT, "cards.json")
ANSWERS = os.path.join(OUT, "hand_answers.txt")

REL_OPTS = ["direct", "related", "tangential"]
CLAIM_OPTS = ["ok", "flag"]


def _drafts():
    return {slug(p): load_draft(p) for p in pool_paths()}


def items() -> list[dict]:
    out = []
    drafts = _drafts()
    r = load_json(os.path.join(OUT, "relevance.json"))
    for s in sorted(r["jev"]):
        a, b = r["ref_opus55"][s][0]["relevance"], r["ref_fable"][s][0]["relevance"]
        for k in sorted(a, key=int):
            if b.get(k) != a[k]:
                p = drafts[s].papers[int(k) - 1]
                out.append({"id": f"R:{short(s)}:{k}", "area": "relevance", "stem": s, "key": k,
                            "topic": drafts[s].topic, "options": REL_OPTS,
                            "models": {"opus55": a[k], "fable": b.get(k)},
                            "title": p.title, "venue": p.venue, "year": p.year,
                            "text": (p.abstract or "")[:1400]})
    d = load_json(os.path.join(OUT, "design.json"))
    import design as D
    for s in sorted(d["regex"]):
        a, b = d["ref_opus55"][s]["labels"], d["ref_sonnet"][s]["labels"]
        for k in sorted(a, key=int):
            if b.get(k) != a[k]:
                p = drafts[s].papers[int(k) - 1]
                out.append({"id": f"D:{short(s)}:{k}", "area": "design", "stem": s, "key": k,
                            "topic": drafts[s].topic, "options": list(D.LABELS),
                            "models": {"opus55": a[k], "sonnet": b.get(k)},
                            "title": p.title, "venue": p.venue, "year": p.year,
                            "types": list(p.publication_types or []),
                            "text": (p.abstract or "")[:1400]})
    c = load_json(os.path.join(OUT, "claims.json"))
    a, b = c["ref_opus55"][0]["verdicts"], c["ref_fable"][0]["verdicts"]
    claims = {x["id"]: x for x in c["claims"]}
    bi = lambda v: "ok" if v == "supported" else "flag"  # noqa: E731
    for k in sorted(a):
        if not k.startswith("final2"):
            continue
        split = bi(a[k]["verdict"]) != bi(b[k]["verdict"])
        both_flag = not split and bi(a[k]["verdict"]) == "flag"
        if split or both_flag:
            cl = claims[k]
            out.append({"id": f"C:{short(cl['stem'])}:{k.rsplit('/', 1)[1]}", "area": "claims",
                        "claim_id": k, "topic": cl["topic"], "options": CLAIM_OPTS,
                        "why": "split" if split else "both flagged",
                        "models": {"opus55": f"{a[k]['verdict']}: {a[k]['problem']}",
                                   "fable": f"{b[k]['verdict']}: {b[k]['problem']}"},
                        "sentence": cl["sentence"],
                        "text": "\n\n".join(f"[{i}] {cl['titles'][str(i)]}\n{cl['material'][str(i)][:2500]}"
                                            for i in cl["cited"])})
    return out


def cmd_build():
    its = items()
    save_json(CARDS, its)
    lines = ["# Hand check — Jev vs LLM evaluation", "",
             f"{len(its)} items where the two reference models disagree (plus claims both flagged).",
             "Write answers in `hand_answers.txt` as `ID: answer`, one per line.", ""]
    for it in its:
        lines += [f"## {it['id']}  ({it['area']})", "", f"**Question:** {it['topic']}", ""]
        if it["area"] == "claims":
            lines += [f"**Sentence:** {it['sentence']}", "", f"**Why here:** {it['why']}", ""]
        else:
            lines += [f"**Paper:** {it['title']} ({it.get('year')}; {it.get('venue')})", ""]
            if it.get("types"):
                lines += [f"**Index types:** {', '.join(it['types'])}", ""]
        lines += [f"**Models said:** " + "; ".join(f"{k}: {v}" for k, v in it["models"].items()), "",
                  f"**Options:** {' / '.join(it['options'])}", "", "```", it["text"], "```", ""]
    with open(MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    if not os.path.exists(ANSWERS):
        with open(ANSWERS, "w", encoding="utf-8") as f:
            f.write("\n".join(f"{it['id']}: " for it in its) + "\n")
    print(f"{len(its)} items -> {MD}")


def cmd_apply():
    its = {it["id"]: it for it in load_json(CARDS)}
    rel, des, cla = {}, {}, {}
    n = 0
    with open(ANSWERS, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\s*([RDC]:[^:]+:[^:\s]+)\s*:\s*(\S*)", line)
            if not m or not m.group(2):
                continue
            it, ans = its.get(m.group(1)), m.group(2).strip().lower()
            if not it:
                continue
            opts = it["options"]
            if len(ans) == 1 and ans.isalpha() and ord(ans) - 97 < len(opts):
                ans = opts[ord(ans) - 97]
            if ans not in opts:
                print(f"skip {m.group(1)}: {ans!r} not in {opts}")
                continue
            n += 1
            if it["area"] == "relevance":
                rel.setdefault(it["stem"], {})[it["key"]] = ans
            elif it["area"] == "design":
                des.setdefault(it["stem"], {})[it["key"]] = ans
            else:
                cla[it["claim_id"]] = ans
    save_json(os.path.join(OUT, "hand_relevance.json"), rel)
    save_json(os.path.join(OUT, "hand_design.json"), des)
    save_json(os.path.join(OUT, "hand_claims.json"), cla)
    print(f"applied {n} answers")


def main(argv):
    {"build": cmd_build, "apply": cmd_apply}[argv[0] if argv else "build"]()
