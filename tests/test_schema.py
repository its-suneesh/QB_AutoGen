import json
from app.schemas import LLMToolOutputSchema, _loads_latex_json

BS = chr(92)          # a single backslash, written this way to stay unambiguous
fails = []


def check(label, got, want):
    ok = got == want
    print(("  ok   " if ok else "  FAIL ") + label)
    if not ok:
        print("         got:  %r" % (got,))
        print("         want: %r" % (want,))
        fails.append(label)


print("_loads_latex_json")
check("plain valid JSON untouched", _loads_latex_json('[{"a":1}]'), [{"a": 1}])
check("real \\n escape still honoured", _loads_latex_json('["line1\\nline2"]'), ["line1\nline2"])
check("unicode escape survives", _loads_latex_json('["caf\\u00e9"]'), ["café"])
check("LaTeX times repaired", _loads_latex_json('["1.4142 ' + BS + 'times 7"]'),
      ["1.4142 " + BS + "times 7"])
check("LaTeX alpha+beta repaired", _loads_latex_json('["' + BS + 'alpha + ' + BS + 'beta"]'),
      [BS + "alpha + " + BS + "beta"])
check("frac repaired", _loads_latex_json('["' + BS + 'frac{a}{b}"]'), [BS + "frac{a}{b}"])
check("begin/end repaired",
      _loads_latex_json('["' + BS + 'begin{tabular}' + BS + 'end{tabular}"]'),
      [BS + "begin{tabular}" + BS + "end{tabular}"])
check("already-escaped LaTeX unchanged", _loads_latex_json('["' + BS + BS + 'times"]'),
      [BS + "times"])
check("garbage refused", _loads_latex_json('not json at all'), None)
check("truncated refused", _loads_latex_json('[{"a":'), None)

print("\nfull schema path")
s = LLMToolOutputSchema()
q = {
    "question": "Compute $1.4142 " + BS + "times 7$",
    "answer": "9.8994",
    "question_latex": "Compute $1.4142 " + BS + "times 7$",
    "answer_latex": "$" + BS + "frac{7}{2}$",
}
# what the model actually sends: LaTeX written raw inside the JSON string
raw = json.dumps([q], ensure_ascii=False).replace(BS + BS, BS)
print("  payload has bare LaTeX:", (BS + "times") in raw and (BS + BS + "times") not in raw)

out = s.load({"questions": raw})
check("stringified + bare LaTeX accepted", len(out["questions"]), 1)
check("LaTeX preserved", out["questions"][0]["answer_latex"], "$" + BS + "frac{7}{2}$")
check("proper list still works", len(s.load({"questions": [q]})["questions"]), 1)

try:
    s.load({"questions": "totally broken ["})
    check("garbage still rejected", "accepted", "rejected")
except Exception:
    print("  ok   garbage still rejected")

print("\n%d failed" % len(fails) if fails else "\nall passed")
raise SystemExit(1 if fails else 0)
