import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
paths = [
    "app/web_app.py",
    "app/services/data_loaders.py",
    "app/ui/components/banner.py",
    "app/ui/components/layout.py",
    "app/ui/tabs/add_manage.py",
    "app/ui/tabs/signals.py",
    "app/ui/tabs/detail.py",
    "app/ui/tabs/fundamentals.py",
    "app/ui/tabs/valuation.py",
    "engine/screener_core.py",
    "engine/fundamentals_core.py",
]
out = []
for p in paths:
    fp = root / p
    out.append(f"\n\n# ===== {p} =====\n")
    out.append(fp.read_text(encoding="utf-8"))
bundle = (root / "docs" / "bundle_ai.md")
bundle.parent.mkdir(parents=True, exist_ok=True)
bundle.write_text("".join(out), encoding="utf-8")
print(f"Wrote {bundle}")