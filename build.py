#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""template.html に genspec.json と gen.js を差し込み、単体HTML(persona_screening.html)を生成する。
使い方: python build.py"""
import pathlib

d = pathlib.Path(__file__).parent
tpl = (d / "template.html").read_text(encoding="utf-8")
# </script> などでスクリプトが途切れないよう </ をエスケープしてから埋め込む
gs = (d / "genspec.json").read_text(encoding="utf-8").replace("</", "<\\/")
gj = (d / "gen.js").read_text(encoding="utf-8")
out = tpl.replace("__GENSPEC__", gs).replace("__GENJS__", gj)
(d / "persona_screening.html").write_text(out, encoding="utf-8")
print(f"built persona_screening.html ({len(out)} bytes)")
