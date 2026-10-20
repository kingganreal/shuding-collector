"""2단계: 라벨 검수 웹페이지 (브라우저에서 고치고 저장).

    python -m shuding.review        # http://localhost:8765 열기

VLM 라벨을 사람이 확인/수정하고 '검수 완료'를 체크한다.
train --only-checked 로 검수된 라벨만 학습에 쓸 수 있다.
"""
from __future__ import annotations

import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .common import ACTIONS, LABELS, PLACES, load_image, read_labels, write_labels

PORT = 8765

PAGE = """<!doctype html><html lang="ko"><meta charset="utf-8"><title>슈딩이 라벨 검수</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#faf7f2;color:#222}
header{position:sticky;top:0;background:#ff9800;color:#fff;padding:10px 16px;display:flex;gap:12px;align-items:center;z-index:1}
header button{margin-left:auto;font-size:15px;padding:6px 16px;border:0;border-radius:6px;cursor:pointer}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px;padding:12px}
.card{background:#fff;border-radius:8px;padding:10px;box-shadow:0 1px 3px #0002}
.card.ok{outline:3px solid #4caf50}
.card img{width:100%;height:220px;object-fit:cover;border-radius:6px}
label{display:block;font-size:12px;color:#666;margin-top:6px}
input,select,textarea{width:100%;box-sizing:border-box;font-size:14px;padding:4px}
textarea{height:48px}
.chk{display:flex;gap:6px;align-items:center;margin-top:8px;font-weight:600}
.chk input{width:auto}
</style>
<header><b>슈딩이 라벨 검수</b><span id="stat"></span><button onclick="save()">저장</button></header>
<main id="m"></main>
<script>
const ACTIONS=__ACTIONS__, PLACES=__PLACES__;
let rows=[];
const opt=(list,v)=>(list.includes(v)?list:[v,...list]).map(x=>`<option ${x===v?'selected':''}>${x}</option>`).join('');
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function render(){
  document.getElementById('m').innerHTML=rows.map((r,i)=>`
  <div class="card ${r.checked?'ok':''}" id="c${i}">
    <img loading="lazy" src="/img?i=${i}">
    <label>행동</label><select data-i="${i}" data-k="행동">${opt(ACTIONS,r['행동'])}</select>
    <label>장소</label><select data-i="${i}" data-k="장소">${opt(PLACES,r['장소'])}</select>
    <label>소품 (쉼표로 구분, 실제로 보이는 것만)</label><input data-i="${i}" data-k="소품" data-list value="${esc((r['소품']||[]).join(', '))}">
    <label>표정</label><input data-i="${i}" data-k="표정" value="${esc(r['표정'])}">
    <label>설명</label><textarea data-i="${i}" data-k="설명">${esc(r['설명'])}</textarea>
    <label>검색어 (쉼표로 구분)</label><textarea data-i="${i}" data-k="검색어" data-list>${esc((r['검색어']||[]).join(', '))}</textarea>
    <div class="chk"><input type="checkbox" data-i="${i}" data-k="checked" ${r.checked?'checked':''}> 검수 완료</div>
  </div>`).join('');
  stat();
}
function stat(){document.getElementById('stat').textContent=`${rows.filter(r=>r.checked).length} / ${rows.length} 검수됨`}
document.addEventListener('input',e=>{
  const el=e.target, i=el.dataset.i, k=el.dataset.k; if(i===undefined)return;
  if(k==='checked'){rows[i].checked=el.checked;document.getElementById('c'+i).classList.toggle('ok',el.checked);stat();}
  else if(el.dataset.list!==undefined) rows[i][k]=el.value.split(',').map(s=>s.trim()).filter(Boolean);
  else rows[i][k]=el.value;
});
async function save(){
  const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(rows)});
  alert(r.ok?'저장했습니다':'저장 실패');
}
fetch('/labels').then(r=>r.json()).then(d=>{rows=d;render();});
</script></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path == "/":
            html = PAGE.replace("__ACTIONS__", json.dumps(ACTIONS, ensure_ascii=False)) \
                       .replace("__PLACES__", json.dumps(PLACES, ensure_ascii=False))
            self._send(html.encode(), "text/html; charset=utf-8")
        elif url.path == "/labels":
            self._send(json.dumps(read_labels(), ensure_ascii=False).encode(), "application/json")
        elif url.path == "/img":
            try:
                i = int(parse_qs(url.query)["i"][0])
                buf = io.BytesIO()
                load_image(read_labels()[i]["path"], max_side=480).save(buf, "JPEG", quality=80)
                self._send(buf.getvalue(), "image/jpeg")
            except Exception:
                self._send(b"", "text/plain", 404)
        else:
            self._send(b"", "text/plain", 404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/save":
            return self._send(b"", "text/plain", 404)
        rows = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        known = {r["path"] for r in read_labels()}
        write_labels([r for r in rows if r.get("path") in known])  # 경로는 바꿀 수 없게
        self._send(b"ok", "text/plain")

    def log_message(self, *a) -> None:  # 콘솔 조용히
        pass


def main() -> None:
    if not LABELS.exists():
        raise SystemExit("labels.jsonl 이 없습니다. 먼저 python -m shuding.label")
    print(f"[review] http://localhost:{PORT}  (Ctrl+C로 종료)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
