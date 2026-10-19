from pathlib import Path
p=Path(__file__).parent/'web'/'app.js'
s=p.read_text(encoding='utf-8')
s=s.replace("Number(r.time||0).toFixed(1)+'s'","(r.mode==='scenario'?(r.index+1)+'회':Number(r.time||0).toFixed(1)+'s')")
s=s.replace("$('frame-info').textContent=(selected+1)+' / '+records.length+' · '+Number(r.time||0).toFixed(1)+'s'","$('frame-info').textContent=(selected+1)+' / '+records.length+' · '+(scenario?'호출 '+(r.index+1):Number(r.time||0).toFixed(1)+'s')")
s=s.replace("function table(){const body=$('events');", "function table(){document.querySelector('thead th').textContent=records[0]?.mode==='scenario'?'호출':'시각';const body=$('events');")
p.write_text(s,encoding='utf-8')
