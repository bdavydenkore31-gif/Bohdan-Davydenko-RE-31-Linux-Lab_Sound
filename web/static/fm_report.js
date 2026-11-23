'use strict';

window.fmReport = (()=>{
 const $=id=>document.getElementById(id), audio=$('audio');
 const kinds={speech:'Мовлення',music:'Музика',mixed:'Мовлення + музика',music_voice:'Музика + голос',music_vocals:'Музика + спів',unknown:'Невизначено'};
 const voices={male_likely:'Чоловічий',female_likely:'Жіночий',unknown:'Не визначено'};
 const nf=new Intl.NumberFormat('uk',{maximumFractionDigits:3});
 const fmt=s=>{const t=Math.round(s*10);return `${Math.floor(t/600).toString().padStart(2,'0')}:${((t%600)/10).toFixed(1).padStart(4,'0').replace('.',',')}`;};
 let base='', segments=[], rowNodes=[], active=-1, pending=null, source='audio', resume=false;
 const voiceLabel=s=>voices[s.voice]||'—';

 function seek(time){
  if(audio.readyState)audio.currentTime=time;
  else pending=time;
  audio.play().catch(()=>{});
 }
 audio.onloadedmetadata=()=>{
  if(pending!==null){audio.currentTime=pending;pending=null;}
  if(resume){resume=false;audio.play().catch(()=>{});}
 };
 // Перемикання доріжки зберігає час і стан відтворення.
 function selectTrack(value){
  if(source===value)return;
  pending=audio.readyState?audio.currentTime:pending;
  resume=!audio.paused || resume;
  source=value;
  $('original').setAttribute('aria-pressed',String(source==='audio'));
  $('neural').setAttribute('aria-pressed',String(source==='neural'));
  audio.src=`${base}/${source}`;
 }
 $('original').onclick=()=>selectTrack('audio');
 $('neural').onclick=()=>selectTrack('neural');
 function highlight(){
  const t=audio.currentTime, next=segments.findIndex(s=>t>=s.start_s&&t<s.end_s);
  if(active===next)return;
  if(active>=0)rowNodes[active]?.classList.remove('active');
  active=next;
  if(next<0){$('reportNow').textContent='';return;}
  const s=segments[next];rowNodes[next]?.classList.add('active');
  $('reportNow').textContent=`${fmt(s.start_s)}–${fmt(s.end_s)} · ${kinds[s.content]}${s.voice?' · '+voiceLabel(s):''}`;
 }
 audio.ontimeupdate=highlight;audio.onseeked=highlight;

 function clear(){
  $('result').hidden=true;
  if(audio.hasAttribute('src')){audio.pause();audio.removeAttribute('src');audio.load();}
  base='';segments=[];rowNodes=[];active=-1;pending=null;resume=false;
 }
 // Кольорова шкала й таблиця будуються з одного набору фрагментів звіту.
 function render(report, url){
  clear(); base=url; segments=report.segments; source='audio';
  $('original').setAttribute('aria-pressed','true');$('neural').setAttribute('aria-pressed','false');
  $('reportMeta').textContent=`${report.filename} · ${fmt(report.duration_s)}${report.station_mhz?' · '+nf.format(report.station_mhz)+' МГц':''}`;
  $('timeline').replaceChildren();$('legend').replaceChildren();$('rows').replaceChildren();$('reportNow').textContent='';
  for(const [key,label] of Object.entries(kinds)){
   if(!report.totals_s[key])continue;
   const item=document.createElement('span'), dot=document.createElement('span');dot.className=`dot ${key}`;
   item.append(dot,`${label} · ${nf.format(report.totals_s[key])} с`);$('legend').append(item);
  }
  for(const s of segments){
   const button=document.createElement('button');button.className=s.content;
   button.style.flex=`${(s.end_s-s.start_s)/report.duration_s} 1 0`;button.title=`${fmt(s.start_s)}–${fmt(s.end_s)} · ${kinds[s.content]}`;
   button.setAttribute('aria-label',button.title);button.onclick=()=>seek(s.start_s);$('timeline').append(button);
   const row=$('rows').insertRow(), start=row.insertCell(), jump=document.createElement('button');
   jump.textContent=fmt(s.start_s);jump.onclick=()=>seek(s.start_s);start.append(jump);
   for(const value of [fmt(s.end_s),kinds[s.content],voiceLabel(s),s.f0_hz==null?'—':`${nf.format(s.f0_hz)} Гц`])row.insertCell().textContent=value;
   rowNodes.push(row);
  }
  audio.src=`${base}/${source}`;$('result').hidden=false;highlight();
 }
 return {clear,render,seek};
})();
