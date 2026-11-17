'use strict';

const NFFT = 1024;
const hann = Float64Array.from({length:NFFT}, (_, i)=>.5-.5*Math.cos(2*Math.PI*i/NFFT));
const dbScale = NFFT*hann.reduce((a,b)=>a+b*b,0);
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
function tuneBounds(center, rate) {
 const span=(rate/2-110000)/1e6;
 return [center-span,center+span];
}

function fft(real, imag) {
 for(let i=1,j=0;i<NFFT;i++){
  let bit=NFFT>>1; for(;j&bit;bit>>=1)j^=bit; j^=bit;
  if(i<j){[real[i],real[j]]=[real[j],real[i]];[imag[i],imag[j]]=[imag[j],imag[i]];}
 }
 for(let size=2;size<=NFFT;size*=2){
  for(let k=0;k<size/2;k++){
   const angle=-2*Math.PI*k/size, co=Math.cos(angle), si=Math.sin(angle);
   for(let i=k;i<NFFT;i+=size){
    const j=i+size/2, x=co*real[j]-si*imag[j], y=si*real[j]+co*imag[j];
    real[j]=real[i]-x;imag[j]=imag[i]-y;real[i]+=x;imag[i]+=y;
   }
  }
 }
}

async function readWav(file, onProgress=()=>{}) {
 const view=async (start,end)=>new DataView(await file.slice(start,end).arrayBuffer());
 const id=(v,at)=>String.fromCharCode(...Array.from({length:4},(_,k)=>v.getUint8(at+k)));
 const h=await view(0,12);
 if(h.byteLength!==12 || id(h,0)!=='RIFF' || id(h,8)!=='WAVE')throw Error('Потрібен файл RIFF/WAVE.');
 let format=null, offset=null, bytes=null;
 for(let p=12,count=0;p+8<=file.size && count<512;count++){
  const ch=await view(p,p+8), name=id(ch,0), len=ch.getUint32(4,true);
  if(p+8+len>file.size)throw Error('Файл WAV пошкоджений або скопійований не повністю.');
  if(name==='fmt ' && len>=16)format=await view(p+8,p+8+Math.min(len,40));
  if(name==='data'){offset=p+8;bytes=len;}
  if(format && offset!==null)break;
  p+=8+len+(len%2);
 }
 if(!format || !bytes)throw Error('Не знайдено параметри або відліки WAV.');
 const tag=format.getUint16(0,true), encoding=tag===65534&&format.byteLength>=40?format.getUint16(24,true):tag;
 const channels=format.getUint16(2,true), rate=format.getUint32(4,true), align=format.getUint16(12,true), bits=format.getUint16(14,true), width=bits/8;
 if(channels<1 || channels>2 || align!==channels*width || rate<1 || !((encoding===1&&[8,16,24,32].includes(bits))||(encoding===3&&bits===32)))throw Error('Підтримується WAV із 1–2 каналами: PCM 8–32 біти або float32.');
 const iq=channels===2&&rate>=240000, frames=Math.floor(bytes/align);
 if(frames<NFFT)throw Error('Запис надто короткий для спектра.');
 const count=Math.min(960,Math.ceil(frames/rate*4));
 const sample=(v,k)=>{
  if(encoding===3){const x=v.getFloat32(k,true);return Number.isFinite(x)?clamp(x,-1,1):0;}
  if(bits===8)return (v.getUint8(k)-128)/128;
  if(bits===16)return v.getInt16(k,true)/32768;
  if(bits===24)return ((v.getUint8(k)|v.getUint8(k+1)<<8|v.getUint8(k+2)<<16)<<8>>8)/8388608;
  return v.getInt32(k,true)/2147483648;
 };
 const rows=[];
 for(let i=0;i<count;i++){
  const a=Math.floor(i*frames/count), b=Math.floor((i+1)*frames/count), last=Math.min(frames-NFFT,Math.max(a,b-NFFT));
  const blocks=Math.min(8,Math.max(1,Math.floor((b-a)/NFFT)),Math.max(1,Math.ceil(6000/count)));
  const starts=Array.from({length:blocks},(_,j)=>Math.floor(Math.min(a,last)+(last-Math.min(a,last))*j/Math.max(1,blocks-1)));
  const raw=await Promise.all(starts.map(n=>view(offset+n*align,offset+(n+NFFT)*align)));
  const sums=new Float64Array(iq?NFFT:NFFT/2+1);
  for(const v of raw){
   const re=new Float64Array(NFFT),im=new Float64Array(NFFT);let mu=0,nu=0;
   for(let k=0;k<NFFT;k++){
    const x=sample(v,k*align),y=channels===2?sample(v,k*align+width):0;
    re[k]=iq?x:(x+y)/channels;im[k]=iq?y:0;mu+=re[k];nu+=im[k];
   }
   for(let k=0;k<NFFT;k++){re[k]=(re[k]-mu/NFFT)*hann[k];im[k]=(im[k]-nu/NFFT)*hann[k];}
   fft(re,im);
   for(let k=0;k<sums.length;k++){const j=iq?(k+NFFT/2)%NFFT:k;sums[k]+=re[j]**2+im[j]**2;}
  }
  rows.push(Array.from(sums,x=>Math.max(-120,10*Math.log10(Math.max(1e-12,x/(dbScale*blocks))))));
  if(i%32===0)onProgress(Math.floor(i/count*100));
 }
 const mhz=iq?Number(file.name.match(/(?:^|[_-])(\d{7,10})\s*hz(?:[_\-.]|$)/i)?.[1] || 0)/1e6:null;
 return {filename:file.name,mode:iq?'iq':'audio',duration_s:frames/rate,sample_rate_hz:rate,center_mhz:mhz||null,station_mhz:mhz||null,rows};
}

function parseFFT(content, filename) {
 const raw=JSON.parse(content), frames=raw.frames||(raw.fft_data?[raw]:[]);
 if(!Array.isArray(frames)||!frames.length)throw Error('У JSON немає кадрів FFT.');
 const f=frames[0].frequency, valid=frames.filter(x=>x.frequency===f);
 const n=valid[0].fft_data?.length;
 if(n<2||!n||n>4096||valid.some(x=>!Array.isArray(x.fft_data)||x.fft_data.length!==n||x.fft_data.some(v=>!Number.isFinite(v))))throw Error('Некоректні або неоднакові за розміром кадри FFT.');
 return {filename,mode:'json',duration_s:valid.length*(Number(raw.sample_interval_ms)||100)/1000,
  sample_rate_hz:Number(raw.sample_rate_hz)||null,center_mhz:Number(f)||null,rows:valid.filter((_,i)=>i%Math.ceil(valid.length/960)===0).map(x=>x.fft_data)};
}

const sorted = a=>Array.from(a).sort((a,b)=>a-b);
const at = (a,q)=>a[Math.floor(q*(a.length-1))];
function floorDb(row){return at(sorted(row.slice(Math.floor(row.length*.07),Math.ceil(row.length*.93))),.25);}
// Пілот-тон стосується перевіреної станції; не переносимо результат на іншу частоту.
function stationEvidence(data, station, center) {
 const p=data.stereo_pilot;
 return p?.present&&Math.abs(p.station_mhz-station)<.001&&Math.abs(p.center_mhz-center)<.001?p:null;
}
if(typeof module!=='undefined')module.exports={readWav,parseFFT,floorDb,stationEvidence,tuneBounds};

if(typeof document!=='undefined'){
 const $=id=>document.getElementById(id), w=$('waterfall'), wp=w.getContext('2d'), spec=$('power'), sp=spec.getContext('2d');
 const audio=$('audio'), time=$('time'), playButton=$('play'), centerField=$('center'), stationField=$('station');
 const nf=new Intl.NumberFormat('uk',{maximumFractionDigits:3});
 let data=null, floors=[], raster=null, lo=0, hi=1, play=null, audioFrame=null, currentTime=0;
 let selectedFile=null, transfer=null, capture=null, timer=null, revision=0, channelKey=null, jobUrl=null;
 let stationFollowsCenter=true;
 const audioPlaying=()=>!audio.paused&&!audio.ended;
 const updatePlayButton=()=>{playButton.textContent=play||audioPlaying()?'Пауза':'Відтворити водоспад';};
 const stop=()=>{clearInterval(play);play=null;updatePlayButton();};
 const fmt=s=>{const q=Math.floor(s*10+1e-7);return `${Math.floor(q/600).toString().padStart(2,'0')}:${Math.floor(q%600/10).toString().padStart(2,'0')},${q%10}`;};
 const msg=(m,error=false)=>{$('message').textContent=m;$('message').classList.toggle('error',error);};
 const colors=[[11,22,54],[25,77,180],[24,178,170],[255,230,109]];
 const db=()=>data.mode!=='json';
 const captureCenter=()=>data.center_mhz||Number(centerField.value);
 const selectedStation=()=>Number(stationField.value);
 function frequency(x){
  if(!data.sample_rate_hz)return `${Math.round(x*(data.rows[0].length-1))} · бін FFT`;
  if(data.mode==='audio')return `${nf.format(x*data.sample_rate_hz/2000)} кГц`;
  const center=captureCenter();
  return center?`${nf.format(center+(x-.5)*data.sample_rate_hz/1e6)} МГц`:`${nf.format((x-.5)*data.sample_rate_hz/1000)} кГц від центру`;
 }
 function buildRaster(){
  const h=data.rows.length, n=data.rows[0].length;
  w.width=Math.min(1024,n);w.height=h;raster=wp.createImageData(w.width,h);
  for(let y=0;y<h;y++)for(let x=0;x<w.width;x++){
   const val=data.rows[y][Math.floor(x*n/w.width)], relative=db()?val-floors[y]:val;
   const t=clamp((relative-lo)/(hi-lo),0,.9999)*3, a=Math.floor(t), b=t-a,p=(y*w.width+x)*4;
   for(let c=0;c<3;c++)raster.data[p+c]=colors[a][c]*(1-b)+colors[a+1][c]*b;raster.data[p+3]=255;
  }
 }
 function stationX(){
  const center=captureCenter(), station=selectedStation();
  return center&&station? .5+(station-center)*1e6/data.sample_rate_hz:.5;
 }
 function selection(){
  if(!selectedFile || !data || data.mode==='json')return null;
  if(data.mode==='audio')return {};
  const hint=$('tuneHint');
  const center=captureCenter(), station=selectedStation();
  if(!(center>0&&center<=100000)){hint.textContent='Вкажи центр I/Q, якщо його немає в назві файлу.';return null;}
  const [low,high]=tuneBounds(center,data.sample_rate_hz), range=`${nf.format(low)}–${nf.format(high)} МГц`;
  hint.textContent=`Доступна смуга ${range}. Введи частоту або натисни на спектр.`;
  if(!stationField.value)return null;
  if(!(station>0&&station>=low-1e-9&&station<=high+1e-9)){
   hint.textContent=`Обери станцію в межах ${range}.`;return null;
  }
  return {center_mhz:center,station_mhz:station};
 }
 function status(text,error=false){
  $('fmStatus').textContent=text;$('fmStatus').classList.toggle('error',error);$('retry').hidden=!error;
 }
 async function jsonResponse(response){
  const payload=await response.json();
  if(!response.ok)throw Error(typeof payload.detail==='string'?payload.detail:'Raspberry Pi не прийняла запит.');
  return payload;
 }
 function upload(file){
  if(capture)return Promise.resolve(capture);
  if(transfer)return transfer.promise;
  const xhr=new XMLHttpRequest();
  xhr.open('POST',`/api/fm/captures?${new URLSearchParams({name:file.name})}`);
  xhr.responseType='json';xhr.setRequestHeader('Content-Type','application/octet-stream');
  xhr.upload.onprogress=e=>{
   if(selectedFile===file&&channelKey!==null&&e.lengthComputable)status(`Передаю WAV на Raspberry Pi · ${Math.floor(100*e.loaded/e.total)}%`);
  };
  const promise=new Promise((resolve,reject)=>{
   xhr.onload=()=>{
    if(xhr.status>=200&&xhr.status<300&&xhr.response?.id){
     if(selectedFile===file)capture=xhr.response;
     resolve(xhr.response);
    }else reject(Error(xhr.response?.detail||'Не вдалося передати WAV на Raspberry Pi.'));
   };
   xhr.onerror=xhr.onabort=()=>reject(Error('Передавання WAV перервано.'));
   xhr.onloadend=()=>{if(transfer?.xhr===xhr)transfer=null;};
  });
  transfer={xhr,promise};xhr.send(file);return promise;
 }
 const cancel=url=>{if(url)fetch(url,{method:'DELETE'}).catch(()=>{});};
 function clearAnalysis(){
  revision++;clearTimeout(timer);cancel(jobUrl);jobUrl=null;fmReport.clear();
  $('resultLink').hidden=$('audioLink').hidden=true;
 }
 async function analyze(choice, version){
  try{
   if(!capture)status('Передаю WAV на Raspberry Pi…');
   const input=await upload(selectedFile);
   if(version!==revision)return;
   const endpoint=`/api/fm/captures/${input.id}/analysis`;
   status('Запускаю аналіз…');
   const job=await jsonResponse(await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(choice)}));
   const url=`${endpoint}/${job.id}`;
   if(version!==revision){cancel(url);return;}
   jobUrl=url;
   while(version===revision){
    const progress=await jsonResponse(await fetch(url,{cache:'no-store'}));
    if(version!==revision)return;
    if(progress.state==='failed'||progress.state==='cancelled')throw Error(progress.message);
    status(progress.message);
    if(progress.state==='ready'){
     const report=await jsonResponse(await fetch(`${url}/report`,{cache:'no-store'}));
     if(version!==revision)return;
     fmReport.render(report,url);$('resultLink').hidden=$('audioLink').hidden=false;
     jobUrl=null;return;
    }
    await new Promise(resolve=>setTimeout(resolve,1500));
   }
  }catch(e){if(version===revision)status(e.message,true);}
 }
 function plan(force=false){
  const choice=selection(), key=choice&&JSON.stringify(choice);
  if(key===channelKey&&!force)return;
  clearAnalysis();channelKey=key;
  status(key===null?'':'Готую аналіз вибраного запису…');
  if(choice){const version=revision;timer=setTimeout(()=>analyze(choice,version),force?0:650);}
 }
 function showElapsed(t){
  currentTime=t;
  const label=`${fmt(t)} / ${fmt(data.duration_s)}`;
  if($('now').textContent!==label)$('now').textContent=label;
 }
 function syncFromAudio(){
  if(!data||$('result').hidden||!audio.readyState)return;
  const t=clamp(audio.currentTime,0,data.duration_s);
  const i=clamp(Math.floor(t/data.duration_s*data.rows.length),0,data.rows.length-1);
  if(Number(time.value)!==i){time.value=i;show(t);}else showElapsed(t);
 }
 function followAudio(){
  audioFrame=null;
  if(!audioPlaying())return;
  syncFromAudio();
  audioFrame=requestAnimationFrame(followAudio);
 }
 function settleAudio(){
  cancelAnimationFrame(audioFrame);audioFrame=null;
  if(!play)syncFromAudio();
  updatePlayButton();
 }
 audio.addEventListener('play',()=>{stop();if(audioFrame===null)followAudio();});
 audio.addEventListener('pause',settleAudio);
 audio.addEventListener('ended',settleAudio);
 audio.addEventListener('seeked',()=>{if(!play)syncFromAudio();});

 function show(position){
  if(!data)return;
  const n=data.rows[0].length, i=Number(time.value), row=data.rows[i];
  wp.putImageData(raster,0,0);wp.fillStyle='rgba(255,255,255,.82)';wp.fillRect(0,i,w.width,1);
  const bottom=db()?floors[i]-6:lo, top=db()?floors[i]+hi:hi;
  const sy=v=>spec.height-10-clamp((v-bottom)/(top-bottom),0,1)*(spec.height-20);
  sp.clearRect(0,0,spec.width,spec.height);
  const fx=data.mode==='iq'?stationX():null;
  if(fx!==null&&fx>=0&&fx<=1){
   const margin=65000/data.sample_rate_hz;sp.fillStyle='#123a3e';sp.fillRect((fx-margin)*spec.width,0,2*margin*spec.width,spec.height);
   wp.fillStyle='rgba(110,231,209,.75)';wp.fillRect(fx*w.width,0,1,w.height);
  }
  sp.strokeStyle='#6ee7d1';sp.beginPath();row.forEach((v,j)=>j?sp.lineTo(j*spec.width/(n-1),sy(v)):sp.moveTo(0,sy(v)));sp.stroke();
  for(const [id,p] of [['fmin',0],['fmid',.5],['fmax',1]]) $(id).textContent=frequency(p);
  $('minDb').textContent=`Низ шкали: ${nf.format(bottom)}${db()?' дБ':''}`;$('maxDb').textContent=`Верх шкали: ${nf.format(top)}${db()?' дБ':''}`;
  showElapsed(position??i*data.duration_s/data.rows.length);
 }
 function set(d){
  if(!d.rows?.length || !(d.duration_s>0))throw Error('Запис не містить спектральних кадрів.');
  data=d;
  $('plots').hidden=$('analysis').hidden=false;$('fmSection').hidden=d.mode==='json';
  $('centerLabel').hidden=$('stationLabel').hidden=$('tuneHint').hidden=d.mode!=='iq';
  centerField.value=d.center_mhz??'';
  stationField.value=d.mode==='iq'?centerField.value:'';
  stationFollowsCenter=true;
  centerField.readOnly=Boolean(d.center_mhz);spec.classList.toggle('tunable',d.mode==='iq');
  time.max=d.rows.length-1;time.value=0;
  floors=db()?d.rows.map(floorDb):[];
  if(db()){lo=0;hi=40;}else{const v=sorted(d.rows.flat());lo=at(v,0);hi=Math.max(lo+1,at(v,.995));}
  buildRaster();show();plan();msg(`${d.filename} · ${fmt(d.duration_s)} · ${d.mode==='iq'?'I/Q WAV':d.mode==='audio'?'звуковий WAV':'FFT JSON'}`);
 }
 $('file').onchange=async()=>{
  const f=$('file').files[0];if(!f)return;
  stop();clearAnalysis();channelKey=null;data=null;status('');
  selectedFile=f;capture=null;transfer?.xhr.abort();transfer=null;
  $('plots').hidden=$('analysis').hidden=$('fmSection').hidden=true;
  $('file').disabled=true;
  try{
   msg('Читаю файл у браузері…');const json=/\.json$/i.test(f.name);
   if(json&&f.size>16*1024*1024)throw Error('Файл JSON завеликий (понад 16 МіБ).');
   set(json?parseFFT(await f.text(),f.name):await readWav(f,p=>msg(`Будую водоспад: ${p}%`)));
  }catch(e){msg(e.message,true);}finally{$('file').disabled=false;}
 };
 function chooseTime(){
  stop();show();
  if(audioPlaying())fmReport.seek(currentTime);
 }
 time.oninput=chooseTime;
 w.onclick=e=>{
  const box=w.getBoundingClientRect();
  time.value=clamp(Math.floor((e.clientY-box.top)/box.height*data.rows.length),0,data.rows.length-1);
  chooseTime();
 };
 playButton.onclick=()=>{
  if(audioPlaying()){audio.pause();updatePlayButton();return;}
  if(play){stop();return;}if(Number(time.value)>=data.rows.length-1)time.value=0;
  const step=data.duration_s/data.rows.length;show();
  play=setInterval(()=>{if(Number(time.value)>=data.rows.length-1){stop();return;}time.value=Number(time.value)+1;show();},Math.max(45,1000*step));
  updatePlayButton();
 };
 centerField.oninput=()=>{
  if(stationFollowsCenter)stationField.value=centerField.value;
  show();plan();
 };
 function tune(){
  stationFollowsCenter=!stationField.value||selectedStation()===captureCenter();
  show();plan();
 }
 stationField.oninput=tune;
 $('retry').onclick=()=>plan(true);
 $('audioLink').onclick=()=>{
  stop();
  const from=currentTime<data.duration_s?currentTime:Number(time.value)*data.duration_s/data.rows.length;
  fmReport.seek(from);
 };
 spec.onclick=e=>{
  if(data.mode!=='iq'||!captureCenter())return;
  const box=spec.getBoundingClientRect(), x=clamp((e.clientX-box.left)/box.width,0,1);
  stationField.value=(Math.round((captureCenter()+(x-.5)*data.sample_rate_hz/1e6)*10)/10).toFixed(1);
  tune();
 };
}
