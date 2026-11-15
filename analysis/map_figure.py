from pathlib import Path
import csv,json,math,hashlib
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor,Color
from reportlab.lib.utils import ImageReader
import pypdfium2 as pdfium
from paths import ROOT, MAP_DIR, FIGURE_DIR
WORK=Path(__file__).parent;FIG=FIGURE_DIR;FIG.mkdir(parents=True,exist_ok=True)
M=MAP_DIR
meta=json.loads((M/'地图数据审计.json').read_text(encoding='utf-8'))
audit=json.loads((M/'地图视野参数.json').read_text(encoding='utf-8'))
assert audit.get('model_version')==meta['model_version'], 'Capture current-model screenshots before drawing figure 15'
assert audit.get('training_sha256')==meta['training_sha256'], 'Screenshot and current training cohort differ'
assert audit.get('viewport')=={'width':1000,'height':620,'deviceScaleFactor':1}, 'Map screenshot proportions differ from the layout'
html=M/'磷灰石分类结果_交互地图.html'
assert audit.get('html_sha256')==hashlib.sha256(html.read_bytes()).hexdigest(), 'Screenshots predate the current HTML'
focus_lat=float(meta['focus_lat']);focus_lon=float(meta['focus_lon'])
W,H=482,550;c=canvas.Canvas(str(FIG/'Fig15.pdf'),pagesize=(W,H))
labels=['IOCG','Kiruna-type','Orogenic','Porphyry','Skarn','Unmineralized'];cols=['#B75E61','#D6AD51','#70A47A','#729DC1','#A48BB5','#79B9B5']
def text(x,y,s,size=6.5,align='left',color='#111111'):
 c.setFillColor(HexColor(color));c.setFont('Times-Roman',size);getattr(c,{'left':'drawString','right':'drawRightString','center':'drawCentredString'}[align])(x,y,str(s))
with (M/'地图示例点_输入与概率.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
assert rows, 'Selected coordinate has no analyses'
assert len({r['input_row'] for r in rows})==len(rows)
for r in rows:
 pp=np.array([float(r['p_'+lab]) for lab in labels])
 assert np.isfinite(pp).all() and ((pp>=0)&(pp<=1)).all() and np.isclose(pp.sum(),1)
 assert labels[int(pp.argmax())]==r['Predicted_Class']
 assert r.get('Model_Version',meta['model_version'])==meta['model_version']
counts={lab:sum(r['Predicted_Class']==lab for r in rows) for lab in labels}
assert len(rows)==int(meta['case_records']) and all(counts[lab]==int(meta['case_predictions'].get(lab,0)) for lab in labels)
for j,(lab,col) in enumerate(zip(labels,cols)):
 x=8+j*78;c.setFillColor(HexColor(col));c.setStrokeColor(HexColor('#333333'));c.setLineWidth(.35);c.circle(x,539,2.5,stroke=1,fill=1);text(x+6,537,lab)
views={r['name']:r for r in audit['views']};merc=lambda lat:math.log(math.tan(math.pi/4+math.radians(lat)/2))
for j,name in enumerate(['global','regional','county','local']):
 x=28+(j%2)*246;y=386 if j<2 else 226;w=208;h=w*620/1000
 assert views[name].get('png_sha256')==hashlib.sha256((M/(name+'.png')).read_bytes()).hexdigest(), 'Screenshot bytes do not match the current capture audit'
 c.drawImage(ImageReader(str(M/(name+'.png'))),x,y,width=w,height=h);c.setStrokeColor(HexColor('#000000'));c.setLineWidth(.55);c.rect(x,y,w,h,stroke=1,fill=0)
 text(x,y+h+6,['(a) Global index','(b) Regional view','(c) Site context','(d) Selected coordinate'][j],8)
 bounds=views[name]['bounds'];south=bounds['_southWest']['lat'];west=bounds['_southWest']['lng'];north=bounds['_northEast']['lat'];east=bounds['_northEast']['lng']
 if name=='global':xticks=[-120,0,120];yticks=[-30,0,30,60];prec=0
 elif name=='regional':xticks=[100,110,120];yticks=[20,30,40];prec=0
 else:prec=1 if name=='county' else 3;xticks=np.linspace(west+.15*(east-west),east-.15*(east-west),3);yticks=np.linspace(south+.15*(north-south),north-.15*(north-south),3)
 for v in xticks:
  xx=x+(v-west)/(east-west)*w;c.line(xx,y,xx,y+3);text(xx,y-10,f'{abs(v):.{prec}f}°'+('W' if v<0 else 'E'),5.8,'center')
 for v in yticks:
  yy=y+(merc(v)-merc(south))/(merc(north)-merc(south))*h;c.line(x,yy,x+3,yy);text(x-3,yy-2,f'{abs(v):.{prec}f}°'+('S' if v<0 else 'N'),5.8,'right')
 # Enlarge the selected coordinate without changing the data encoded by its pie.
 # The original Leaflet symbols are fixed-size pies, not proportional-area symbols.
 fx=x+(focus_lon-west)/(east-west)*w
 fy_map=y+(merc(focus_lat)-merc(south))/(merc(north)-merc(south))*h
 if j in [2,3]:
  c.setFillColor(HexColor('#ffffff'));c.circle(fx,fy_map,8,stroke=0,fill=1)
  angle=90
  for lab,col in zip(labels,cols):
   if counts[lab]:
    extent=360*counts[lab]/len(rows);c.setFillColor(HexColor(col));c.setStrokeColor(HexColor('#ffffff'));c.setLineWidth(.4)
    c.wedge(fx-6.5,fy_map-6.5,fx+6.5,fy_map+6.5,angle,-extent,stroke=1,fill=1);angle-=extent
  c.setStrokeColor(HexColor('#111111'));c.setLineWidth(.55);c.circle(fx,fy_map,6.5,stroke=1,fill=0)
  c.line(fx+8,fy_map,fx+18,fy_map+10);text(fx+19,fy_map+10,f'{len(rows)} analyses',6.5,color='#ffffff')
 if j==3:
  lines=[];line='Selected: '
  for lab in labels:
   if not counts[lab]:continue
   piece=f'{lab} {counts[lab]}';candidate=line+(' / ' if not line.endswith(': ') else '')+piece
   if c.stringWidth(candidate,'Times-Roman',6.1)>151 and not line.endswith(': '):lines.append(line);line=piece
   else:line=candidate
  if line:lines.append(line)
  lines.append('Sector angles follow record counts');boxh=len(lines)*8+5
  c.setFillColor(HexColor('#ffffff'));c.rect(x+4,y+h-boxh-4,159,boxh,stroke=0,fill=1)
  for k,line in enumerate(lines):text(x+8,y+h-13-k*8,line,6.1)
 if j in [0,2]:
  xx=x+w+8;yy=y+h+9;c.line(xx,yy,xx+18,yy);c.line(xx+18,yy,xx+14,yy+3);c.line(xx+18,yy,xx+14,yy-3)
samples=sorted(set(r['SAMPLE_NAME'] for r in rows));assert len(samples)==int(meta['case_samples'])
assert 1<=len(samples)<=2, 'Choose a larger sample-symbol palette before expanding this site example'
sc=['#426B83','#AF7657'];x,y,w,h=34,60,202,128
text(28,202,f'(e) Composition of {len(samples)} registered '+('sample' if len(samples)==1 else 'samples'),7.2)
c.setStrokeColor(HexColor('#000000'));c.setLineWidth(.55);c.rect(x,y,w,h,stroke=1,fill=0)
values=np.array([[float(r[e]) for e in ['Th','U','Sr','Y','Nd','La']] for r in rows]);assert np.isfinite(values).all() and (values>0).all()
lo=math.floor(math.log10(float(values.min())));hi=math.ceil(math.log10(float(values.max())))
if lo==hi:hi+=1
fy=lambda v:y+(math.log10(v)-lo)/(hi-lo)*h
for power in range(lo,hi+1):
 v=10**power;yy=fy(v);c.line(x,yy,x+3,yy);text(x-4,yy-2,f'{v:g}',6,'right')
for j,e in enumerate(['Th','U','Sr','Y','Nd','La']):
 xx=x+(j+.5)/6*w
 for si,s in enumerate(samples):
  vals=np.array([float(r[e]) for r in rows if r['SAMPLE_NAME']==s]);center=xx+(si-(len(samples)-1)/2)*9
  for off,v in zip(np.linspace(-2,2,len(vals)),vals):
   px,py=center+off,fy(v);c.setStrokeColor(HexColor(sc[si]));c.setFillColor(HexColor(sc[si]));c.setLineWidth(.6)
   if si==0:c.circle(px,py,1.35,stroke=1,fill=0)
   else:
    p=c.beginPath();p.moveTo(px,py+1.6);p.lineTo(px-1.4,py-1.1);p.lineTo(px+1.4,py-1.1);p.close();c.drawPath(p,stroke=1,fill=1)
  c.setLineWidth(.8);c.line(center-3,fy(np.median(vals)),center+3,fy(np.median(vals)))
 c.setStrokeColor(HexColor('#111111'));c.setLineWidth(.55);c.line(xx,y,xx,y+3);text(xx,y-10,e,6.5,'center')
c.saveState();c.translate(9,112);c.rotate(90);text(0,0,'Concentration (ppm; log scale)',6.5,'center');c.restoreState()
for si,s in enumerate(samples):
 yy=29-si*10;c.setStrokeColor(HexColor(sc[si]));c.setFillColor(HexColor(sc[si]));c.setLineWidth(.6)
 if si==0:c.circle(30,yy+2,1.5,stroke=1,fill=0)
 else:
  pp=c.beginPath();pp.moveTo(30,yy+3.7);pp.lineTo(28.5,yy+.7);pp.lineTo(31.5,yy+.7);pp.close();c.drawPath(pp,stroke=1,fill=1)
 text(36,yy,s.replace('samp. ','')+f' (n = {sum(r["SAMPLE_NAME"]==s for r in rows)})',6.3,color=sc[si])
text(276,202,'(f) Probabilities for each analysis (%)',7.2)
x,y,w,h=294,60,164,128;rw=h/len(rows);cw=w/6
rows=sorted(rows,key=lambda r:(r['SAMPLE_NAME'],int(r['input_row'])))
for i,r in enumerate(rows):
 yy=y+h-(i+1)*rw;si=samples.index(r['SAMPLE_NAME']);c.setFillColor(HexColor(sc[si]));c.rect(x-19,yy,3,rw,stroke=0,fill=1);text(x-3,yy+rw/2-2,r['input_row'],5.6,'right')
 for j,l in enumerate(labels):
  p=float(r['p_'+l]);r0,g0,b0=(.965,.979,.985);r1,g1,b1=(.12,.31,.49);c.setFillColor(Color(r0+(r1-r0)*p,g0+(g1-g0)*p,b0+(b1-b0)*p));c.setStrokeColor(HexColor('#ffffff'));c.setLineWidth(.4);c.rect(x+j*cw,yy,cw,rw,stroke=1,fill=1)
  text(x+(j+.5)*cw,yy+rw/2-2,'<0.1' if p*100<.1 else f'{p*100:.1f}',5.7,'center',color='#ffffff' if p>.6 else '#202D36')
for j,l in enumerate(['IOCG','Kiruna','Orogenic','Porphyry','Skarn','Unmin.']):text(x+(j+.5)*cw,y-11,l,5.8,'center')
text(x-4,y+h+3,'Row',5.5,'right')
for i in range(100):
 p=i/99;c.setFillColor(Color(.965+(.12-.965)*p,.979+(.31-.979)*p,.985+(.49-.985)*p));c.rect(464,y+i*h/100,4,h/100+.1,stroke=0,fill=1)
for v in [0,50,100]:text(471,y+v/100*h-2,str(v),5)
text(376,29,'Uncalibrated; grouped by registered sample',6.2,'center')
text(0,12,'Basemap: Esri World Imagery (Esri, Maxar, Earthstar Geographics); imagery is not used in classification.',6)
lat_text=f'{abs(focus_lat):.4f} '+('N' if focus_lat>=0 else 'S');lon_text=f'{abs(focus_lon):.4f} '+('E' if focus_lon>=0 else 'W')
text(0,3,f'LightGBM refit on {int(meta["training_n"]):,} analyses. Selected coordinate: {lat_text}, {lon_text}. No per-analysis ground truth.',6)
c.save();doc=pdfium.PdfDocument(str(FIG/'Fig15.pdf'));doc[0].render(scale=400/72).to_pil().save(FIG/'Fig15.png',dpi=(400,400));doc.close();print('Figure 15 rebuilt with sample-specific symbols and per-analysis probabilities')
