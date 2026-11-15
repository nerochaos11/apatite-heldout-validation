from experiment_core import *
import joblib,shutil
M=MAP_DIR;M.mkdir(parents=True,exist_ok=True)
old=ROOT/'04_分析代码'
assets=ROOT/'06_模型与地图/交互地图/assets'
if assets.resolve()!=(M/'assets').resolve():shutil.copytree(assets,M/'assets',dirs_exist_ok=True)
bundle=joblib.load(MODEL_DIR/'LightGBM_主数据_全量重拟合.joblib');mo=bundle['pipeline'];labs=bundle['labels'];train=load()
assert bundle['features']==F and bundle['n']==len(train)
assert bundle['training_sha256']==hashlib.sha256((OUT/'主数据_清洗后.csv').read_bytes()).hexdigest()
assert bundle['model_version']=='LGBM-main-refit-20261022'
inp=ROOT/'02_原始数据与来源/应用输入/附坐标资料.csv'
d=pd.read_csv(inp,encoding='utf-8-sig',low_memory=False,float_precision='round_trip');d['input_row']=np.arange(len(d))+2
for e in E:d[e]=pd.to_numeric(d[e],errors='coerce')
ok=np.isfinite(d[E]).all(axis=1)&(d[E]>0).all(axis=1)
q=d.loc[ok,['input_row']+E+['Latitude','Longitude','SAMPLE_NAME','ROCK_NAME','LOCATION','CITATION','SOURCE_FILE']].copy()
def feats(a):
 x=pd.DataFrame({e+'_log':np.log10(a[e]) for e in E},index=a.index)
 for k,u,v in [('Sr_Y_ratio','Sr','Y'),('Th_U_ratio','Th','U'),('La_Y_ratio','La','Y')]:x[k]=x[u+'_log']-x[v+'_log']
 return x[F]
p=mo.predict_proba(feats(q));assert np.allclose(p.sum(1),1)
for j,l in enumerate(labs):q['p_'+l]=p[:,j]
q['Predicted_Class']=np.array(labs)[p.argmax(1)];q['Uncalibrated_Max_Probability']=p.max(1)
# Floating inputs from independently exported tables may differ below analytical precision.
# Use an explicitly documented eight-decimal comparison key, with exact matches also retained.
raw_exact=set(train[E].itertuples(index=False,name=None));sig=set(map(tuple,train[E].round(8).to_numpy()))
q['Matches_Main_Six_Elements']=[v in sig for v in map(tuple,q[E].round(8).to_numpy())]
q['Exact_Binary_Float_Match']=[v in raw_exact for v in q[E].itertuples(index=False,name=None)]
q['Latitude']=pd.to_numeric(q.Latitude,errors='coerce');q['Longitude']=pd.to_numeric(q.Longitude,errors='coerce')
q['Coordinate_Valid']=q.Latitude.between(-85.05112878,85.05112878)&q.Longitude.between(-180,180)
q['Model_Version']='LGBM-main-refit-20261022'
q.to_csv(M/'地图逐记录预测.csv',index=False,encoding='utf-8-sig')
mapped=q[q.Coordinate_Valid];rows=[]
for (lat,lon,cl),g in mapped.groupby(['Latitude','Longitude','Predicted_Class'],sort=True):
 mm=~g.Matches_Main_Six_Elements
 rows.append({'lat':float(lat),'lon':float(lon),'class':cl,'n':len(g),'matches':int((~mm).sum()),'mean_p':float(g.Uncalibrated_Max_Probability.mean()),'n_nonmatch':int(mm.sum()),'mean_p_nonmatch':float(g.loc[mm,'Uncalibrated_Max_Probability'].mean()) if mm.any() else None,'samples':g.SAMPLE_NAME.fillna('').astype(str).drop_duplicates().head(6).tolist(),'rocks':g.ROCK_NAME.fillna('').astype(str).drop_duplicates().head(3).tolist(),'locations':g.LOCATION.fillna('').astype(str).drop_duplicates().head(3).tolist(),'citations':g.CITATION.fillna('').astype(str).drop_duplicates().head(4).tolist(),'input_rows':g.input_row.tolist(),'elements':{e:float(g[e].median()) for e in E}})
pd.DataFrame(rows).to_csv(M/'坐标类别分组.csv',index=False,encoding='utf-8-sig')
focus=(22.2914,110.1661);case=mapped[(mapped.Latitude==focus[0])&(mapped.Longitude==focus[1])]
case.to_csv(M/'地图示例点_输入与概率.csv',index=False,encoding='utf-8-sig')
meta=dict(model_version='LGBM-main-refit-20261022',input_records=len(d),positive_complete_records=len(q),mapped_records=len(mapped),invalid_coordinates=len(q)-len(mapped),unique_coordinates=len(mapped[['Latitude','Longitude']].drop_duplicates()),coordinate_class_groups=len(rows),six_element_matches_all=int(q.Matches_Main_Six_Elements.sum()),six_element_matches_mapped=int(mapped.Matches_Main_Six_Elements.sum()),display_nonmatch_records=int((~mapped.Matches_Main_Six_Elements).sum()),nonmatch_coordinates=len(mapped.loc[~mapped.Matches_Main_Six_Elements,['Latitude','Longitude']].drop_duplicates()),focus_lat=focus[0],focus_lon=focus[1],training_n=len(train),case_records=len(case),case_samples=case.SAMPLE_NAME.nunique(),case_predictions=case.Predicted_Class.value_counts().to_dict(),class_counts=q.Predicted_Class.value_counts().to_dict(),overlap_definition='six concentrations rounded to 8 decimal places in ppm; sensitivity to export precision only, not source independence',training_sha256=hashlib.sha256((OUT/'主数据_清洗后.csv').read_bytes()).hexdigest(),input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),probability='uncalibrated; application demonstration, no ground-truth accuracy')
(M/'地图数据审计.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
(M/'地图分组数据.json').write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
template=(old/'map_template.html').read_text(encoding='utf-8').replace('六元素重合','六元素数值重合').replace('默认隐藏数值重合记录','按ppm保留8位小数比较，默认隐藏数值重合记录')
for k,v in [('__MAP_DATA__',json.dumps(rows,ensure_ascii=False)),('__MAP_META__',json.dumps(meta,ensure_ascii=False)),('__LEAFLET_JS__',(M/'assets/leaflet.js').read_text(encoding='utf-8')),('__LEAFLET_CSS__',(M/'assets/leaflet.css').read_text(encoding='utf-8'))]:template=template.replace(k,v.replace('</','<\\/') if k.startswith('__MAP_') else v)
(M/'磷灰石分类结果_交互地图.html').write_text(template,encoding='utf-8')
print('MAP',json.dumps(meta,ensure_ascii=False),flush=True)
# Recalculate the three pre-existing application datasets. No per-analysis ground truth is assumed.
base=ROOT/'02_原始数据与来源/应用输入'
fo=ROOT/'02_原始数据与来源/应用输入/Fosterville.csv'
er=[]
for name,path in [('Fosterville',fo),('Bushveld',base/'Bushveld.csv'),('Darongxi',base/'Darongxi.csv')]:
 a=pd.read_csv(path,encoding='latin1' if name=='Fosterville' else 'utf-8-sig',low_memory=False)
 if name=='Fosterville':
  v=pd.DataFrame(index=a.index)
  for e in E:
   col=next(c for c in a.columns if c.upper().replace(' ','')==e.upper()+'(PPM)')
   v[e]=pd.to_numeric(a[col],errors='coerce')
  a=v
 a=a[np.isfinite(a[E]).all(axis=1)&(a[E]>0).all(axis=1)].copy()
 pp=mo.predict_proba(feats(a));a['prediction']=np.array(labs)[pp.argmax(1)];a['confidence']=pp.max(1)
 for j,l in enumerate(labs):a['p_'+l]=pp[:,j]
 a['main_numeric_overlap']=[tuple(r) in sig for r in a[E].round(8).to_numpy()]
 a.to_csv(OUT/f'应用资料_{name}_预测.csv',index=False,encoding='utf-8-sig')
 er.append(dict(dataset=name,n=len(a),mean_confidence=float(a.confidence.mean()),overlap=int(a.main_numeric_overlap.sum()),prediction_counts=a.prediction.value_counts().to_dict()))
(OUT/'应用资料汇总.json').write_text(json.dumps(er,ensure_ascii=False,indent=2),encoding='utf-8')
print('EXTERNAL',json.dumps(er,ensure_ascii=False),flush=True)
