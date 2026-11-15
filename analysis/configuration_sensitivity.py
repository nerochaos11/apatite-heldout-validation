from experiment_core import *
from sklearn.preprocessing import LabelEncoder
d=pd.read_excel(ROOT/'02_原始数据与来源/合并输入_6707条.xlsx')
L={e:np.log10(d[e].astype(float)) for e in E}
X=pd.DataFrame({**{e+'_log':L[e] for e in E},'Sr_Y':L['Sr']-L['Y'],'Th_U':L['Th']-L['U'],'La_Y':L['La']-L['Y']})
y=LabelEncoder().fit(LABELS).transform(d.Dtype)
g=d.Citation.astype(str).str.strip()+'_'+d.Sample_Name.astype(str).str.strip()
s=d.Citation.astype(str).str.strip();rows=[]
for seed in [42,7,13,21,33]:
 for scheme,groups in [('record',None),('sample',g),('source',s)]:
  p=np.zeros((len(d),6))
  cv=StratifiedKFold(5,shuffle=True,random_state=seed) if groups is None else StratifiedGroupKFold(5,shuffle=True,random_state=seed)
  for tr,te in cv.split(X,y,groups):
   xx,yy=SMOTETomek(random_state=seed).fit_resample(X.iloc[tr],y[tr]);mo=model('LightGBM',seed,True);mo.fit(xx,yy)
   pp=mo.predict_proba(X.iloc[te]);p[np.ix_(te,mo.classes_.astype(int))]=pp
  m=metrics(d.Dtype.to_numpy(),p,LABELS);rows.append(dict(seed=seed,scheme=scheme,**m))
  if seed==42:
   pd.DataFrame({**{'p_'+l:p[:,j] for j,l in enumerate(LABELS)},'y_true':d.Dtype,'y_pred':np.array(LABELS)[p.argmax(1)]}).to_csv(OUT/f'配置敏感性_Excel直读_{scheme}_s42.csv',index=False,encoding='utf-8-sig')
  print('CONFIGURATION',seed,scheme,m['macro_f1'],flush=True)
r=pd.DataFrame(rows);r.to_csv(OUT/'配置敏感性_逐种子.csv',index=False,encoding='utf-8-sig')
