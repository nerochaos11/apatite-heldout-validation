from experiment_core import *
from scipy.optimize import minimize,minimize_scalar
import joblib
d=load();labs=sorted(d.Dtype.unique());y=d.Dtype.to_numpy();yi=np.array([labs.index(v) for v in y]);X=d[F]
def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def ece(p,truth):
 conf=p.max(1);correct=p.argmax(1)==truth;rows=[];total=0.
 for j in range(10):
  m=(conf>=j/10)&(conf<(j+1)/10 if j<9 else conf<=1)
  a=float(correct[m].mean()) if m.any() else None;c=float(conf[m].mean()) if m.any() else None
  rows.append(dict(bin=j,n=int(m.sum()),confidence=c,accuracy=a))
  if m.any():total+=m.mean()*abs(a-c)
 return total,rows
def temp(p,t):
 z=np.log(np.clip(p,1e-12,1))/t;z-=z.max(1,keepdims=True);a=np.exp(z);return a/a.sum(1,keepdims=True)
def calibration():
 raw=np.zeros((len(d),6));cal=raw.copy();temps=[]
 for k,(tr,te) in enumerate(split(d,'sample'),1):
  inner=list(StratifiedGroupKFold(4,shuffle=True,random_state=42).split(X.iloc[tr],y[tr],d['sample'].to_numpy()[tr]))
  fit,hold=inner[0];a,b=tr[fit],tr[hold]
  assert not set(d['sample'].iloc[a])&set(d['sample'].iloc[b])
  m=pipe('LightGBM',42);m.fit(X.iloc[a],yi[a]);p=m.predict_proba(X.iloc[b])
  assert np.array_equal(m.classes_,np.arange(6))
  opt=minimize_scalar(lambda logt:log_loss(yi[b],temp(p,np.exp(logt)),labels=range(6)),bounds=(-2,2),method='bounded');t=float(np.exp(opt.x));temps.append(t)
  raw[te]=m.predict_proba(X.iloc[te]);cal[te]=temp(raw[te],t)
  print('calibration',k,t,flush=True)
 a,b=metrics(y,raw,labs),metrics(y,cal,labs);a['ece'],ar=ece(raw,yi);b['ece'],br=ece(cal,yi)
 assert np.array_equal(raw.argmax(1),cal.argmax(1))
 save('calibration.json',dict(temperatures=temps,before=a,after=b,before_bins=ar,after_bins=br))
 pd.DataFrame({**{'raw_'+l:raw[:,j] for j,l in enumerate(labs)},**{'cal_'+l:cal[:,j] for j,l in enumerate(labs)},'y_true':y,'input_row':d.input_row}).to_csv(OUT/'calibration_per_record.csv',index=False,encoding='utf-8-sig')
def shap():
 rows=[];classrows=[]
 for k,(tr,te) in enumerate(split(d,'sample'),1):
  mo=model('LightGBM',42);mo.fit(X.iloc[tr],yi[tr]);v=mo.predict(X.iloc[te],pred_contrib=True).reshape(len(te),6,len(F)+1)
  raw=mo.predict(X.iloc[te],raw_score=True);assert np.allclose(v.sum(2),raw)
  avg=np.abs(v[:,:,:-1]).mean((0,1));order=np.argsort(np.argsort(-avg))+1
  rows += [dict(fold=k,feature=f,mean_abs_shap=float(avg[j]),rank=int(order[j])) for j,f in enumerate(F)]
  classrows += [dict(fold=k,cls=l,feature=f,mean_abs_shap=float(np.abs(v[:,c,j]).mean())) for c,l in enumerate(labs) for j,f in enumerate(F)]
  print('shap',k,flush=True)
 pd.DataFrame(rows).to_csv(OUT/'SHAP_per_fold.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(classrows).to_csv(OUT/'SHAP_per_class_per_fold.csv',index=False,encoding='utf-8-sig')
def variance():
 results=[]
 for element in E:
  chunks=[]
  for src,g in d.groupby('source'):
   rows=[]
   for key,h in g.groupby('sample'):
    v=h[element+'_log'].to_numpy();rows.append((len(v),v.sum(),v@v,labs.index(h.Dtype.iloc[0])))
   chunks.append(np.array(rows))
  def objective(theta):
   a,b,e=np.exp(theta);ld=0.;yy=0.;xx=np.zeros((6,6));xy=np.zeros(6)
   for block in chunks:
    n,s,t,c=block.T;c=c.astype(int);dd=e+n*b;q=n/dd;r=s/dd
    Q=q.sum();R=r.sum();fac=a/(1+a*Q);qc=np.bincount(c,weights=q,minlength=6);rc=np.bincount(c,weights=r,minlength=6)
    ld+=np.sum((n-1)*np.log(e)+np.log(dd))+np.log1p(a*Q)
    yy+=np.sum(t/e-b*s*s/(e*dd))-fac*R*R;xx+=np.diag(qc)-fac*np.outer(qc,qc);xy+=rc-fac*qc*R
   sg,lx=np.linalg.slogdet(xx)
   return 1e30 if sg<=0 else .5*(ld+lx+yy-xy@np.linalg.solve(xx,xy)+(len(d)-6)*np.log(2*np.pi))
  fits=[minimize(objective,np.log(v),method='L-BFGS-B',bounds=[(-18,5)]*3,options={'maxiter':500,'ftol':1e-10}) for v in [[.1,.1,.1],[.5,.05,.2]]]
  fit=min(fits,key=lambda x:x.fun)
  if not fit.success:
   extra=minimize(objective,fit.x,method='Powell',bounds=[(-18,5)]*3,options={'maxiter':1500,'ftol':1e-10,'xtol':1e-8})
   if extra.success and extra.fun<=fit.fun+1e-5:fit=extra
  a,b,e=np.exp(fit.x);total=a+b+e
  results.append(dict(element=element,source_variance=a,sample_variance=b,residual_variance=e,source_fraction=a/total,sample_fraction=b/total,residual_fraction=e/total,same_sample_icc=(a+b)/total,converged=bool(fit.success),boundary=bool((fit.x<-17.9).any()),objective=fit.fun))
  print('REML',element,results[-1],flush=True)
 pd.DataFrame(results).to_csv(OUT/'nested_variance_REML.csv',index=False,encoding='utf-8-sig')
def loso():
 dest=OUT/'LOSO';dest.mkdir(exist_ok=True);allrows=[]
 for src in sorted(d.source.unique()):
  f=dest/(src.replace("'",'')+'.csv')
  if f.exists() and os.environ.get('APATITE_FORCE_RETRAIN')!='1':allrows.append(pd.read_csv(f));continue
  te=np.flatnonzero(d.source.to_numpy()==src);tr=np.flatnonzero(d.source.to_numpy()!=src)
  mo=pipe('LightGBM',42);mo.fit(X.iloc[tr],yi[tr]);p=mo.predict_proba(X.iloc[te]);assert np.array_equal(mo.classes_,np.arange(6))
  q=pd.DataFrame(p,columns=['p_'+l for l in labs]);q['y_true']=y[te];q['y_pred']=np.array(labs)[p.argmax(1)];q['source']=src;q['input_row']=d.input_row.to_numpy()[te]
  q.to_csv(f,index=False,encoding='utf-8-sig');allrows.append(q);print('LOSO',src,len(q),float((q.y_true==q.y_pred).mean()),flush=True)
 pd.concat(allrows).to_csv(OUT/'LOSO_record_level.csv',index=False,encoding='utf-8-sig')
def fullfit():
 MODEL_DIR.mkdir(parents=True,exist_ok=True)
 mo=pipe('LightGBM',42);mo.fit(X,yi)
 training_hash=hashlib.sha256((OUT/'main_data_clean.csv').read_bytes()).hexdigest()
 joblib.dump(dict(pipeline=mo,labels=labs,features=F,n=len(d),training_sha256=training_hash,model_version='LGBM-main-refit-20261022'),MODEL_DIR/'LightGBM_main_full_refit.joblib')
 print('fullfit saved',flush=True)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--phase');a=p.parse_args()
 if a.phase=='diagnostics':calibration();shap();variance();fullfit()
 if a.phase=='loso':loso()
 if a.phase=='variance':variance()
