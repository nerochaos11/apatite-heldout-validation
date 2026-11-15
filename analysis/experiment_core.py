from pathlib import Path
import sys,json,time,warnings,hashlib,os
import numpy as np,pandas as pd
from sklearn.base import clone,ClassifierMixin,BaseEstimator
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score,average_precision_score,log_loss,confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,StackingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore');sys.stdout.reconfigure(encoding='utf-8')
from paths import ROOT, OUT, INPUT, MODEL_DIR, MAP_DIR, FIGURE_DIR
E=['Th','U','Sr','Y','Nd','La'];F=[e+'_log' for e in E]+['Sr_Y_ratio','Th_U_ratio','La_Y_ratio']
LABELS=['IOCG','Kiruna-type','Orogenic','Porphyry','Skarn','Unmineralized']
SEEDS=[42,7,13,21,33,99,123,2024,555,888]
THREADS=4
def model(name,seed,compat=False):
 common=dict(n_estimators=260,max_depth=8,num_leaves=31,subsample=.85,colsample_bytree=.85,reg_lambda=1.,random_state=seed,verbose=-1,n_jobs=THREADS)
 if name=='LightGBM':return LGBMClassifier(**common,**(dict(learning_rate=.05,class_weight='balanced') if compat else {}))
 if name=='XGBoost':return XGBClassifier(n_estimators=220,max_depth=5,learning_rate=.05,subsample=.85,colsample_bytree=.85,reg_lambda=1.,random_state=seed,n_jobs=THREADS,tree_method='hist',verbosity=0)
 if name=='CatBoost':return CatBoostClassifier(iterations=260,depth=6,learning_rate=.06,l2_leaf_reg=3,random_seed=seed,verbose=0,allow_writing_files=False,thread_count=THREADS)
 if name=='ExtraTrees':return ExtraTreesClassifier(n_estimators=400,max_features='sqrt',class_weight='balanced',random_state=seed,n_jobs=THREADS)
 raise ValueError(name)
def pipe(name,seed,use_smote=True,features=None):
 steps=[('scale',StandardScaler())]
 if use_smote:steps.append(('resample',SMOTETomek(random_state=seed)))
 steps.append(('model',model(name,seed)));return Pipeline(steps)
class LocalClassifier(ClassifierMixin,BaseEstimator):
 def __init__(self,estimator):self.estimator=estimator
 def fit(self,X,y):
  self.classes_,loc=np.unique(y,return_inverse=True);self.estimator_=clone(self.estimator).fit(X,loc);return self
 def predict_proba(self,X):return self.estimator_.predict_proba(X)
 def predict(self,X):return self.classes_[self.predict_proba(X).argmax(1)]
def split(d,scheme,seed=42):
 if scheme=='record':return list(StratifiedKFold(5,shuffle=True,random_state=seed).split(d,d.Dtype))
 g=d[dict(sample='sample',source='source',deposit='deposit_group')[scheme]].to_numpy()
 folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(d,d.Dtype,g))
 assert all(not set(g[a])&set(g[b]) for a,b in folds)
 return folds
def metrics(y,p,labs):
 pred=np.array(labs)[p.argmax(1)];oh=(np.asarray(y)[:,None]==np.array(labs)[None,:]).astype(int)
 out={'accuracy':accuracy_score(y,pred),'balanced_accuracy':recall_score(y,pred,labels=labs,average='macro',zero_division=0),'macro_precision':precision_score(y,pred,labels=labs,average='macro',zero_division=0),'macro_f1':f1_score(y,pred,labels=labs,average='macro',zero_division=0),'macro_roc_auc':roc_auc_score(oh,p,average='macro'),'macro_ap':average_precision_score(oh,p,average='macro'),'log_loss':log_loss(y,p,labels=labs),'mean_brier':np.mean((oh-p)**2)}
 return {k:float(v) for k,v in out.items()}
def load(cohort='main'):
 return pd.read_csv(OUT/({'main':'main_data_clean.csv','raw':'candidates_merged.csv','dedup':'sensitivity_deduplication.csv'}[cohort]))
def run(d,scheme,name='LightGBM',seed=42,task='six',cohort='main',compat=False,use_smote=True,features=None):
 stem=f'{cohort}_{task}_{scheme}_{name}_s{seed}'+('_compat' if compat else '')+('_nosmote' if not use_smote else '')+('_{:d}features'.format(len(features)) if features is not None else '')
 dest=OUT/'OOF';dest.mkdir(exist_ok=True)
 mp=dest/(stem+'_metrics.json');op=dest/(stem+'.csv')
 fingerprint_columns=list(dict.fromkeys(E+(features or F)+['Dtype','source','sample','deposit_group','input_row']))
 data_fingerprint=hashlib.sha256(d[fingerprint_columns].to_csv(index=False).encode()).hexdigest()
 run_config=dict(scheme=scheme,model=name,seed=seed,task=task,cohort=cohort,compat=compat,use_smote=use_smote,features=features or F,threads=THREADS,core_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 run_fingerprint=hashlib.sha256(json.dumps(run_config,sort_keys=True).encode()+data_fingerprint.encode()).hexdigest()
 if mp.exists() and op.exists() and os.environ.get('APATITE_FORCE_RETRAIN')!='1':
  cached=json.loads(mp.read_text())
  if cached.get('run_fingerprint')==run_fingerprint:return cached,pd.read_csv(op)
 X=d[features or F];y=d.Dtype.to_numpy();labs=sorted(set(y));p=np.zeros((len(d),len(labs)));fids=np.zeros(len(d),int);rows=[];inner_rows=[];t=time.time()
 for k,(tr,te) in enumerate(split(d,scheme,seed),1):
  present=sorted(set(y[tr]));local={c:j for j,c in enumerate(present)};yt=np.array([local[v] for v in y[tr]])
  if compat:
   est=model(name,seed,True);Xi,yi=SMOTETomek(random_state=seed).fit_resample(X.iloc[tr],yt);est.fit(Xi,yi)
  elif name=='Stacking':
   if scheme=='record':inner=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(X.iloc[tr],yt))
   else:
    gr=d[dict(sample='sample',source='source',deposit='deposit_group')[scheme]].to_numpy()[tr]
    inner=list(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(X.iloc[tr],yt,gr))
    assert all(not set(gr[a])&set(gr[b]) for a,b in inner)
   for inner_fold,(a,b) in enumerate(inner,1):
    for c in present:
     inner_rows.append(dict(outer_fold=k,inner_fold=inner_fold,cls=c,n_train=int((y[tr[a]]==c).sum()),n_test=int((y[tr[b]]==c).sum()),n_train_sources=d.iloc[tr[a]].loc[y[tr[a]]==c,'source'].nunique(),group_overlap=0 if scheme=='record' else len(set(gr[a])&set(gr[b]))))
   est=StackingClassifier(estimators=[(m,LocalClassifier(pipe(m,seed))) for m in ['XGBoost','LightGBM','CatBoost']],final_estimator=LogisticRegression(max_iter=2000,class_weight='balanced',random_state=seed),cv=inner,stack_method='predict_proba',n_jobs=1)
   est.fit(X.iloc[tr],yt)
  else:est=pipe(name,seed,use_smote);est.fit(X.iloc[tr],yt)
  pp=est.predict_proba(X.iloc[te])
  for j,c in enumerate(est.classes_):p[te,labs.index(present[int(c)])]=pp[:,j]
  assert np.allclose(p[te].sum(1),1) and np.isfinite(p[te]).all()
  assert np.array_equal(np.array(labs)[p[te].argmax(1)],np.array(present)[np.asarray(est.predict(X.iloc[te])).astype(int).ravel()])
  fids[te]=k
  r={'fold':k,'n_train':len(tr),'n_test':len(te),'missing_train':';'.join(sorted(set(labs)-set(present))),'n_train_sources':d.iloc[tr].source.nunique(),'n_test_sources':d.iloc[te].source.nunique(),'macro_f1_fixed':f1_score(y[te],np.array(labs)[p[te].argmax(1)],labels=labs,average='macro',zero_division=0)}
  for c in labs:
   r[c+'_train_records']=int((y[tr]==c).sum());r[c+'_test_records']=int((y[te]==c).sum());r[c+'_train_sources']=d.iloc[tr].loc[y[tr]==c,'source'].nunique()
  rows.append(r)
  print(stem,'fold',k,'complete',round(time.time()-t,1),'s',flush=True)
 o=pd.DataFrame(p,columns=['p_'+l for l in labs]);o['y_true']=y;o['y_pred']=np.array(labs)[p.argmax(1)];o['fold']=fids
 for c in ['source','sample','input_row','deposit_group']:o[c]=d[c].to_numpy()
 o.to_csv(op,index=False,encoding='utf-8-sig');pd.DataFrame(rows).to_csv(dest/(stem+'_folds.csv'),index=False,encoding='utf-8-sig')
 if inner_rows:pd.DataFrame(inner_rows).to_csv(dest/(stem+'_inner_folds.csv'),index=False,encoding='utf-8-sig')
 m=metrics(y,p,labs);m.update({'labels':labs,'n':len(d),'seed':seed,'scheme':scheme,'model':name,'task':task,'cohort':cohort,'seconds':time.time()-t,'data_values_sha256':hashlib.sha256(d[E+['Dtype','source','sample']].to_csv(index=False).encode()).hexdigest()})
 m.update(run_fingerprint=run_fingerprint,run_config=run_config,data_fingerprint=data_fingerprint)
 mp.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8');print('METRICS',stem,json.dumps(m,ensure_ascii=False),flush=True)
 return m,o
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--phase',required=True);args=a.parse_args();d=load()
 if args.phase=='primary':
  for task in ['six','five']:
   q=d if task=='six' else d[d.Dtype!='Unmineralized'].reset_index(drop=True)
   for scheme in ['record','sample','source']+(['deposit'] if task=='five' else []):run(q,scheme,task=task)
 elif args.phase=='benchmark':
  for name in ['XGBoost','CatBoost','ExtraTrees','Stacking']:
   for scheme in ['record','sample','source']:run(d,scheme,name)
 elif args.phase=='repeated':
  for seed in SEEDS[1:]:
   for task in ['six','five']:
    q=d if task=='six' else d[d.Dtype!='Unmineralized'].reset_index(drop=True)
    for scheme in ['record','sample','source']+(['deposit'] if task=='five' else []):run(q,scheme,seed=seed,task=task)
 elif args.phase=='comparison':
  for cohort in ['raw','dedup']:
   q=load(cohort)
   for scheme in ['record','sample','source']:run(q,scheme,cohort=cohort)
  for scheme in ['record','sample','source']:run(load('raw'),scheme,cohort='raw',compat=True)
 elif args.phase=='ablation':
  for fs in [F,F[:6],F[6:]]:
   for sm in [True,False]:run(d,'sample',use_smote=sm,features=fs)
 elif args.phase=='main_compat':
  for scheme in ['record','sample','source']:run(d,scheme,compat=True)
 elif args.phase=='legacy_control':
  old=pd.read_csv(ROOT/'02_source_data/historical/6501_analyses.csv')
  old.to_csv(OUT/'legacy_cohort_6501.csv',index=False,encoding='utf-8-sig')
  for scheme in ['record','sample','source']:run(old,scheme,cohort='legacy_6501')
