from experiment_core import *
from sklearn.metrics import confusion_matrix
def read_numeric_csv(*args,**kwargs):
 kwargs.setdefault('float_precision','round_trip')
 return pd.read_csv(*args,**kwargs)
files=list((OUT/'OOF').glob('*_metrics.json'));rs=[];audit=[]
for f in files:
 m=json.loads(f.read_text(encoding='utf-8'));o=read_numeric_csv(f.with_name(f.name.replace('_metrics.json','.csv')))
 labs=m['labels'];p=o[['p_'+l for l in labs]].to_numpy();assert np.allclose(p.sum(1),1)
 assert len(o)==m['n'] and o.input_row.nunique()==len(o)
 assert np.array_equal(o.y_pred,np.array(labs)[p.argmax(1)])
 mm=metrics(o.y_true.to_numpy(),p,labs)
 assert max(abs(mm[k]-m[k]) for k in mm)<1e-10
 audit.append(dict(file=f.name,n=len(o),all_metrics_recomputed=True));rs.append({**m,'stem':f.name.replace('_metrics.json','')})
t=pd.DataFrame(rs);t.drop(columns=['labels']).to_csv(OUT/'all_experiment_metrics.csv',index=False,encoding='utf-8-sig')
rep=t[(t.cohort=='main')&(t.model=='LightGBM')&~t.stem.str.contains('compat|nosmote|features')]
summary=rep.groupby(['task','scheme']).agg(n_seeds=('seed','size'),mean_f1=('macro_f1','mean'),std_f1=('macro_f1','std'),mean_auc=('macro_roc_auc','mean'),std_auc=('macro_roc_auc','std'),min_f1=('macro_f1','min'),max_f1=('macro_f1','max')).reset_index()
summary.to_csv(OUT/'ten_seed_summary.csv',index=False,encoding='utf-8-sig')
source_rows=[]
for protocol,op in [('source_fivefold',OUT/'OOF/main_six_source_LightGBM_s42.csv'),('LOSO',OUT/'LOSO_record_level.csv')]:
 o=read_numeric_csv(op)
 for (source,cls),g in o.groupby(['source','y_true']):
  source_rows.append(dict(protocol=protocol,source=source,cls=cls,n=len(g),correct=int((g.y_true==g.y_pred).sum()),recall=float((g.y_true==g.y_pred).mean()),predictions=json.dumps(g.y_pred.value_counts().to_dict(),ensure_ascii=False)))
pd.DataFrame(source_rows).to_csv(OUT/'per_source_diagnostics.csv',index=False,encoding='utf-8-sig')
perclass=[]
for scheme in ['sample','source']:
 o=read_numeric_csv(OUT/f'OOF/main_six_{scheme}_LightGBM_s42.csv')
 for c in LABELS:
  a=o.y_true==c;b=o.y_pred==c;tp=int((a&b).sum());pr=tp/max(b.sum(),1);rc=tp/a.sum();ff=2*pr*rc/(pr+rc) if pr+rc else 0
  perclass.append(dict(scheme=scheme,cls=c,n=int(a.sum()),precision=pr,recall=rc,f1=ff))
pd.DataFrame(perclass).to_csv(OUT/'results_by_class.csv',index=False,encoding='utf-8-sig')
o=read_numeric_csv(OUT/'OOF/main_six_sample_LightGBM_s42.csv');labs=LABELS
rows=[dict(target='All records',**metrics(o.y_true.to_numpy(),o[['p_'+l for l in labs]].to_numpy(),labs))]
first=o.groupby('sample',sort=False).head(1);rows.append(dict(target='First per sample',**metrics(first.y_true.to_numpy(),first[['p_'+l for l in labs]].to_numpy(),labs)))
agg=o.groupby('sample',sort=False).agg({**{'p_'+l:'mean' for l in labs},'y_true':'first'})
rows.append(dict(target='Mean per sample',**metrics(agg.y_true.to_numpy(),agg[['p_'+l for l in labs]].to_numpy(),labs)))
pd.DataFrame(rows).to_csv(OUT/'scoring_target_comparison.csv',index=False,encoding='utf-8-sig')
# Conditional descriptive bootstrap of held-out source contributions, keeping fixed OOF predictions.
o=read_numeric_csv(OUT/'OOF/main_six_source_LightGBM_s42.csv');cms=np.array([confusion_matrix(g.y_true,g.y_pred,labels=labs) for _,g in o.groupby('source')])
rng=np.random.default_rng(42);boots=[]
for b in range(2000):
 cm=cms[rng.integers(0,len(cms),len(cms))].sum(0);den=cm.sum(0)+cm.sum(1);boots.append(np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean())
interval=np.quantile(boots,[.025,.975]).tolist()
for c in ['macro_f1_bootstrap_low95','macro_f1_bootstrap_high95','bootstrap_replicates']:
 t[c]=np.nan
t['bootstrap_unit']='';t['bootstrap_refit']=''
mask=t.stem.eq('main_six_source_LightGBM_s42');assert int(mask.sum())==1
t.loc[mask,'macro_f1_bootstrap_low95']=interval[0]
t.loc[mask,'macro_f1_bootstrap_high95']=interval[1]
t.loc[mask,'bootstrap_replicates']=2000
t.loc[mask,'bootstrap_unit']='source';t.loc[mask,'bootstrap_refit']='False'
t.drop(columns=['labels']).to_csv(OUT/'all_experiment_metrics.csv',index=False,encoding='utf-8-sig')
variance=pd.read_csv(OUT/'nested_variance_REML.csv');assert variance.converged.all()
folds=pd.read_csv(OUT/'OOF/main_six_source_LightGBM_s42_folds.csv');assert folds.missing_train.isna().all()
save={'oof_files_verified':len(audit),'primary_source_folds_all_classes_present':True,'variance_converged':True,'source_bootstrap_interval':interval,'verification':audit}
(OUT/'verify_summary.json').write_text(json.dumps(save,ensure_ascii=False,indent=2),encoding='utf-8')
print(summary.to_string(index=False));print('MODEL_TABLE',t[(t.cohort=='main')&(t.seed==42)&(t.task=='six')&~t.stem.str.contains('compat|nosmote|features')][['model','scheme','macro_f1','macro_roc_auc']].to_string(index=False));print('class',pd.DataFrame(perclass).to_string(index=False));print('audit',len(audit),'bootstrap',interval)
