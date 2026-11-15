from pathlib import Path
import argparse,ast,csv,hashlib,json,math,os,subprocess,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
D=ROOT/'01_data_and_results'
M=ROOT/'06_model_and_map'
W=ROOT/'04_analysis_code'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
 with p.open(encoding='utf-8-sig',newline='')as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ranking(scores,truth):
 npos=sum(truth);nneg=len(truth)-npos
 assert npos>0 and nneg>0
 order=sorted(zip(scores,truth));i=0;rank_sum=0.
 while i<len(order):
  j=i+1
  while j<len(order)and order[j][0]==order[i][0]:j+=1
  rank_sum+=sum(v for _,v in order[i:j])*(i+1+j)/2;i=j
 auc=(rank_sum-npos*(npos+1)/2)/(npos*nneg)
 order.reverse();i=0;tp=0;seen=0;ap=0.
 while i<len(order):
  j=i+1
  while j<len(order)and order[j][0]==order[i][0]:j+=1
  add=sum(v for _,v in order[i:j]);tp+=add;seen+=j-i;ap+=(add/npos)*(tp/seen);i=j
 return auc,ap
def evaluate(rs,labels,prefix='p_'):
 k=len(labels);where={c:i for i,c in enumerate(labels)};cm=[[0]*k for _ in labels];pp=[];yy=[];loss=0.;brier=0.
 for r in rs:
  p=[float(r[prefix+c])for c in labels]
  assert all(math.isfinite(v)and -1e-12<=v<=1+1e-12 for v in p)and abs(sum(p)-1)<1e-7
  yi=where[r['y_true']];pi=max(range(k),key=lambda j:p[j]);cm[yi][pi]+=1
  if prefix=='p_':assert r['y_pred']==labels[pi]
  loss-=math.log(max(min(p[yi],1-sys.float_info.epsilon),sys.float_info.epsilon));brier+=sum((v-(j==yi))**2 for j,v in enumerate(p));pp.append(p);yy.append(yi)
 n=len(rs);precision=[];recall=[];f1=[];aucs=[];aps=[]
 for j in range(k):
  tp=cm[j][j];actual=sum(cm[j]);predicted=sum(row[j]for row in cm)
  precision.append(tp/predicted if predicted else 0.);recall.append(tp/actual if actual else 0.);f1.append(2*tp/(actual+predicted)if actual+predicted else 0.)
  a,b=ranking([p[j]for p in pp],[int(v==j)for v in yy]);aucs.append(a);aps.append(b)
 return dict(accuracy=sum(cm[j][j]for j in range(k))/n,balanced_accuracy=sum(recall)/k,macro_precision=sum(precision)/k,macro_f1=sum(f1)/k,macro_roc_auc=sum(aucs)/k,macro_ap=sum(aps)/k,log_loss=loss/n,mean_brier=brier/(n*k))
def verify():
 manifest={r['file']:{'bytes':int(r['bytes']),'sha256':r['sha256']}for r in rows(ROOT/'file_checksums.csv')};version=read(ROOT/'version_info.json')
 for rel,entry in manifest.items():
  p=ROOT/rel
  assert p.is_file()and p.stat().st_size==entry['bytes']and sha(p)==entry['sha256'],rel
  if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'),filename=rel)
 data=rows(D/'main_data_clean.csv');assert(len(data),len({r['sample']for r in data}),len({r['source']for r in data}))==(6502,4223,90)
 assert sha(D/'main_data_clean.csv')==version['main_data_sha256']
 assert len({r['input_row']for r in data})==6502
 assert all(float(r[e])>0 and math.isfinite(float(r[e]))for r in data for e in ['Th','U','Sr','Y','Nd','La'])
 assert sum(r['restored_reported_analysis'].lower()=='true'for r in data)==1
 assert len(rows(D/'Mercer231_row_inclusion_audit.csv'))==231 and len(rows(D/'Mercer231_nonpositive_excluded_44.csv'))==44
 all_metrics=sorted((D/'OOF').glob('*_metrics.json'));assert len(all_metrics)==106
 core_sha=version['analysis_module_sha256_at_run']
 for i,p in enumerate(all_metrics,1):
  m=read(p);rs=rows(p.with_name(p.name.replace('_metrics.json','.csv')));assert len(rs)==m['n']
  assert m['run_config']['core_sha256']==core_sha,p.name
  folds=rows(p.with_name(p.name.replace('_metrics.json','_folds.csv')))
  assert len(folds)==5 and {int(r['fold'])for r in rs}=={1,2,3,4,5}
  assert len({r['input_row']for r in rs})==len(rs)
  scheme=m['scheme']
  if scheme!='record':
   key={'sample':'sample','source':'source','deposit':'deposit_group'}[scheme]
   group_fold={}
   for r in rs:
    group_fold.setdefault(r[key],r['fold']);assert group_fold[r[key]]==r['fold'],(p.name,key,r[key])
  for f in folds:
   assert int(f['n_train'])+int(f['n_test'])==len(rs)
   assert sum(int(r['fold'])==int(f['fold'])for r in rs)==int(f['n_test'])
  inner=p.with_name(p.name.replace('_metrics.json','_inner_folds.csv'))
  if inner.exists():assert all(int(r['group_overlap'])==0 for r in rows(inner))
  got=evaluate(rs,m['labels'])
  for key,value in got.items():assert math.isfinite(float(m[key]))and abs(value-float(m[key]))<1e-6,(p.name,key,value,m[key])
  if i%20==0:print('OOF checked',i,flush=True)
 loso=sorted((D/'LOSO').glob('*.csv'));assert len(loso)==90
 got_ids=[];sources=set()
 for p in loso:
  rs=rows(p);ss={r['source']for r in rs};assert len(ss)==1 and not sources&ss;sources|=ss;got_ids.extend(r['input_row']for r in rs)
  labels=sorted(c[2:]for c in rs[0]if c.startswith('p_'))
  for r in rs:
   probs=[float(r['p_'+c])for c in labels];assert abs(sum(probs)-1)<1e-7 and all(math.isfinite(v)and 0<=v<=1 for v in probs);assert r['y_pred']==labels[max(range(len(labels)),key=lambda j:probs[j])]
 assert len(got_ids)==6502 and set(got_ids)=={r['input_row']for r in data}
 cal=read(D/'calibration.json');calrows=rows(D/'calibration_per_record.csv');labels=sorted({r['Dtype']for r in data});assert len(calrows)==6502
 for prefix,name in [('raw_','before'),('cal_','after')]:
  got=evaluate(calrows,labels,prefix)
  for key,val in got.items():assert abs(val-float(cal[name][key]))<1e-6,('calibration',name,key)
  assert sum(int(r['n'])for r in cal[name+'_bins'])==6502
 assert len(rows(D/'nested_variance_REML.csv'))==6
 for r in rows(D/'nested_variance_REML.csv'):
  assert r['converged'].lower()=='true';assert abs(sum(float(r[k])for k in ['source_fraction','sample_fraction','residual_fraction'])-1)<1e-7
 mapmeta=read(M/'interactive_map/map_data_audit.json');assert mapmeta['training_sha256']==version['main_data_sha256']and mapmeta['training_n']==6502
 assert sha(M/'LightGBM_main_full_refit.joblib')==version['model_sha256']
 interval=[r for r in rows(D/'all_experiment_metrics.csv')if r['stem']=='main_six_source_LightGBM_s42'];assert len(interval)==1
 interval=interval[0];assert int(interval['bootstrap_replicates'])==2000 and interval['bootstrap_unit']=='source'and interval['bootstrap_refit']=='False'
 assert 0<float(interval['macro_f1_bootstrap_low95'])<float(interval['macro_f1_bootstrap_high95'])<1
 print(json.dumps(dict(files=len(manifest),main_n=6502,samples=4223,sources=90,oof=106,loso=90,all_saved_metrics_verified=True,default_performs_no_training=True),ensure_ascii=False),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description='verify data, or recompute in a separate directory.')
 p.add_argument('--retrain',action='store_true',help='reprepare the data and run all analyses; output goes to recomputed/')
 p.add_argument('--redraw-statistics',action='store_true',help='redraw statistical figures from saved results')
 p.add_argument('--redraw-map',action='store_true',help='redraw the map figure from the bundled view and figure data')
 a=p.parse_args();verify()
 env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONIOENCODING']='utf-8'
 if a.retrain:
  base=ROOT/'recomputed';base.mkdir(exist_ok=True)
  env['APATITE_FORCE_RETRAIN']='1'
  env['APATITE_RESULTS_DIR']=str(base/'data_and_results')
  env['APATITE_APPLICATION_DIR']=str(base/'model_and_map')
  env['APATITE_FIGURE_DIR']=str(base/'figures')
  for script in ['prepare_data.py','run_all.py']:
   subprocess.run([sys.executable,str(W/script)],cwd=ROOT,env=env,check=True)
 if a.redraw_statistics:
  subprocess.run([sys.executable,str(W/'figures.py')],cwd=ROOT,env=env,check=True)
 if a.redraw_map:
  subprocess.run([sys.executable,str(W/'map_figure.py')],cwd=ROOT,env=env,check=True)
