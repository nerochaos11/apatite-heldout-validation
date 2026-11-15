from pathlib import Path
import csv,json,sys,shutil,os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,roc_curve,precision_recall_curve

sys.stdout.reconfigure(encoding='utf-8')
from paths import ROOT, OUT
P=ROOT;D=OUT;G=Path(os.environ.get('APATITE_FIGURE_DATA_DIR',ROOT/'recomputed/figure_data'))
E=['Th','U','Sr','Y','Nd','La'];LABS=['IOCG','Kiruna-type','Orogenic','Porphyry','Skarn','Unmineralized']
def read(name):return pd.read_csv(D/name,float_precision='round_trip')
def save(n,name,data):
    folder=G/f'Fig{n:02d}';folder.mkdir(exist_ok=True,parents=True)
    data.to_csv(folder/name,index=False,encoding='utf-8-sig')
def cp(n,name):
    folder=G/f'Fig{n:02d}';folder.mkdir(exist_ok=True,parents=True)
    shutil.copyfile(D/name,folder/Path(name).name)
def oof(task='six',scheme='sample',seed=42):return read(f'OOF/main_{task}_{scheme}_LightGBM_s{seed}.csv')
main=read('main_data_clean.csv');med=read('group_medians.csv');stats=read('composition_summary.csv')
audit=json.loads((D/'screening_audit.json').read_text(encoding='utf-8'))
metrics=read('all_experiment_metrics.csv')
main42=metrics[(metrics.cohort=='main')&(metrics.task=='six')&(metrics.seed==42)&~metrics.stem.str.contains('compat|nosmote|features')]
save(1,'screening_counts.csv',pd.DataFrame([{'stage':k,'n':audit[k]}for k in ['input_n','joint_duplicates','deduplicated_n','explicit_standard_n','duplicate_standard_overlap','main_n','main_samples','main_sources']]))
cp(1,'main_class_coverage.csv')
save(1,'source_class_counts.csv',main.groupby(['source','Dtype']).size().rename('n').reset_index())
save(2,'group_six_element_medians.csv',med[['Dtype','sample']+E]);save(2,'six_element_summary.csv',stats[stats.feature.isin(E)])
save(3,'group_ratios.csv',med[['Dtype','sample','Sr/Y','Th/U','La/Y']]);save(3,'ratio_summary.csv',stats[~stats.feature.isin(E)])
save(4,'models_and_units.csv',main42)
repeat=metrics[(metrics.cohort=='main')&(metrics.model=='LightGBM')&~metrics.stem.str.contains('compat|nosmote|features')]
save(5,'ten_seed_paired.csv',repeat[repeat.scheme.isin(['record','sample'])])
wide=repeat.pivot(index=['task','seed'],columns='scheme',values='macro_f1').reset_index();wide['record_minus_sample']=wide['record']-wide['sample'];save(5,'paired_differences.csv',wide)
for scheme in ['sample','source']:
    q=oof(scheme=scheme);cm=confusion_matrix(q.y_true,q.y_pred,labels=LABS);rows=[]
    for i,a in enumerate(LABS):
        for j,b in enumerate(LABS):rows.append({'true_class':a,'predicted_class':b,'n':int(cm[i,j]),'row_percent':100*cm[i,j]/cm[i].sum()})
    save(6,f'{scheme}_confusion.csv',pd.DataFrame(rows))
curves=[];grid=np.linspace(0,1,501)
for scheme in ['record','sample','source']:
    q=oof(scheme=scheme);roc=[];pr=[]
    for c in LABS:
        yy=(q.y_true==c).astype(int);a,b,_=roc_curve(yy,q['p_'+c]);roc.append(np.interp(grid,a,b));a,b,_=precision_recall_curve(yy,q['p_'+c]);pr.append(np.interp(grid,b[::-1],a[::-1]))
    curves.extend({'scheme':scheme,'grid':float(x),'macro_roc_tpr':float(y),'macro_pr_precision':float(z)}for x,y,z in zip(grid,np.mean(roc,0),np.mean(pr,0)))
save(7,'macro_roc_pr_curves.csv',pd.DataFrame(curves));save(7,'auc_ap_metrics.csv',main42[main42.model=='LightGBM'])
save(8,'mineralized_five_class_per_seed.csv',repeat[repeat.task=='five'])
for scheme in ['sample','source']:
    q=oof(task='five',scheme=scheme);labs=LABS[:-1];cm=confusion_matrix(q.y_true,q.y_pred,labels=labs)
    save(8,f'{scheme}_five_class_confusion.csv',pd.DataFrame([{'true_class':a,'predicted_class':b,'n':int(cm[i,j]),'row_percent':100*cm[i,j]/cm[i].sum()}for i,a in enumerate(labs)for j,b in enumerate(labs)]))
cp(9,'per_source_diagnostics.csv');cp(9,'main_source_coverage.csv')
for name in ['Mercer_case_sample_medians.csv','Mercer_case_sample_predictions.csv','Mercer_case_source_predictions.csv']:cp(10,name)
cp(11,'nested_variance_REML.csv');save(11,'group_record_counts.csv',main.groupby('sample').size().rename('analyses').reset_index())
save(12,'feature_and_resampling_branches.csv',metrics[(metrics.cohort=='main')&(metrics.model=='LightGBM')&metrics.stem.str.contains('features')]);cp(12,'scoring_target_comparison.csv')
cp(13,'SHAP_per_fold.csv');cp(13,'SHAP_per_class_per_fold.csv')
cal=json.loads((D/'calibration.json').read_text(encoding='utf-8'))
save(14,'reliability_bins.csv',pd.DataFrame([dict(stage=s,**r)for s in ['before','after']for r in cal[s+'_bins']]))
save(14,'calibration_metrics.csv',pd.DataFrame([dict(stage=s,**cal[s])for s in ['before','after']]))
save(14,'per_fold_temperature.csv',pd.DataFrame({'fold':range(1,len(cal['temperatures'])+1),'temperature':cal['temperatures']}))
mapdir=P/'06_model_and_map/interactive_map'
for name in ['map_example_inputs_and_probabilities.csv','coordinate_class_groups.csv']:
    folder=G/'Fig15';folder.mkdir(exist_ok=True);shutil.copyfile(mapdir/name,folder/name)
for name in ['application_Fosterville_predictions.csv','application_Bushveld_predictions.csv','application_Darongxi_predictions.csv']:cp(16,name)
hist=pd.read_csv(P/'02_source_data/historical/3522_six_elements_derived.csv',float_precision='round_trip')
save(17,'historical_vs_current_source_coverage.csv',pd.DataFrame({'class':LABS,'historical_sources':hist.groupby('Dtype').Citation.nunique().reindex(LABS,fill_value=0),'current_sources':main.groupby('Dtype').source.nunique().reindex(LABS,fill_value=0)}))
h=json.loads((P/'02_source_data/historical/3522_validation_results.json').read_text(encoding='utf-8'))
save(17,'historical_vs_current_macroF1.csv',pd.DataFrame([{'scheme':s,'historical_macro_f1':h['schemes'][old]['results']['LightGBM']['macro_f1'],'current_macro_f1':float(main42[(main42.model=='LightGBM')&(main42.scheme==s)].macro_f1.iloc[0])}for s,old in zip(['record','sample','source'],['record_stratified','group_sample','group_citation'])]))
save(17,'processing_branch_metrics.csv',metrics[(metrics.task=='six')&(metrics.model=='LightGBM')&(metrics.seed==42)&metrics.cohort.isin(['raw','dedup','main'])&~metrics.stem.str.contains('compat|nosmote|features')])
labels={1:'data screening, source structure and validation units',2:'six-element concentration distributions',3:'element ratio relationships',4:'five models and validation units',5:'ten-seed paired results',6:'six-class confusion matrices',7:'ROC and PR curves',8:'mineralized five-class',9:'source-wise recognition',10:'Pea Ridge geological units',11:'nested variance and replication',12:'features and scoring targets',13:'SHAP contributions',14:'probability calibration',15:'spatial index and example site',16:'application datasets',17:'data coverage and processing sensitivity'}
index=[]
for n,title in labels.items():
    for f in sorted((G/f'Fig{n:02d}').glob('*.csv')):index.append({'figure':f'Fig{n:02d}','subject':title,'data_file':f.relative_to(P).as_posix()})
pd.DataFrame(index).to_csv(G/'figure_data_index.csv',index=False,encoding='utf-8-sig')
print(json.dumps({'figure_data_files':len(index),'figure_numbers':len(labels),'table_files':len(list((G/'tables').glob('*.csv')))},ensure_ascii=False))
