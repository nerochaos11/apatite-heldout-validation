from pathlib import Path
import csv,json,sys,shutil,os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,roc_curve,precision_recall_curve

sys.stdout.reconfigure(encoding='utf-8')
from paths import ROOT, OUT
P=ROOT;D=OUT;G=Path(os.environ.get('APATITE_FIGURE_DATA_DIR',ROOT/'再计算/图表数据'))
E=['Th','U','Sr','Y','Nd','La'];LABS=['IOCG','Kiruna-type','Orogenic','Porphyry','Skarn','Unmineralized']
def read(name):return pd.read_csv(D/name,float_precision='round_trip')
def save(n,name,data):
    folder=G/f'Fig{n:02d}';folder.mkdir(exist_ok=True,parents=True)
    data.to_csv(folder/name,index=False,encoding='utf-8-sig')
def cp(n,name):
    folder=G/f'Fig{n:02d}';folder.mkdir(exist_ok=True,parents=True)
    shutil.copyfile(D/name,folder/Path(name).name)
def oof(task='six',scheme='sample',seed=42):return read(f'OOF/main_{task}_{scheme}_LightGBM_s{seed}.csv')
main=read('主数据_清洗后.csv');med=read('样品组中位数.csv');stats=read('组成描述统计.csv')
audit=json.loads((D/'清洗审计.json').read_text(encoding='utf-8'))
metrics=read('实验指标总表.csv')
main42=metrics[(metrics.cohort=='main')&(metrics.task=='six')&(metrics.seed==42)&~metrics.stem.str.contains('compat|nosmote|features')]
save(1,'筛选计数.csv',pd.DataFrame([{'stage':k,'n':audit[k]}for k in ['input_n','joint_duplicates','deduplicated_n','explicit_standard_n','duplicate_standard_overlap','main_n','main_samples','main_sources']]))
cp(1,'主数据_类别覆盖.csv')
save(1,'来源类别记录数.csv',main.groupby(['source','Dtype']).size().rename('n').reset_index())
save(2,'样品组六元素中位数.csv',med[['Dtype','sample']+E]);save(2,'六元素描述统计.csv',stats[stats.feature.isin(E)])
save(3,'样品组比值.csv',med[['Dtype','sample','Sr/Y','Th/U','La/Y']]);save(3,'比值描述统计.csv',stats[~stats.feature.isin(E)])
save(4,'模型与验证单元.csv',main42)
repeat=metrics[(metrics.cohort=='main')&(metrics.model=='LightGBM')&~metrics.stem.str.contains('compat|nosmote|features')]
save(5,'十种子配对.csv',repeat[repeat.scheme.isin(['record','sample'])])
wide=repeat.pivot(index=['task','seed'],columns='scheme',values='macro_f1').reset_index();wide['record_minus_sample']=wide['record']-wide['sample'];save(5,'配对差.csv',wide)
for scheme in ['sample','source']:
    q=oof(scheme=scheme);cm=confusion_matrix(q.y_true,q.y_pred,labels=LABS);rows=[]
    for i,a in enumerate(LABS):
        for j,b in enumerate(LABS):rows.append({'true_class':a,'predicted_class':b,'n':int(cm[i,j]),'row_percent':100*cm[i,j]/cm[i].sum()})
    save(6,f'{scheme}_混淆矩阵.csv',pd.DataFrame(rows))
curves=[];grid=np.linspace(0,1,501)
for scheme in ['record','sample','source']:
    q=oof(scheme=scheme);roc=[];pr=[]
    for c in LABS:
        yy=(q.y_true==c).astype(int);a,b,_=roc_curve(yy,q['p_'+c]);roc.append(np.interp(grid,a,b));a,b,_=precision_recall_curve(yy,q['p_'+c]);pr.append(np.interp(grid,b[::-1],a[::-1]))
    curves.extend({'scheme':scheme,'grid':float(x),'macro_roc_tpr':float(y),'macro_pr_precision':float(z)}for x,y,z in zip(grid,np.mean(roc,0),np.mean(pr,0)))
save(7,'宏平均ROC_PR绘图坐标.csv',pd.DataFrame(curves));save(7,'AUC_AP指标.csv',main42[main42.model=='LightGBM'])
save(8,'矿化五分类_逐种子指标.csv',repeat[repeat.task=='five'])
for scheme in ['sample','source']:
    q=oof(task='five',scheme=scheme);labs=LABS[:-1];cm=confusion_matrix(q.y_true,q.y_pred,labels=labs)
    save(8,f'{scheme}_五分类混淆矩阵.csv',pd.DataFrame([{'true_class':a,'predicted_class':b,'n':int(cm[i,j]),'row_percent':100*cm[i,j]/cm[i].sum()}for i,a in enumerate(labs)for j,b in enumerate(labs)]))
cp(9,'逐来源逐类别诊断.csv');cp(9,'主数据_来源覆盖.csv')
for name in ['Mercer案例_样品中位数.csv','Mercer案例_sample_预测.csv','Mercer案例_source_预测.csv']:cp(10,name)
cp(11,'嵌套方差_REML.csv');save(11,'登记样品组记录数.csv',main.groupby('sample').size().rename('analyses').reset_index())
save(12,'特征与重采样分支.csv',metrics[(metrics.cohort=='main')&(metrics.model=='LightGBM')&metrics.stem.str.contains('features')]);cp(12,'评分对象比较.csv')
cp(13,'SHAP逐折.csv');cp(13,'SHAP逐类别逐折.csv')
cal=json.loads((D/'校准.json').read_text(encoding='utf-8'))
save(14,'可靠性分箱.csv',pd.DataFrame([dict(stage=s,**r)for s in ['before','after']for r in cal[s+'_bins']]))
save(14,'校准指标.csv',pd.DataFrame([dict(stage=s,**cal[s])for s in ['before','after']]))
save(14,'折内温度.csv',pd.DataFrame({'fold':range(1,len(cal['temperatures'])+1),'temperature':cal['temperatures']}))
mapdir=P/'06_模型与地图/交互地图'
for name in ['地图示例点_输入与概率.csv','坐标类别分组.csv']:
    folder=G/'Fig15';folder.mkdir(exist_ok=True);shutil.copyfile(mapdir/name,folder/name)
for name in ['应用资料_Fosterville_预测.csv','应用资料_Bushveld_预测.csv','应用资料_Darongxi_预测.csv']:cp(16,name)
hist=pd.read_csv(P/'02_原始数据与来源/历史比较/3522条_六元素与派生特征.csv',float_precision='round_trip')
save(17,'历史与当前来源覆盖.csv',pd.DataFrame({'class':LABS,'historical_sources':hist.groupby('Dtype').Citation.nunique().reindex(LABS,fill_value=0),'current_sources':main.groupby('Dtype').source.nunique().reindex(LABS,fill_value=0)}))
h=json.loads((P/'02_原始数据与来源/历史比较/3522条_验证结果.json').read_text(encoding='utf-8'))
save(17,'历史与当前MacroF1.csv',pd.DataFrame([{'scheme':s,'historical_macro_f1':h['schemes'][old]['results']['LightGBM']['macro_f1'],'current_macro_f1':float(main42[(main42.model=='LightGBM')&(main42.scheme==s)].macro_f1.iloc[0])}for s,old in zip(['record','sample','source'],['record_stratified','group_sample','group_citation'])]))
save(17,'处理分支指标.csv',metrics[(metrics.task=='six')&(metrics.model=='LightGBM')&(metrics.seed==42)&metrics.cohort.isin(['raw','dedup','main'])&~metrics.stem.str.contains('compat|nosmote|features')])
labels={1:'资料筛选、来源结构与验证单元',2:'六元素浓度分布',3:'元素比值关系',4:'五种模型及验证单元',5:'十种子配对',6:'六分类混淆矩阵',7:'ROC与PR曲线',8:'矿化五分类',9:'逐来源识别',10:'Pea Ridge地质单元',11:'嵌套方差与重复分析',12:'特征与评分对象',13:'SHAP贡献',14:'概率校准',15:'空间索引与例点',16:'应用资料',17:'数据覆盖与处理敏感性'}
index=[]
for n,title in labels.items():
    for f in sorted((G/f'Fig{n:02d}').glob('*.csv')):index.append({'figure':f'Fig{n:02d}','subject':title,'data_file':f.relative_to(P).as_posix()})
pd.DataFrame(index).to_csv(G/'图件数据索引.csv',index=False,encoding='utf-8-sig')
print(json.dumps({'figure_data_files':len(index),'figure_numbers':len(labels),'table_files':len(list((G/'表格').glob('*.csv')))},ensure_ascii=False))
