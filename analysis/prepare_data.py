from pathlib import Path
import sys,json,hashlib,re,unicodedata
from collections import defaultdict,deque,Counter
# USGS Table S2 reports PR-163/A6/57 and /58 as separate analysis IDs with equal
# compositions. Preserve the two reported records; this does not establish
# physically independent measurements, samples, or geological generations.
import numpy as np,pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
from paths import ROOT, OUT, INPUT
OUT.mkdir(parents=True,exist_ok=True)
E=['Th','U','Sr','Y','Nd','La'];F=[e+'_log' for e in E]+['Sr_Y_ratio','Th_U_ratio','La_Y_ratio']
p=ROOT/'02_source_data/merged_input_6707.xlsx'
d=pd.read_excel(p);d['input_row']=np.arange(len(d))+2
for c in ['Citation','Sample_Name','Location','Deposit_Name','Dtype']:d[c]=d[c].astype(str).str.strip()
d['source']=d.Citation;d['sample']=d.Citation+' | '+d.Sample_Name
assert np.isfinite(d[E]).all().all() and (d[E]>0).all().all()
origin=d.copy();JOINT=['Citation','Sample_Name']+E
mp=INPUT/'Mercer2019_six_elements_231_negatives_kept.xlsx'
zp=INPUT/'Zheng2024_new_apatite_data.xlsx'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(zp)=='edd6467a1515d8748fff306e7757c42c8997a3acd80cc6ccb829adf7b14713a5'
m=pd.read_excel(mp,sheet_name='six-element data');m['source_excel_row']=np.arange(len(m))+2
me=[e+'(ppm)' for e in E]
positive=np.isfinite(m[me]).all(axis=1)&(m[me]>0).all(axis=1)
assert len(m)==231 and int(positive.sum())==187
assert not m.duplicated(['Sample_no','analysis_area','Analysis_no']).any()
m['canonical_label']=m['deposit_type'].replace({'Kiruna-type (IOA-REE)':'Kiruna-type'})
def key(sample,values):return(str(sample).strip(),*(float(v) for v in values))
def mkey(i):return key(m.at[i,'Sample_no'],m.loc[i,me])
def dkey(row):return key(row.Sample_Name,[row[e] for e in E])
oldm=pd.read_excel(ROOT/'02_source_data/Mercer2020_USGS_apatite_trace_elements.xlsx')
assert Counter(mkey(i) for i in m.index[positive])==Counter(key(r['Sample_Name'],[r[e+'\n(ppm)'] for e in E])for _,r in oldm.iterrows())
metadata=['analysis_area','analysis_no','geological_unit','rock_suite','analysis_location','source_file','source_sheet','source_excel_row','source_file_sha256','source_citation_raw','source_dataset_doi','reported_analysis_id']
for c in metadata:d[c]=pd.Series(['']*len(d),dtype=object)
d['restored_reported_analysis']=False;d['cloned_from_input_row']=pd.Series(['']*len(d),dtype=object)
available=defaultdict(deque)
for i in m.index[positive]:available[mkey(i)].append(int(i))
assigned={};existing_match={};clone_parent={}
def attach(idx,mi):
 r=m.loc[mi]
 assert d.at[idx,'Dtype']==r.canonical_label
 fields={'analysis_area':str(r['analysis_area']).strip(),'analysis_no':int(r.Analysis_no),'geological_unit':r['Unit'],'rock_suite':r['Suite'],'analysis_location':r['Location'],'source_file':mp.name,'source_sheet':'six-element data','source_excel_row':int(r.source_excel_row),'source_file_sha256':sha(mp),'source_citation_raw':r['source'],'source_dataset_doi':'10.5066/P9YIHMO8','reported_analysis_id':f"Mercer2020 | {str(r['Sample_no']).strip()} | {str(r['analysis_area']).strip()} | {int(r.Analysis_no)}"}
 for c,v in fields.items():d.at[idx,c]=v
legacy_indices=d.index[d.Citation=='Mercer2020'].tolist()
assert len(legacy_indices)==186
for idx in legacy_indices:
 k=dkey(d.loc[idx]);assert available[k],f'Unmatched Mercer input {idx+2}'
 mi=available[k].popleft();attach(idx,mi)
 assigned[mi]=int(d.at[idx,'input_row']);existing_match[mi]=True
remaining=sorted(i for q in available.values() for i in q)
assert len(remaining)==1
for mi in remaining:
 r=m.loc[mi];assert r['Sample_no']=='PR-163' and r['analysis_area']=='A6' and r.Analysis_no==58
 parents=[idx for idx in legacy_indices if dkey(d.loc[idx])==mkey(mi)]
 assert len(parents)==1
 parent=parents[0];new=d.loc[parent].copy();new['input_row']=int(d.input_row.max())+1
 new['restored_reported_analysis']=True;new['cloned_from_input_row']=int(d.at[parent,'input_row'])
 d=pd.concat([d,pd.DataFrame([new])],ignore_index=True);idx=len(d)-1;attach(idx,mi)
 assigned[mi]=int(d.at[idx,'input_row']);existing_match[mi]=False;clone_parent[mi]=int(d.at[parent,'input_row'])
for e in E:d[e]=pd.to_numeric(d[e],errors='raise')
d['input_row']=d.input_row.astype(int);d['restored_reported_analysis']=d.restored_reported_analysis.astype(bool)
pd.testing.assert_frame_equal(d.iloc[:6707][origin.columns].reset_index(drop=True),origin.reset_index(drop=True),check_dtype=False,check_exact=True)
assert d.input_row.tolist()==list(range(2,6710))
strict_dup=d.duplicated(JOINT);protected=pd.Series(False,index=d.index)
for _,g in d.groupby(JOINT,sort=False,dropna=False):
 known=g[g.reported_analysis_id.ne('')]
 if known.reported_analysis_id.nunique()>1:
  ix=known.drop_duplicates('reported_analysis_id').index;protected.loc[ix]=strict_dup.loc[ix]
dup=strict_dup&~protected
assert (int(strict_dup.sum()),int(protected.sum()),int(dup.sum()))==(50,1,49)
std=d.Sample_Name.str.contains(r'standard',case=False,regex=True)
d['excluded_duplicate']=dup;d['excluded_reference_standard']=std
d['strict_duplicate']=strict_dup;d['protected_distinct_analysis']=protected
d['exclusion_reason']=np.select([dup,std],['identical source, sample name and six elements','explicitly labelled analytical reference material'],default='keep')
d.to_csv(OUT/'per_record_screening.csv',index=False,encoding='utf-8-sig')
ma=m.copy();ma['canonical_source']='Mercer2020';ma['finite_positive_six']=positive
ma['candidate_input_row']=[assigned.get(int(i),'')for i in m.index]
ma['existing_occurrence_match']=[existing_match.get(int(i),False)for i in m.index]
oldkeys={dkey(origin.loc[i])for i in origin.index[origin.Citation=='Mercer2020']}
ma['existing_vector_match']=[mkey(i)in oldkeys for i in m.index]
ma['restored_reported_analysis']=[int(i)in remaining for i in m.index]
ma['cloned_from_input_row']=[clone_parent.get(int(i),'')for i in m.index]
ma['reported_analysis_id']=[f"Mercer2020 | {str(m.at[i,'Sample_no']).strip()} | {str(m.at[i,'analysis_area']).strip()} | {int(m.at[i,'Analysis_no'])}" for i in m.index]
ma['source_file']=mp.name;ma['source_sheet']='six-element data';ma['source_file_sha256']=sha(mp)
ma['negative_elements']=[','.join(e for e,c in zip(E,me)if m.at[i,c]<0)for i in m.index]
ma['zero_elements']=[','.join(e for e,c in zip(E,me)if m.at[i,c]==0)for i in m.index]
ma['disposition']=np.select([~positive,ma.restored_reported_analysis],['non-positive; excluded from log models','restored reported analysis with a distinct identifier'],default='composition row already in 6707; analysis metadata added')
ma.to_csv(OUT/'Mercer231_row_inclusion_audit.csv',index=False,encoding='utf-8-sig')
ma.loc[~positive].to_csv(OUT/'Mercer231_nonpositive_excluded_44.csv',index=False,encoding='utf-8-sig')
ma.loc[ma.restored_reported_analysis].to_csv(OUT/'Mercer231_restored_spots.csv',index=False,encoding='utf-8-sig')
def build(x,name):
 x=x.copy().reset_index(drop=True)
 for e in E:x[e+'_log']=np.log10(x[e].astype(float))
 for name_,a,b in [('Sr_Y_ratio','Sr','Y'),('Th_U_ratio','Th','U'),('La_Y_ratio','La','Y')]:x[name_]=x[a+'_log']-x[b+'_log']
 # A deliberately conservative grouping: literal normalized deposit names, not inferred new deposits.
 def norm(v):return re.sub(r'[^a-z0-9\u4e00-\u9fff]+','',unicodedata.normalize('NFKC',v).lower())
 x['deposit_group']=x.Deposit_Name.map(norm)
 x.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
 return x
historical=origin.copy();old_dup=historical.duplicated(JOINT);old_std=historical.Sample_Name.str.contains(r'standard',case=False,regex=True)
historical['excluded_duplicate']=old_dup;historical['excluded_reference_standard']=old_std
historical['exclusion_reason']=np.select([old_dup,old_std],['identical source, sample name and six elements','explicitly labelled analytical reference material'],default='keep')
historical=build(historical,'historical_reference_6707')
raw=build(d,'candidates_merged');ded=build(d[~dup],'sensitivity_deduplication');main=build(d[~dup & ~std],'main_data_clean')
assert (len(raw),len(ded),len(main),main['sample'].nunique(),main.source.nunique())==(6708,6659,6502,4223,90)
case=main[main.Citation=='Mercer2020'];assert len(case)==187 and case.reported_analysis_id.nunique()==187
assert set(case.loc[case.Sample_Name=='PR-163','analysis_no'])>={57,58}
assert main.groupby('sample').Dtype.nunique().max()==1
cover=main.groupby('Dtype').agg(records=('Dtype','size'),samples=('sample','nunique'),sources=('source','nunique'),deposit_text_groups=('deposit_group','nunique'))
cover.to_csv(OUT/'main_class_coverage.csv',encoding='utf-8-sig')
sources=main.groupby('source').agg(records=('Dtype','size'),samples=('sample','nunique'),classes=('Dtype','nunique'))
sources.to_csv(OUT/'main_source_coverage.csv',encoding='utf-8-sig')
med=main.groupby(['Dtype','sample'])[E].median().reset_index()
for k,a,b in [('Sr/Y','Sr','Y'),('Th/U','Th','U'),('La/Y','La','Y')]:med[k]=med[a]/med[b]
med.to_csv(OUT/'group_medians.csv',index=False,encoding='utf-8-sig')
stats=[]
for cls,g in med.groupby('Dtype'):
 for k in E+['Sr/Y','Th/U','La/Y']:
  stats.append(dict(cls=cls,feature=k,n=len(g),q1=g[k].quantile(.25),median=g[k].median(),q3=g[k].quantile(.75),minimum=g[k].min(),maximum=g[k].max()))
pd.DataFrame(stats).to_csv(OUT/'composition_summary.csv',index=False,encoding='utf-8-sig')
gc=main.groupby('sample').size();old=pd.read_csv(ROOT/'02_source_data/historical/3522_six_elements_derived.csv')
audit={'input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'input_n':len(d),'input_samples':d['sample'].nunique(),'joint_duplicates':int(dup.sum()),'explicit_standard_n':int(std.sum()),'duplicate_standard_overlap':int((dup&std).sum()),'deduplicated_n':len(ded),'main_n':len(main),'main_samples':main['sample'].nunique(),'main_sources':main.source.nunique(),'class_coverage':cover.reset_index().to_dict('records'),'mineral_n':int((main.Dtype!='Unmineralized').sum()),'mineral_sources':main[main.Dtype!='Unmineralized'].source.nunique(),'mineral_samples':main[main.Dtype!='Unmineralized']['sample'].nunique(),'mineral_deposit_text_groups':main[main.Dtype!='Unmineralized'].deposit_group.nunique(),'singleton_samples':int((gc==1).sum()),'multiple_samples':int((gc>1).sum()),'multiple_sample_records':int(gc[gc>1].sum()),'max_sample_size':int(gc.max()),'single_class_sources':int((sources.classes==1).sum()),'single_class_source_records':int(sources.loc[sources.classes==1,'records'].sum()),'added_sources':sorted(set(main.Citation)-set(old.Citation)),'top5_records':int(sources.records.nlargest(5).sum()),'main_sha256':hashlib.sha256((OUT/'main_data_clean.csv').read_bytes()).hexdigest()}
audit.update({'input_origin_n':len(origin),'input_origin_joint_duplicates':int(old_dup.sum()),'strict_duplicate_count':int(strict_dup.sum()),'restored_reported_analyses':len(remaining),'protected_distinct_analyses':int(protected.sum()),'mercer_raw_n':len(m),'mercer_positive_n':int(positive.sum()),'mercer_nonpositive_n':int((~positive).sum()),'mercer_matched_existing_occurrences':sum(existing_match.values()),'mercer_sha256':sha(mp),'reuploaded_zheng_sha256':sha(zp),'reuploaded_zheng_identical':True,'original_6707_order_and_values_preserved':True,'restored_input_rows':d.loc[d.restored_reported_analysis,'input_row'].tolist(),'historical_raw_file':'historical_reference_6707.csv','historical_raw_sha256':sha(OUT/'historical_reference_6707.csv'),'raw_file':'candidates_merged.csv','deduplicated_file':'sensitivity_deduplication.csv'})
(OUT/'screening_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False,indent=2))
