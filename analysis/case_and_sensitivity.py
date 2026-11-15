from experiment_core import *
from scipy.spatial.distance import cdist
d=load()
case=d[d.Citation=='Mercer2020'].copy();case['Unit']=case.geological_unit
assert case.Unit.notna().all()
assert case.reported_analysis_id.notna().all() and case.reported_analysis_id.is_unique
assert case.groupby('Sample_Name').Unit.nunique().max()==1
case['case_unit']=case.Unit.where(case.Dtype=='Kiruna-type','Regional igneous')
case.to_csv(OUT/'Mercer_case_records_and_units.csv',index=False,encoding='utf-8-sig')
med=case.groupby(['Sample_Name','Dtype','case_unit'])[E].median().reset_index()
for k,x,z in [('Th/U','Th','U'),('Sr/Y','Sr','Y'),('La/Y','La','Y')]:med[k]=med[x]/med[z]
med.to_csv(OUT/'Mercer_case_sample_medians.csv',index=False,encoding='utf-8-sig')
for scheme in ['sample','source']:
 o=pd.read_csv(OUT/f'OOF/main_six_{scheme}_LightGBM_s42.csv',float_precision='round_trip')
 q=case.merge(o.drop(columns=['source','sample','deposit_group']),on='input_row',validate='one_to_one')
 q.to_csv(OUT/f'Mercer_case_{scheme}_predictions.csv',index=False,encoding='utf-8-sig')
print('MERCER',len(case),len(med),case.groupby(['Dtype','case_unit']).size().to_dict(),flush=True)
# The submitted cohort excludes Zirner2015. Test inclusion under the compilation's inherited label,
# rather than treating high REE concentrations as sufficient evidence of a label error.
z=pd.read_excel(ROOT/'02_source_data/Zheng2024_new_apatite_data.xlsx',sheet_name=0)
z=z[z.Source_Key=='Zirner2015'].copy()
z=z.rename(columns={**{e+'(ppm)':e for e in E},'Category':'Dtype'})
z['Citation']=z.Source_Key;z['source']=z.Source_Key;z['sample']=z.source+' | '+z.Sample_Name.astype(str).str.strip();z['input_row']=100000+np.arange(len(z))
z['Location']=z.Country;z['deposit_group']=z.Deposit_Name
z['excluded_duplicate']=False;z['excluded_reference_standard']=False;z['exclusion_reason']='included in sensitivity branch with original label'
for e in E:z[e+'_log']=np.log10(z[e].astype(float))
for key,x,y in [('Sr_Y_ratio','Sr','Y'),('Th_U_ratio','Th','U'),('La_Y_ratio','La','Y')]:z[key]=z[x+'_log']-z[y+'_log']
zz=pd.concat([d,z.reindex(columns=d.columns)],ignore_index=True)
zz.to_csv(OUT/'sensitivity_with_Zirner2015.csv',index=False,encoding='utf-8-sig')
for scheme in ['sample','source']:run(zz,scheme,cohort='include_Zirner')
# Join whole registered sample groups connected by exactly identical vectors.
# This preserves the original sample grouping, including its non-identical analyses.
sens=d.copy();sig=pd.util.hash_pandas_object(sens[E],index=False)
mask=sig.duplicated(keep=False);parent={s:s for s in sens['sample'].unique()}
def find(s):
 while parent[s]!=s:parent[s]=parent[parent[s]];s=parent[s]
 return s
for _,g in sens[mask].groupby(sig[mask]):
 names=sorted(g['sample'].unique())
 for s in names[1:]:parent[find(s)]=find(names[0])
sens['sample']=sens['sample'].map(find)
mapping=d[['input_row','sample']].copy();mapping['sensitivity_group']=sens['sample']
mapping.to_csv(OUT/'sensitivity_linked_groups.csv',index=False,encoding='utf-8-sig')
run(sens,'sample',cohort='shared_vector_grouped')
