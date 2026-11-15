"""Publication figures; all graphical marks are computed from saved observations or OOF predictions."""
from experiment_core import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def read_numeric_csv(*args,**kwargs):
 kwargs.setdefault('float_precision','round_trip')
 return pd.read_csv(*args,**kwargs)
from matplotlib.colors import LogNorm,Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch,Rectangle,FancyBboxPatch
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve,precision_recall_curve,confusion_matrix
from matplotlib.gridspec import GridSpec
from PIL import Image
WORK=Path(__file__).parent;FIG=FIGURE_DIR;FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,'axes.linewidth':.65,'xtick.labelsize':7,'ytick.labelsize':7,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,'lines.linewidth':1.1,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
COL=['#B75E61','#D6AD51','#70A47A','#729DC1','#A48BB5','#79B9B5'];SHORT=['IOCG','Kiruna','Orogenic','Porphyry','Skarn','Unmin.'];SC=['record','sample','source'];SNAME=['Record','Sample','Source'];SCOL=['#3B3B3B','#4C86A1','#B56657'];MARK=['o','^','s','D','P','X']
NUMBER_MAP={1:1,2:2,3:3,4:4,5:5,6:6,7:7,13:8,12:9,17:10,14:11,8:12,10:13,9:14,15:15,11:16,16:17}
d=load();med=read_numeric_csv(OUT/'group_medians.csv');audit=json.loads((OUT/'screening_audit.json').read_text(encoding='utf-8'))
allmetrics=read_numeric_csv(OUT/'all_experiment_metrics.csv');rep=read_numeric_csv(OUT/'ten_seed_summary.csv')
def m(task='six',scheme='sample',name='LightGBM',seed=42,cohort='main',suffix=''):
 return json.loads((OUT/f'OOF/{cohort}_{task}_{scheme}_{name}_s{seed}{suffix}_metrics.json').read_text(encoding='utf-8'))
def oof(task='six',scheme='sample',name='LightGBM',seed=42):return read_numeric_csv(OUT/f'OOF/main_{task}_{scheme}_{name}_s{seed}.csv')
def title(ax,s):ax.set_title(s,loc='left',pad=7)
def save(fig,n):
 new=NUMBER_MAP[n]
 fig.savefig(FIG/f'Fig{new:02d}.pdf',bbox_inches='tight',pad_inches=.06)
 fig.savefig(FIG/f'Fig{new:02d}.png',dpi=400,bbox_inches='tight',pad_inches=.06)
 plt.close(fig);print(f'FIGURE old={n} new={new}',flush=True)
def violin(ax,field,lab):
 vals=[np.log10(med.loc[med.Dtype==c,field]) for c in LABELS]
 parts=ax.violinplot(vals,positions=np.arange(6),widths=.78,showextrema=False,bw_method=.25)
 for p,c in zip(parts['bodies'],COL):p.set_facecolor(c);p.set_edgecolor('#333333');p.set_alpha(.55);p.set_linewidth(.45)
 rng=np.random.default_rng(42)
 for j,v in enumerate(vals):
  ax.scatter(j+rng.uniform(-.18,.18,len(v)),v,s=1.1,color=COL[j],alpha=.15,rasterized=True,linewidths=0)
  q=np.quantile(v,[.25,.5,.75]);ax.plot([j,j],[q[0],q[2]],color='#333333',lw=2);ax.scatter(j,q[1],s=9,c='white',edgecolor='#333333',lw=.5,zorder=3)
 ax.set_xticks(range(6),SHORT,rotation=25,ha='right');ax.set_ylabel(lab)
 lo,hi=ax.get_ylim();ticks=np.arange(np.ceil(lo),np.floor(hi)+1);ax.set_yticks(ticks,[r'$10^{%d}$'%i for i in ticks]);ax.grid(axis='y',lw=.3,color='#dddddd',zorder=0)
def confusion(ax,o,labs,title_):
 cm=confusion_matrix(o.y_true,o.y_pred,labels=labs);v=cm/cm.sum(1,keepdims=True)*100;im=ax.imshow(v,cmap='Blues',vmin=0,vmax=100)
 ax.set_xticks(range(len(labs)),SHORT[:len(labs)],rotation=35,ha='right');ax.set_yticks(range(len(labs)),[f'{s} (n={n:,})' for s,n in zip(SHORT[:len(labs)],cm.sum(1))]);ax.set_xlabel('Predicted class');ax.set_ylabel('Compilation label');title(ax,title_)
 for i in range(len(labs)):
  for j in range(len(labs)):ax.text(j,i,f'{v[i,j]:.1f}',ha='center',va='center',fontsize=6.7,color='white' if v[i,j]>55 else '#222222')
 return im
def figure1():
 fig,axs=plt.subplots(2,2,figsize=(6.69,4.8),layout='constrained',gridspec_kw={'height_ratios':[.95,1.05]})
 ax=axs[0,0];ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');title(ax,'(a) Data screening')
 raw_n=len(load('raw'));dedup_n=len(load('dedup'));main_n=len(d)
 assert [raw_n,dedup_n,main_n]==[int(audit['input_n']),int(audit['deduplicated_n']),int(audit['main_n'])]
 for yy,label,count,fill in [(.81,'Merged candidates',raw_n,'#EDF1F3'),(.49,'Jointly deduplicated',dedup_n,'#DAE5EA'),(.17,'Main compilation',main_n,'#B3CBD5')]:
  ax.add_patch(FancyBboxPatch((.04,yy-.095),.78,.19,boxstyle='round,pad=.012,rounding_size=.015',facecolor=fill,edgecolor='#555555',lw=.6));ax.text(.075,yy,label,ha='left',va='center',fontsize=7.4);ax.text(.78,yy,f'{count:,}',ha='right',va='center',fontsize=8)
 for y1,y2,note in [(.71,.60,f'{raw_n-dedup_n:,} duplicate records removed'),(.39,.28,f'{dedup_n-main_n:,} reference-material analyses')]:
  ax.annotate('',xy=(.43,y2),xytext=(.43,y1),arrowprops=dict(arrowstyle='-|>',lw=.65,color='#555555'));ax.text(.47,(y1+y2)/2,note,fontsize=6.5,va='center')
 restored=int(audit.get('restored_reported_analyses',0));protected=int(audit.get('protected_distinct_analyses',0))
 if restored or protected:ax.text(.04,.018,f'Restored: {restored}; retained by distinct analysis ID: {protected}',fontsize=5.8,color='#444444')
 cov=d.groupby('Dtype').agg(n=('Dtype','size'),samples=('sample','nunique')).reindex(LABELS)
 ax=axs[0,1];ax.barh(np.arange(6)+.16,cov.n,height=.3,color=COL);ax.barh(np.arange(6)-.16,cov.samples,height=.3,facecolor='white',edgecolor=COL,lw=.8);ax.set_yticks(range(6),SHORT);ax.invert_yaxis();ax.set_xlabel('Count');ax.set_xlim(0,float(cov[['n','samples']].to_numpy().max())*1.43);title(ax,'(b) Records and registered groups');ax.legend([Patch(facecolor='#729DC1'),Patch(facecolor='white',edgecolor='#729DC1')],['Records','Groups'],frameon=False,fontsize=6.5,loc='upper right',ncol=2,handlelength=1)
 for j,(nr,ng) in enumerate(zip(cov.n,cov.samples)):ax.text(max(nr,ng)+60,j,f'{nr:,} / {ng:,}',va='center',fontsize=6)
 src=d.groupby('source').size().sort_values(ascending=False).index;v=pd.crosstab(d.Dtype,d.source).reindex(index=LABELS,columns=src,fill_value=0)
 ticks=np.unique(np.rint(np.linspace(0,len(src)-1,min(4,len(src)))).astype(int))
 ax=axs[1,0];im=ax.imshow(v,aspect='auto',cmap='Blues',norm=LogNorm(vmin=1,vmax=v.to_numpy().max()));ax.set_yticks(range(6),SHORT);ax.set_xticks(ticks,ticks+1);ax.set_xlabel('Source rank by record count');title(ax,'(c) Source and class structure');fig.colorbar(im,ax=ax,shrink=.78,label='Records',pad=.025)
 ax=axs[1,1];ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');title(ax,'(d) Validation units (schematic)')
 train,test='#66889A','#D6B47E';left=.28;width=.21;gap=.025
 for k in range(3):ax.text(left+k*(width+gap)+width/2,.94,f'Source {chr(65+k)}',ha='center',fontsize=6.5)
 for ri,(label,yy) in enumerate([('Record',.77),('Sample',.50),('Source',.23)]):
  ax.text(0,yy,label,fontsize=7,va='center')
  for source in range(3):
   xx=left+source*(width+gap);ax.add_patch(Rectangle((xx,yy-.085),width,.17,facecolor='none',edgecolor='#777777',lw=.6))
   for group in range(2):
    gx=xx+.012+group*.10;ax.add_patch(Rectangle((gx,yy-.055),.088,.11,facecolor='none',edgecolor='#AAAAAA',lw=.45,ls='--'))
    for point in range(3):
     is_test=(point==2 and (source+group)%2==0) if ri==0 else ((source+group)%3==0 if ri==1 else source==2)
     ax.scatter(gx+.019+point*.025,yy,s=13,marker='s',color=test if is_test else train,linewidths=0)
  ax.text(left,yy-.125,['Groups may occur in both sets','Registered groups held apart','Source identifiers held apart'][ri],fontsize=5.8,color='#444444')
 ax.legend([Patch(facecolor=train),Patch(facecolor=test)],['Training','Test'],ncol=2,loc='lower left',bbox_to_anchor=(-.03,-.13),frameon=False,fontsize=6.3,handlelength=1)
 save(fig,1)
def figure2():
 fig,axs=plt.subplots(2,3,figsize=(6.69,4.5),layout='constrained')
 for j,(ax,e) in enumerate(zip(axs.flat,E)):violin(ax,e,e+' (ppm)');title(ax,f'({chr(97+j)}) {e}')
 save(fig,2)
def figure3():
 fig,axs=plt.subplots(2,3,figsize=(6.69,4.8),layout='constrained')
 for j,(a,b) in enumerate([('Sr/Y','Th/U'),('Th/U','La/Y'),('Sr/Y','La/Y')]):
  ax=axs[0,j]
  for k in [5,0,1,2,3,4]:
   q=med[med.Dtype==LABELS[k]];ax.scatter(q[a],q[b],s=3.2 if k==5 else 5.3,alpha=.17 if k==5 else .50,c=COL[k],marker=MARK[k],edgecolors='none',rasterized=True,zorder=1 if k==5 else 2)
  ax.set(xscale='log',yscale='log',xlabel=a,ylabel=b);title(ax,f'({chr(97+j)}) {b} versus {a}')
 for j,(ax,k) in enumerate(zip(axs[1],['Sr/Y','Th/U','La/Y'])):violin(ax,k,k);title(ax,f'({chr(100+j)}) {k}')
 fig.legend([Line2D([0],[0],marker=mk,ls='',color=c,ms=4) for c,mk in zip(COL,MARK)],SHORT,ncol=6,loc='outside upper center',frameon=False,fontsize=7)
 save(fig,3)
def figure4():
 fig,axs=plt.subplots(2,2,figsize=(6.69,4.1),layout='constrained');names=['XGBoost','LightGBM','CatBoost','ExtraTrees','Stacking'];colors=['#888888','#4C86A1','#7D9276','#B59248','#9879A0']
 for ax,key,lab,letter in zip(axs.flat,['accuracy','macro_f1','balanced_accuracy','macro_roc_auc'],['Accuracy','Macro-F1','Balanced accuracy','Macro ROC-AUC'],'abcd'):
  for name,c,mk in zip(names,colors,['o','s','^','D','P']):ax.plot(range(3),[m(scheme=s,name=name)[key] for s in SC],marker=mk,ms=3,lw=1.15 if name=='LightGBM' else .8,color=c,label=name)
  if key=='macro_roc_auc':ax.axhline(.5,ls='--',lw=.65,color='#aaaaaa')
  ax.set(xticks=range(3),xticklabels=SNAME,ylim=(0,1.025),ylabel=lab);title(ax,f'({letter}) {lab}');ax.grid(axis='y',lw=.3,color='#dddddd')
 fig.legend(*axs.flat[0].get_legend_handles_labels(),ncol=5,loc='outside upper center',frameon=False,fontsize=7)
 save(fig,4)
def figure5():
 fig,axs=plt.subplots(1,3,figsize=(6.69,2.6),layout='constrained')
 for j,task in enumerate(['six','five']):
  ax=axs[j]
  for seed in SEEDS:
   vals=[m(task=task,scheme=s,seed=seed)['macro_f1'] for s in ['record','sample']];ax.plot([0,1],vals,color='#bdc5c9',lw=.7,zorder=1);ax.scatter([0,1],vals,c=SCOL[:2],s=14,zorder=2)
  for k,s in enumerate(['record','sample']):
   mean=np.mean([m(task=task,scheme=s,seed=seed)['macro_f1'] for seed in SEEDS]);ax.plot([k-.15,k+.15],[mean,mean],color='#222222',lw=1,zorder=3)
  ax.set(xticks=[0,1],xticklabels=['Record','Sample'],ylabel='Pooled OOF Macro-F1',xlim=(-.3,1.3));title(ax,f'({chr(97+j)}) '+('Six classes' if task=='six' else 'Five mineralized classes'))
 ax=axs[2]
 for j,task in enumerate(['six','five']):
  vals=np.array([m(task=task,scheme='record',seed=s)['macro_f1']-m(task=task,scheme='sample',seed=s)['macro_f1'] for s in SEEDS]);ax.scatter(j+np.linspace(-.12,.12,len(SEEDS)),vals,s=17,color=SCOL[j]);ax.plot([j-.2,j+.2],[vals.mean()]*2,color='black',lw=1)
 ax.axhline(0,color='#999999',ls='--',lw=.7);ax.set(xticks=[0,1],xticklabels=['Six classes','Five classes'],ylabel='Record minus sample Macro-F1');title(ax,'(c) Paired differences')
 save(fig,5)
def figure6():
 fig,axs=plt.subplots(1,2,figsize=(6.69,3.2),layout='constrained')
 for ax,s,letter in zip(axs,SC[1:],'ab'):im=confusion(ax,oof(scheme=s),LABELS,f'({letter}) {s.capitalize()} disjoint')
 fig.colorbar(im,ax=list(axs),shrink=.75,label='Row percentage (%)');save(fig,6)
def figure7():
 fig,axs=plt.subplots(1,2,figsize=(6.69,2.8),layout='constrained')
 for s,col,sn in zip(SC,SCOL,SNAME):
  o=oof(scheme=s);p=o[['p_'+l for l in LABELS]].to_numpy();grid=np.linspace(0,1,501);roc=[];pr=[]
  for j,l in enumerate(LABELS):
   yy=(o.y_true==l).astype(int);a,b,_=roc_curve(yy,p[:,j]);roc.append(np.interp(grid,a,b));a,b,_=precision_recall_curve(yy,p[:,j]);pr.append(np.interp(grid,b[::-1],a[::-1]))
  ls={'record':'-','sample':'--','source':'-.'}[s];axs[0].plot(grid,np.mean(roc,0),color=col,ls=ls,label=f'{sn}: AUC = {m(scheme=s)["macro_roc_auc"]:.3f}');axs[1].plot(grid,np.mean(pr,0),color=col,ls=ls,label=f'{sn}: AP = {m(scheme=s)["macro_ap"]:.3f}')
 axs[0].plot([0,1],[0,1],ls='--',color='#aaaaaa',lw=.7)
 for ax in axs:ax.set(xlim=(0,1),ylim=(0,1.02));ax.legend(frameon=False,fontsize=7,loc='lower right' if ax==axs[0] else 'lower left')
 axs[0].set(xlabel='False-positive rate',ylabel='True-positive rate');axs[1].set(xlabel='Recall',ylabel='Precision');title(axs[0],'(a) Macro-average ROC');title(axs[1],'(b) Macro-average PR');save(fig,7)
def figure8():
 fig,axs=plt.subplots(1,2,figsize=(6.69,2.7),layout='constrained',gridspec_kw={'width_ratios':[1.1,1]})
 for j,fs in enumerate([F,F[:6],F[6:]]):
  for i,sm in enumerate([True,False]):
   suff=('_nosmote' if not sm else '')+f'_{len(fs)}features';v=m(suffix=suff)['macro_f1'];yy=j+(i-.5)*.20;axs[0].scatter(v,yy,color=SCOL[i],s=23,marker=['o','s'][i],label=['SMOTE-Tomek','No resampling'][i] if j==0 else None);axs[0].text(v+.025,yy,f'{v:.3f}',va='center',fontsize=6.8)
 axs[0].set(yticks=range(3),yticklabels=['Six + ratios','Six elements','Three ratios'],xlim=(0,1),ylim=(2.5,-.5),xlabel='Macro-F1');axs[0].legend(frameon=False,fontsize=6.4,loc='lower left',handlelength=1);axs[0].grid(axis='x',lw=.3,color='#dddddd');title(axs[0],'(a) Feature representation')
 q=read_numeric_csv(OUT/'scoring_target_comparison.csv');axs[1].scatter(q.macro_f1,range(3),color=['#7390A0','#A5B6BF','#597F92'],s=26,marker='D')
 for j,v in enumerate(q.macro_f1):axs[1].text(v+.025,j,f'{v:.3f}',va='center',fontsize=7)
 axs[1].set(yticks=range(3),yticklabels=['All records','First / group','Mean / group'],xlim=(0,1),ylim=(2.5,-.5),xlabel='Macro-F1');axs[1].grid(axis='x',lw=.3,color='#dddddd');title(axs[1],'(b) Scoring targets');save(fig,8)
def figure9():
 q=json.loads((OUT/'calibration.json').read_text(encoding='utf-8'));fig,axs=plt.subplots(2,2,figsize=(6.69,4.4),layout='constrained')
 for ax,k,letter in zip(axs[0],['before','after'],'ab'):
  bins=pd.DataFrame(q[k+'_bins']);nn=bins.n>0;color=SCOL[2 if k=='before' else 1];ax.plot([0,1],[0,1],ls='--',color='#999999',lw=.7);ax.plot(bins.loc[nn,'confidence'],bins.loc[nn,'accuracy'],color=color,lw=.9);ax.scatter(bins.loc[nn,'confidence'],bins.loc[nn,'accuracy'],s=10+26*np.sqrt(bins.loc[nn,'n']/bins.n.max()),color=color,edgecolors='white',lw=.4,zorder=3);ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean top-1 probability',ylabel='Observed accuracy');title(ax,f'({letter}) '+('Before scaling' if k=='before' else 'After scaling'))
 for k,c in [('before',SCOL[2]),('after',SCOL[1])]:bins=pd.DataFrame(q[k+'_bins']);axs[1,0].stairs(bins.n,np.arange(11)/10,color=c,label=k.capitalize(),lw=1.1)
 axs[1,0].set(xlabel='Top-1 probability bin',ylabel='Records');axs[1,0].legend(frameon=False,fontsize=7);title(axs[1,0],'(c) Confidence distribution')
 for j,key in enumerate(['log_loss','ece','mean_brier']):
  for i,k in enumerate(['before','after']):v=q[k][key];axs[1,1].bar(j+(i-.5)*.32,v,width=.3,color=[SCOL[2],SCOL[1]][i]);axs[1,1].text(j+(i-.5)*.32,v+.012,f'{v:.3f}',ha='center',fontsize=6.6)
 axs[1,1].set(xticks=range(3),xticklabels=['Log loss','ECE','Mean Brier'],ylabel='Score (lower is better)',ylim=(0,1));title(axs[1,1],'(d) Probability quality');save(fig,9)
def featurelabel(x):return x.replace('_log','').replace('_Y_ratio','/Y').replace('Th_U_ratio','Th/U')
def figure10():
 q=read_numeric_csv(OUT/'SHAP_per_fold.csv');agg=q.groupby('feature').mean(numeric_only=True).sort_values('mean_abs_shap',ascending=False);order=agg.index.tolist();fig=plt.figure(figsize=(6.69,4.8),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1,1]);a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,:])
 a.barh(range(9),agg.mean_abs_shap,xerr=q.groupby('feature').mean_abs_shap.std().reindex(order),color='#66889A',height=.65,error_kw={'lw':.7,'capsize':2});a.set_yticks(range(9),[featurelabel(f) for f in order]);a.invert_yaxis();a.set_xlabel('Mean |SHAP| (raw-score units)');title(a,'(a) Diagnostic feature contributions')
 ranks=q.pivot(index='feature',columns='fold',values='rank').reindex(order);im=b.imshow(ranks,aspect='auto',cmap='Blues_r',vmin=1,vmax=9);b.set_xticks(range(5),range(1,6));b.set_yticks(range(9),[featurelabel(f) for f in order]);b.set_xlabel('Outer sample-disjoint fold');title(b,'(b) Fold-specific ranks')
 for i in range(9):
  for j in range(5):b.text(j,i,str(ranks.iloc[i,j]),ha='center',va='center',fontsize=7,color='white' if ranks.iloc[i,j]<4 else '#222222')
 cc=read_numeric_csv(OUT/'SHAP_per_class_per_fold.csv').groupby(['cls','feature']).mean_abs_shap.mean().unstack().reindex(index=LABELS,columns=order);im=c.imshow(cc,aspect='auto',cmap='YlGnBu',vmin=0);c.set_xticks(range(9),[featurelabel(f) for f in order]);c.set_yticks(range(6),SHORT);title(c,'(c) Mean absolute contribution by model output class');fig.colorbar(im,ax=c,shrink=.8,label='Mean |SHAP|');save(fig,10)
def figure11():
 a=json.loads((OUT/'application_summary.json').read_text(encoding='utf-8'));fig,axs=plt.subplots(1,2,figsize=(6.69,2.6),layout='constrained');left=np.zeros(3)
 for l,col,sn in zip(LABELS,COL,SHORT):v=np.array([r['prediction_counts'].get(l,0)/r['n'] for r in a]);axs[0].barh(range(3),v,left=left,color=col,label=sn,height=.6);left+=v
 axs[0].set(yticks=range(3),yticklabels=[r['dataset']+f' (n={r["n"]})' for r in a],xlabel='Predicted fraction',xlim=(0,1));axs[0].invert_yaxis();title(axs[0],'(a) Prediction composition')
 rng=np.random.default_rng(42)
 for j,r in enumerate(a):
  p=read_numeric_csv(OUT/f'application_{r["dataset"]}_predictions.csv').confidence;axs[1].scatter(p,j+rng.uniform(-.16,.16,len(p)),color='#7092A3',alpha=.35,s=6,linewidths=0,rasterized=True);axs[1].scatter(r['mean_confidence'],j,marker='D',s=22,color='#222222',zorder=4);axs[1].text(.99,j-.27,f'mean {r["mean_confidence"]:.3f}',ha='right',fontsize=6.4)
 axs[1].set(yticks=range(3),yticklabels=[r['dataset'] for r in a],xlabel='Uncalibrated top-1 probability',xlim=(0,1),ylim=(2.5,-.5));title(axs[1],'(b) Record-wise confidence');fig.legend(*axs[0].get_legend_handles_labels(),ncol=6,loc='outside upper center',frameon=False,fontsize=7);save(fig,11)
def figure12():
 a=read_numeric_csv(OUT/'per_source_diagnostics.csv');ncl=d.groupby('source').Dtype.nunique();a=a[a.source.isin(ncl[ncl==1].index)];top=a[a.protocol=='source_fivefold'].nlargest(14,'n').sort_values('n').source.tolist();fig,axs=plt.subplots(1,2,figsize=(6.69,4),layout='constrained',gridspec_kw={'width_ratios':[1.25,1]})
 paired=a.pivot(index='source',columns='protocol',values='recall').reindex(top)
 for j,row in enumerate(paired.itertuples()):axs[0].plot([row.source_fivefold,row.LOSO],[j,j],color='#c2c8cb',lw=.7,zorder=1)
 for p,off,col in [('source_fivefold',-.1,SCOL[1]),('LOSO',.1,SCOL[2])]:
  q=a[a.protocol==p].set_index('source').loc[top];axs[0].scatter(q.recall,np.arange(len(top))+off,color=col,s=17,label='Source five-fold' if p=='source_fivefold' else 'LOSO')
 q=a[a.protocol=='source_fivefold'].set_index('source').loc[top];clsabbr={'IOCG':'I','Kiruna-type':'K','Orogenic':'O','Porphyry':'P','Skarn':'S','Unmineralized':'U'};rows=[f'{src} ({clsabbr[cl]}; n={n:,})' for src,cl,n in zip(top,q.cls,q.n)]
 axs[0].set(yticks=range(len(top)),yticklabels=rows,xlim=(-.03,1.03),xlabel='Source-wise recall');axs[0].tick_params(axis='y',labelsize=6.3);axs[0].legend(frameon=False,fontsize=6.4,loc='upper right');title(axs[0],'(a) Source holdout comparisons')
 q=a[a.protocol=='source_fivefold']
 for l,col,sn,mk in zip(LABELS,COL,SHORT,MARK):g=q[q.cls==l];axs[1].scatter(g.n,g.recall,color=col,s=18,marker=mk,edgecolors='#444444',lw=.3,label=sn,alpha=.75)
 axs[1].set(xscale='log',ylim=(-.04,1.04),xlabel='Records per held-out source',ylabel='Source-wise recall');axs[1].legend(ncol=2,frameon=False,fontsize=6.4,loc='upper center',bbox_to_anchor=(.5,-.18));title(axs[1],'(b) Source size and recall');save(fig,12)
def figure13():
 fig,axs=plt.subplots(2,2,figsize=(6.69,5.3),layout='constrained')
 schemes=['record','sample','deposit','source']
 for offset,seed in zip(np.linspace(-.12,.12,len(SEEDS)),SEEDS):axs[0,0].plot(np.arange(4)+offset,[m(task='five',scheme=s,seed=seed)['macro_f1'] for s in schemes],color='#d5dade',lw=.5,zorder=1)
 for j,s in enumerate(schemes):
  vals=np.array([m(task='five',scheme=s,seed=seed)['macro_f1'] for seed in SEEDS]);axs[0,0].scatter(j+np.linspace(-.12,.12,len(SEEDS)),vals,color=([*SCOL[:2],'#937DA2',SCOL[2]][j]),s=13,zorder=3);axs[0,0].plot([j-.22,j+.22],[vals.mean()]*2,color='black',lw=1,zorder=4);axs[0,0].text(j,vals.max()+.035,f'{vals.mean():.3f}',ha='center',fontsize=6.5)
 axs[0,0].set(xticks=range(4),xticklabels=['Record','Sample','Deposit text','Source'],ylim=(0,1),ylabel='Macro-F1');axs[0,0].tick_params(axis='x',labelrotation=20);title(axs[0,0],'(a) Ten-seed five-class results')
 for j,s in enumerate(schemes):
  value=m(task='five',scheme=s)['macro_roc_auc'];axs[0,1].scatter(j,value,color=[*SCOL[:2],'#937DA2',SCOL[2]][j],s=26,marker='D');axs[0,1].text(j,value-.065,f'{value:.3f}',ha='center',fontsize=6.5)
 axs[0,1].axhline(.5,ls='--',color='#aaaaaa',lw=.6)
 axs[0,1].set(xticks=range(4),xticklabels=['Record','Sample','Deposit text','Source'],xlim=(-.35,3.35),ylim=(0,1),ylabel='Macro ROC-AUC');axs[0,1].tick_params(axis='x',labelrotation=20);title(axs[0,1],'(b) Ranking at seed 42')
 for ax,s,letter in zip(axs[1],['sample','source'],'cd'):im=confusion(ax,oof(task='five',scheme=s),LABELS[:5],f'({letter}) {s.capitalize()} disjoint (%)')
 fig.colorbar(im,ax=list(axs[1]),location='bottom',shrink=.5,pad=.02,aspect=35,label='Row percentage (%)')
 save(fig,13)
def figure14():
 q=read_numeric_csv(OUT/'nested_variance_REML.csv').set_index('element').loc[E];fig,axs=plt.subplots(1,3,figsize=(6.69,3.0),layout='constrained',gridspec_kw={'width_ratios':[1,1.05,1]});bottom=np.zeros(6)
 for key,col,name in [('source_fraction','#66889A','Source'),('sample_fraction','#ADBFC6','Sample'),('residual_fraction','#D8B88A','Residual')]:axs[0].bar(range(6),q[key],bottom=bottom,color=col,label=name,width=.7);bottom+=q[key].to_numpy()
 axs[0].set(xticks=range(6),xticklabels=E,ylim=(0,1),ylabel='Conditional variance fraction');axs[0].legend(frameon=False,fontsize=6.3,ncol=3,loc='upper left',bbox_to_anchor=(-.08,-.13),columnspacing=.8,handlelength=1);title(axs[0],'(a) Nested REML components')
 for j,e in enumerate(E):axs[1].plot([q.source_fraction.iloc[j],q.same_sample_icc.iloc[j]],[j,j],color='#bbbbbb',lw=.8)
 axs[1].scatter(q.source_fraction,range(6),s=17,color='#66889A',label='Same source');axs[1].scatter(q.same_sample_icc,range(6),s=17,marker='s',color='#B98D51',label='Same sample');axs[1].set(yticks=range(6),yticklabels=E,xlim=(0,1),xlabel='Model-based correlation');axs[1].invert_yaxis();axs[1].legend(frameon=False,fontsize=6.2,loc='upper left',bbox_to_anchor=(-.08,-.13),ncol=2,columnspacing=.8,handlelength=1);title(axs[1],'(b) Intraclass correlations')
 sizes=d.groupby('sample').size().sort_values();axs[2].step(sizes,np.arange(1,len(sizes)+1)/len(sizes),where='post',color='#66889A');axs[2].set(xscale='log',xlabel='Analyses per registered group',ylabel='Cumulative group fraction',ylim=(0,1.03));title(axs[2],'(c) Unequal replication');axs[2].text(.05,.40,f'{int((sizes==1).sum()):,} singleton groups\n{int((sizes>1).sum()):,} repeated groups\n{int(sizes[sizes>1].sum()):,} analyses in repeated groups',transform=axs[2].transAxes,fontsize=6.6)
 save(fig,14)
def figure16():
 fig,axs=plt.subplots(1,3,figsize=(6.69,2.7),layout='constrained')
 # Explicit, read-only historical baseline; no previous metrics are copied into the new OUT.
 hist=read_numeric_csv(ROOT/'02_source_data/historical/3522_six_elements_derived.csv')
 hist_metrics=json.loads((ROOT/'02_source_data/historical/3522_validation_results.json').read_text(encoding='utf-8'))
 old=hist.groupby('Dtype').Citation.nunique().reindex(LABELS,fill_value=0).to_numpy();new=d.groupby('Dtype').source.nunique().reindex(LABELS,fill_value=0).to_numpy()
 for j,(before,after) in enumerate(zip(old,new)):axs[0].plot([before,after],[j,j],color='#c6cdd1',lw=.9);axs[0].scatter(before,j,color='#adb9c0',marker='o',s=18,label=f'Previous {len(hist):,}' if j==0 else None);axs[0].scatter(after,j,color=SCOL[1],marker='s',s=18,label=f'Current {len(d):,}' if j==0 else None);axs[0].text(max(before,after)+3,j,f'{before} / {after}',va='center',fontsize=6.3)
 axs[0].set(yticks=range(6),yticklabels=SHORT,xlabel='Source identifiers',xlim=(0,float(max(old.max(),new.max()))*1.38));axs[0].invert_yaxis();axs[0].legend(frameon=False,fontsize=6.1,loc='upper right');title(axs[0],'(a) Source coverage')
 historical_schemes=['record_stratified','group_sample','group_citation']
 previous_f1=[hist_metrics['schemes'][s]['results']['LightGBM']['macro_f1'] for s in historical_schemes]
 axs[1].plot(range(3),previous_f1,'-o',ms=3,color='#999999',label='Previous compilation');axs[1].plot(range(3),[m(scheme=s)['macro_f1'] for s in SC],'-s',ms=3,color=SCOL[1],label='Current compilation');axs[1].set(xticks=range(3),xticklabels=SNAME,ylim=(0,1),ylabel='Macro-F1');axs[1].legend(frameon=False,fontsize=6.3);title(axs[1],'(b) Historical comparison')
 for c,color,n in [('raw','#b6c3c9','Candidates'),('dedup','#8ba6b4','Deduplicated'),('main','#476D83','Main')]:axs[2].plot(range(3),[m(scheme=s,cohort=c)['macro_f1'] for s in SC],marker='o',ms=3,color=color,label=n)
 axs[2].set(xticks=range(3),xticklabels=SNAME,ylim=(0,1),ylabel='Macro-F1');axs[2].legend(frameon=False,fontsize=6.3);title(axs[2],'(c) Cohort processing sensitivity');save(fig,16)
def figure17():
 q=read_numeric_csv(OUT/'Mercer_case_sample_medians.csv');units=['Amphibole alteration zone','Magnetite ore','Hematite ore','Breccia pipe','Regional igneous'];uc=['#648FA4','#889B61','#B16E62','#967CAA','#B9B9B1'];un=['Amphibole alteration','Magnetite ore','Hematite ore','Breccia pipe','Regional igneous'];um=['o','^','s','D','P'];fig=plt.figure(figsize=(6.69,6.0),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[.85,1.4]);a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,0]);axd=fig.add_subplot(gs[1,1])
 assert set(q.case_unit)<=set(units), 'Map any new documented geological units explicitly before plotting'
 for u,col,name,mk in zip(units,uc,un,um):
  g=q[q.case_unit==u];a.scatter(g['Sr/Y'],g['Th/U'],s=24,color=col,marker=mk,edgecolors='#333333',lw=.4,label=name)
 a.set(xscale='log',yscale='log',xlabel='Sr/Y',ylabel='Th/U');title(a,'(a) Geological-unit compositions')
 ns=[]
 for j,(u,col,mk) in enumerate(zip(units,uc,um)):
  g=q[q.case_unit==u];v=g['La/Y'].to_numpy();ns.append(len(v));b.scatter(j+np.linspace(-.12,.12,len(v)),v,s=20,color=col,marker=mk,edgecolors='#333333',lw=.3)
  if len(v):b.plot([j-.25,j+.25],[np.median(v)]*2,color='#333333',lw=.9)
 b.set(yscale='log',xticks=range(5),xticklabels=[f'{name}\n(n={n})' for name,n in zip(['Amph.','Mag.','Hem.','Breccia','Regional'],ns)],ylabel='La/Y');title(b,'(b) Sample-level variation')
 q['order']=q.case_unit.map({u:j for j,u in enumerate(units)});q=q.sort_values(['order','Sample_Name']);im=c.imshow(q[E],aspect='auto',cmap='YlGnBu',norm=LogNorm(vmin=.1,vmax=max(10000,float(q[E].to_numpy().max()))));c.set_xticks([-1.25,-.85]+list(range(6)),['Unit','Label']+E);c.set_yticks(range(len(q)),q.Sample_Name,fontsize=6);c.set_xlim(-1.52,5.5);title(c,'(c) Composition and compilation label');fig.colorbar(im,ax=c,location='bottom',shrink=.72,label='Concentration (ppm; log scale)',pad=.025,aspect=30)
 # Side columns share the exact row order with both heatmaps. Unit colours refer
 # to the geological legend, K/U are original compilation labels, not predictions.
 for j,row in enumerate(q.itertuples()):
  c.add_patch(Rectangle((-1.39,j-.5),.28,1,facecolor=uc[row.order],edgecolor='white',lw=.35));klass=1 if row.Dtype=='Kiruna-type' else 5;c.add_patch(Rectangle((-1.00,j-.5),.30,1,facecolor=COL[klass],edgecolor='white',lw=.35));c.text(-.85,j,'K' if klass==1 else 'U',ha='center',va='center',fontsize=5.6,color='#222222')
 for t in c.get_xticklabels()[:2]:t.set_rotation(90)
 prob=[]
 for s in ['sample','source']:
  pp=read_numeric_csv(OUT/f'Mercer_case_{s}_predictions.csv');prob.append(pp.groupby('Sample_Name')['p_Kiruna-type'].mean().reindex(q.Sample_Name).to_numpy())
 v=np.array(prob).T;im=axd.imshow(v,aspect='auto',cmap='Blues',vmin=0,vmax=1);axd.set_xticks([0,1],['Sample OOF','Source OOF']);axd.set_yticks(range(len(q)),q.Sample_Name,fontsize=6);title(axd,'(d) Mean Kiruna-type probability');fig.colorbar(im,ax=axd,location='bottom',shrink=.72,label='Uncalibrated probability',pad=.025,aspect=30)
 for i in range(len(q)):
  for j in range(2):axd.text(j,i,'<0.01' if 0<v[i,j]<.005 else f'{v[i,j]:.2f}',ha='center',va='center',fontsize=5.6,color='white' if v[i,j]>.6 else '#243238')
 boundaries=[j-.5 for j in range(1,len(q)) if q.case_unit.iloc[j]!=q.case_unit.iloc[j-1]]
 for yy in boundaries:
  c.axhline(yy,color='white',lw=1.1);axd.axhline(yy,color='#777777',lw=.55)
 # The shared ordering, ticks and unit boundaries preserve one-to-one row alignment.
 c.set_ylim(len(q)-.5,-.5);axd.set_ylim(len(q)-.5,-.5)
 for ax in [c,axd]:ax.tick_params(top=False,right=False)
 fig.legend(*a.get_legend_handles_labels(),ncol=3,loc='outside upper center',frameon=False,fontsize=7);save(fig,17)
if __name__=='__main__':
 for n in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17]:globals()['figure'+str(n)]()
