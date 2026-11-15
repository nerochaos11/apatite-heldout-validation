from experiment_core import *
d=load();parts=[]
for l,g in d.groupby('Dtype'):
 keys=g['sample'].drop_duplicates().head(3);parts += [g[g['sample']==k].head(4) for k in keys]
q=pd.concat(parts);y=q.Th_log.to_numpy();ss=q.source.to_numpy();gg=q['sample'].to_numpy();X=np.eye(6)[[LABELS.index(c) for c in q.Dtype]];errors=[]
for a,b,e in [[.1,.2,.3],[1,.05,.01],[.01,.01,.5]]:
 V=a*(ss[:,None]==ss[None,:])+b*(gg[:,None]==gg[None,:])+e*np.eye(len(q));vi=np.linalg.inv(V);xx=X.T@vi@X;xy=X.T@vi@y
 dense=.5*(np.linalg.slogdet(V)[1]+np.linalg.slogdet(xx)[1]+y@vi@y-xy@np.linalg.solve(xx,xy)+(len(q)-6)*np.log(2*np.pi))
 ld=0.;yy=0.;xx=np.zeros((6,6));xy=np.zeros(6)
 for _,h in q.groupby('source'):
  Q=0.;R=0.;qc=np.zeros(6);rc=np.zeros(6);t=0.
  for _,z in h.groupby('sample'):
   n=len(z);s=z.Th_log.sum();den=e+n*b;k=LABELS.index(z.Dtype.iloc[0]);u=n/den;v=s/den;Q+=u;R+=v;qc[k]+=u;rc[k]+=v
   ld+=(n-1)*np.log(e)+np.log(den);t+=np.sum(z.Th_log**2)/e-b*s*s/(e*den)
  fac=a/(1+a*Q);ld+=np.log1p(a*Q);yy+=t-fac*R*R;xx+=np.diag(qc)-fac*np.outer(qc,qc);xy+=rc-fac*qc*R
 block=.5*(ld+np.linalg.slogdet(xx)[1]+yy-xy@np.linalg.solve(xx,xy)+(len(q)-6)*np.log(2*np.pi));errors.append(abs(dense-block));assert errors[-1]<1e-7
print('maximum covariance difference: ',format(max(errors),'.3g'))
print('Dense covariance verification',errors)
