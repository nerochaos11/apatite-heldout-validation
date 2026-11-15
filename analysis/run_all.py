from pathlib import Path
import os,sys,subprocess,json,datetime,hashlib,importlib.metadata,platform

WORK=Path(__file__).resolve().parent
from paths import ROOT, OUT
OUT.mkdir(parents=True,exist_ok=True)
LOG=OUT/'logs';LOG.mkdir(exist_ok=True)
sys.stdout.reconfigure(encoding='utf-8')
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='4'
os.environ['PYTHONIOENCODING']='utf-8'
os.environ['APATITE_FORCE_RETRAIN']='1'
environment={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
             'packages':{name:importlib.metadata.version(name) for name in
                         ['numpy','pandas','scikit-learn','imbalanced-learn','lightgbm','xgboost','catboost','scipy','matplotlib','joblib']},
             'threads':4,'force_retrain':True,'previous_predictions_reused':False}
(OUT/'environment_versions.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2),encoding='utf-8')
phases=[('primary',['experiment_core.py','--phase','primary']),
        ('legacy_control',['experiment_core.py','--phase','legacy_control']),
        ('benchmark',['experiment_core.py','--phase','benchmark']),
        ('repeated',['experiment_core.py','--phase','repeated']),
        ('comparison',['experiment_core.py','--phase','comparison']),
        ('main_compat',['experiment_core.py','--phase','main_compat']),
        ('ablation',['experiment_core.py','--phase','ablation']),
        ('configuration_sensitivity',['configuration_sensitivity.py']),
        ('diagnostics',['supplementary.py','--phase','diagnostics']),
        ('loso',['supplementary.py','--phase','loso']),
        ('case_sensitivity',['case_and_sensitivity.py']),
        ('collate',['collate.py']),
        ('variance_verification',['verify_variance.py']),
        ('application',['application.py'])]
resume='--resume' in sys.argv
history=json.loads((OUT/'retrain_stage_log.json').read_text(encoding='utf-8')) if resume else []
for phase,args in phases:
    previous=next((r for r in history if r['phase']==phase),None)
    input_hash=hashlib.sha256((OUT/'main_data_clean.csv').read_bytes()).hexdigest()
    if previous and previous['exit_code']==0:
        assert previous['input_sha256']==input_hash
        print('ALREADY_COMPLETED_THIS_RUN',phase,flush=True)
        continue
    record=dict(phase=phase,started=datetime.datetime.now().isoformat(),command=[sys.executable,*args],
                input_sha256=input_hash)
    if previous:
        import shutil
        previous_log=LOG/(phase+'_attempt1.log')
        if not previous_log.exists():shutil.copy2(LOG/(phase+'.log'),previous_log)
        record['previous_attempts']=previous.get('previous_attempts',[])+[previous]
    print('START',phase,record['started'],flush=True)
    with (LOG/(phase+'.log')).open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,str(WORK/args[0]),*args[1:]],cwd=WORK,stdout=log,stderr=subprocess.STDOUT)
    record.update(finished=datetime.datetime.now().isoformat(),exit_code=result.returncode)
    if previous:history[history.index(previous)]=record
    else:history.append(record)
    (OUT/'retrain_stage_log.json').write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding='utf-8')
    print('END',phase,record['exit_code'],record['finished'],flush=True)
    if result.returncode:
        print((LOG/(phase+'.log')).read_text(encoding='utf-8')[-6000:],flush=True)
        raise SystemExit(result.returncode)
assert len(list((OUT/'OOF').glob('*_metrics.json')))==106
print('ALL_PHASES_COMPLETE: 106 OOF experiments plus diagnostics, LOSO and applications',flush=True)
