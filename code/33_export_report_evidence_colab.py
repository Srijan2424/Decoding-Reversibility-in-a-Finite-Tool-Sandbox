"""Export compact report evidence, excluding activations, credentials and private labels."""
import json,hashlib,zipfile
from pathlib import Path
from datetime import datetime,timezone

def main():
    from google.colab import drive,files
    root=Path('/content/drive/MyDrive/ADNN_Irreversibility')
    if not root.is_dir():drive.mount('/content/drive')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    dest=root/'submission_exports';dest.mkdir(exist_ok=True)
    target=dest/f'report_evidence_{stamp}.zip'
    # All matching summaries are included, not selected by score.
    allowed={'.csv','.png','.pdf'}
    names={'summary.json','config.json','REPORT_NOTES.md','selected_results.json','extended_summary.json','quick_summary.json'}
    prefixes=('sandbox_statistics_','sandbox_ablation_audit_','qwen_sandbox_main_probes_','llama_sandbox_main_probes_',
              'qwen_sandbox_reduced_controls_','llama_sandbox_reduced_controls_','qwen_sandbox_reduced_ablations_',
              'llama_sandbox_reduced_ablations_','qwen_rq3_v2_','llama_rq3_v2_','knowledge_answer_audit_',
              'qwen_paired_knowledge_','llama_paired_knowledge_','scienceworld_semantics_audit_')
    inventory=[]
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for folder in sorted((root/'results').iterdir()):
            if not folder.is_dir() or not folder.name.startswith(prefixes):continue
            for path in sorted(folder.rglob('*')):
                if not path.is_file():continue
                if path.name not in names and path.suffix.lower() not in allowed:continue
                if path.stat().st_size>20*1024*1024:continue
                data=path.read_bytes();relative=str(path.relative_to(root))
                z.writestr(relative,data)
                inventory.append({'path':relative,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        z.writestr('manifest.json',json.dumps({'files':inventory,'note':'Report evidence export only; original experiment logs remain in Drive. Annotation labels and approval correspondence must be supplied separately.'},indent=2))
    assert inventory,'No report files found; check project path'
    print(json.dumps({'export_complete':True,'files':len(inventory),'size_mib':round(target.stat().st_size/2**20,2),'saved_to':str(target),'scope':'Report evidence, not complete reproducibility submission'},indent=2))
    files.download(str(target))
if __name__=='__main__':main()
