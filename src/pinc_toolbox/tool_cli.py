"""Installed launcher; no research directory or PYTHONPATH configuration."""
import argparse,json,os,sys,tempfile,shutil,subprocess
from pathlib import Path

def _temporary_kernel(directory):
    """Use the launcher's interpreter, rather than a different global python3."""
    share=directory/'share'/'jupyter';spec=share/'kernels'/'gq-tool';spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Python (GQ Tool)','language':'python'}))
    env=os.environ.copy();env['JUPYTER_PATH']=str(share)+(os.pathsep+env['JUPYTER_PATH'] if env.get('JUPYTER_PATH') else '')
    return env

def main():
    p=argparse.ArgumentParser(description='Launch the installed GQ notebook or copy a self-contained starter folder.')
    p.add_argument('--check',action='store_true',help='Construct installed menus; does not run a scientific validation.')
    p.add_argument('--copy-notebook',type=Path,help='Copy the notebook to a new file.')
    p.add_argument('--copy-example',type=Path,help='Create a new folder with CSVs, notebook, settings and English guide.')
    a=p.parse_args()
    if sum([a.check,a.copy_notebook is not None,a.copy_example is not None])>1:p.error('Choose one operation at a time')
    root=Path(__file__).resolve().parent/'_resources'
    if a.check:
        from .tool import launch_tool
        ui=launch_tool(show=False)
        print('Installed GQ Tool ready:',root, 'menus:', sorted(ui));return
    if a.copy_example:
        from .product import demo_data,demo_settings
        target=a.copy_example;target.mkdir(parents=True,exist_ok=False)
        panel,public,mapping=demo_data();panel.to_csv(target/'panel.csv',index=False,float_format='%.17g');public.to_csv(target/'public.csv',index=False,float_format='%.17g')
        (target/'settings.json').write_text(json.dumps({'panel_mapping':mapping,'public_value':'macro_W','public_release':'publication','settings':demo_settings()},indent=2))
        for name in ['GQ_tool.ipynb','START_HERE.md']:shutil.copyfile(root/name,target/name)
        print('Starter folder:',target.resolve());return
    target=a.copy_notebook or Path(tempfile.mkdtemp(prefix='gq-tool-'))/'GQ_tool.ipynb'
    if target.exists():raise SystemExit('Choose a new notebook path; existing file preserved')
    shutil.copyfile(root/'GQ_tool.ipynb',target)
    if a.copy_notebook:print(target);return
    notebook=json.loads(target.read_text());notebook['metadata']['kernelspec']={'display_name':'Python (GQ Tool)','language':'python','name':'gq-tool'};target.write_text(json.dumps(notebook,ensure_ascii=False,indent=2))
    env=_temporary_kernel(target.parent)
    raise SystemExit(subprocess.call([sys.executable,'-m','notebook',str(target)],env=env))
if __name__=='__main__':main()
