"""Launcher, starter files, no-overwrite and actual isolated Jupyter kernel."""
import json,sys,tempfile,subprocess
from pathlib import Path
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from pinc_toolbox.tool_cli import _temporary_kernel
out=Path(tempfile.mkdtemp(prefix='gq-launch-test-'));example=out/'example'
subprocess.run([sys.executable,'-m','pinc_toolbox.tool_cli','--copy-example',str(example)],check=True)
assert all((example/x).is_file() for x in ['panel.csv','public.csv','settings.json','GQ_tool.ipynb','START_HERE.md'])
original=(example/'panel.csv').read_bytes();refusal=subprocess.run([sys.executable,'-m','pinc_toolbox.tool_cli','--copy-example',str(example)],capture_output=True);assert refusal.returncode!=0 and (example/'panel.csv').read_bytes()==original
runtime=out/'runtime';env=_temporary_kernel(runtime)
spec=json.loads((runtime/'share/jupyter/kernels/gq-tool/kernel.json').read_text());assert spec['argv'][0]==sys.executable
km=KernelManager(kernel_name='gq-tool',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(runtime/'share/jupyter/kernels')]))
km.start_kernel(env=env,cwd=str(out));client=km.client();client.start_channels();streams=[]
try:
 client.wait_for_ready(timeout=30)
 request=client.execute("import sys,json\nfrom pinc_toolbox.tool import launch_tool\nw=launch_tool(show=False)\nprint('GQ_KERNEL_OK',json.dumps({'python':sys.executable,'menus':sorted(w)}))")
 while True:
  msg=client.get_iopub_msg(timeout=30)
  if msg.get('parent_header',{}).get('msg_id')!=request:continue
  if msg['msg_type']=='error':raise AssertionError(msg['content'])
  if msg['msg_type']=='stream':streams.append(msg['content']['text'])
  if msg['msg_type']=='status' and msg['content']['execution_state']=='idle':break
 assert 'GQ_KERNEL_OK' in ''.join(streams)
 assert sys.executable in ''.join(streams)
finally:
 client.stop_channels();km.shutdown_kernel(now=True)
print(json.dumps({'status':'passed','starter_files':True,'existing_directory_preserved':True,'actual_jupyter_kernel_launch':True,'correct_installed_interpreter':True,'kernel_output':streams}))
