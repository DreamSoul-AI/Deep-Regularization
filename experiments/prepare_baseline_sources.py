"""Stage pinned author sources for the paired ranking suite, without checking out branches."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.request

LEGACY='555ba11e8872bd6618cd721bd3b08f7128db9e6b'
LOCALIZED='f7944f2913696a3fd0fb67d4e43552af69797d52'
DEEP='a02a7ee6989549738ec5924ec006839007342771'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory; existing experiment sources are immutable.')
    output.mkdir(parents=True,exist_ok=True)
    git=['git','-c',f'safe.directory={root.as_posix()}','-C',str(root)]
    files=subprocess.check_output(git+['ls-tree','-r','--name-only',LEGACY],text=True).splitlines()
    entries=[]
    def save(path,data,source):
        dest=output/path
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(data)
        entries.append(dict(path=str(path),source=source,sha256=hashlib.sha256(data).hexdigest()))
    for file in files:
        if file.endswith('.py') and (file.startswith('src/') or file.startswith('vendor_pkgs/e2efs/')):
            data=subprocess.check_output(git+['show',f'{LEGACY}:{file}'])
            save(Path('legacy')/file,data,f'git:{LEGACY}:{file}')
    localized_url=f'https://raw.githubusercontent.com/myamada0321/Localizedlasso/{LOCALIZED}/LocalizedLasso.py'
    raw=urllib.request.urlopen(localized_url).read()
    save(Path('localized-LocalizedLasso.py'),raw,localized_url)
    converted=re.sub(r'(?m)^(\s*)print (.*)$',r'\1print(\2)',raw.decode()).encode()
    save(Path('localized-py3.py'),converted,'Python 2 print syntax conversion only')
    for file in ['deep_lasso.py','deep_tabular/models/mlp.py',
                 'deep_tabular/models/ft_transformer_attention_map.py']:
        url=f'https://raw.githubusercontent.com/vcherepanova/tabular-feature-selection/{DEEP}/{file}'
        save(Path('upstream')/file,urllib.request.urlopen(url).read(),url)
    compatibility=[]
    def patch(file,change,description):
        path=output/file
        old=path.read_bytes()
        changed=change(old.decode()).encode()
        path.write_bytes(changed)
        compatibility.append(dict(file=file,description=description,
            before=hashlib.sha256(old).hexdigest(),after=hashlib.sha256(changed).hexdigest()))
    for file in ['legacy/src/nn_wrapper.py','legacy/src/fsnet.py']:
        patch(file,lambda text:text.replace('verbose=False,',''),'Remove unsupported scheduler verbose keyword')
    patch('legacy/src/e2efs_wrapper.py',lambda text:text.replace('parents[2] / "vendor_pkgs"','parents[1] / "vendor_pkgs"'),
          'Resolve vendor path inside the staged tree')
    # Clone validation snapshots, rather than keeping aliases to live tensors.
    file='legacy/src/nn_wrapper.py'
    patch(file,lambda text:text.replace('self.model.state_dict()',
          '{k: v.detach().clone() for k, v in self.model.state_dict().items()}'),
          'Freeze stored validation state dictionaries; record this correctness fix separately from historical execution')
    (output/'sources.json').write_text(json.dumps(entries,indent=2)+'\n')
    (output/'compatibility.json').write_text(json.dumps(compatibility,indent=2)+'\n')


if __name__=='__main__':
    main()
