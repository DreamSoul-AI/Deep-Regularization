"""Apply the minimal NumPy-2 empty-priors compatibility fix to pymrmre 1.0.7.

Run with the isolated baseline interpreter. The original file and patch hashes
are retained in --record-dir; numerical estimators and the C++ kernel are intact.
"""
import argparse
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record-dir',type=Path,required=True)
    args=p.parse_args()
    dist=distribution('pymrmre')
    if dist.version!='1.0.7':
        raise RuntimeError('Patch is audited only for pymrmre 1.0.7')
    args.record_dir.mkdir(parents=True,exist_ok=True)
    record_path=args.record_dir/'pymrmre-compatibility.json'
    records=json.loads(record_path.read_text(encoding='utf-8')) if record_path.exists() else []
    if isinstance(records,dict):
        records=[dict(file='MrmreFilter.py',**records)]
    patches=[('MrmreFilter.py',b'if data._priors and len(data._priors) != 0:',b'if data._priors.size != 0:',1),
             ('MrmreData.py',b'.iteritems()',b'.items()',5)]
    for name,old,new,count in patches:
        target=Path(dist.locate_file('pymrmre/'+name))
        original=target.read_bytes()
        if old not in original and new in original:
            print(name,'already patched')
            continue
        if original.count(old)!=count:
            raise RuntimeError('Unexpected source; refusing to patch '+name)
        before=hashlib.sha256(original).hexdigest()
        backup=args.record_dir/f'{Path(name).stem}-before-{before}.py'
        if backup.exists() and backup.read_bytes()!=original:
            raise RuntimeError('Conflicting backup')
        backup.write_bytes(original)
        patched=original.replace(old,new)
        target.write_bytes(patched)
        records.append(dict(package='pymrmre',version=dist.version,file=name,before_sha256=before,
                        after_sha256=hashlib.sha256(patched).hexdigest(),old=old.decode(),new=new.decode()))
    record_path.write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    print(records)


if __name__=='__main__':
    main()
