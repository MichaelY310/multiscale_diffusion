#!/usr/bin/env python3
"""Prepare native ImageFolder data from an owned ImageNet train tar or gated HF.

No resize/re-encoding. HF requires access already granted to the current user.
Does not accept terms, acquire tokens, or download validation/test data.
"""
import argparse
import io
import json
from pathlib import Path
import re
import tarfile

from workflow import inspect_data, write


def save_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Existing image differs: {path}')
        return
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_bytes(data)
    temporary.replace(path)


def from_tar(archive, output):
    with tarfile.open(archive, 'r|*') as outer:
        classes = set()
        for member in outer:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if not re.fullmatch(r'n\d{8}\.tar', name):
                raise ValueError('Expected original ILSVRC2012_img_train.tar containing class tar files')
            synset = name[:-4]
            if synset in classes:
                raise ValueError('Duplicate class archive')
            classes.add(synset)
            with tarfile.open(fileobj=outer.extractfile(member), mode='r|*') as inner:
                for image in inner:
                    if not image.isfile():
                        raise ValueError('Unexpected link/non-file inside ImageNet class tar')
                    filename = Path(image.name).name
                    if not re.fullmatch(r'n\d{8}_\d+\.(JPEG|jpeg|jpg|png)', filename):
                        raise ValueError(f'Unexpected training image name: {filename}')
                    if not filename.startswith(synset+'_'):
                        raise ValueError('Image synset differs from class archive')
                    save_bytes(output/synset/filename, inner.extractfile(image).read())
            print(f'Extracted {len(classes)} classes', flush=True)
    write(output/'TAR_EXPORT_COMPLETE.json', {'archive': str(archive.resolve()),
          'inventory': inspect_data(output), 'raw_encoded_bytes_preserved': True})


def from_hf(output):
    import os
    os.environ['HF_HUB_OFFLINE'] = '0'
    os.environ['HF_DATASETS_OFFLINE'] = '0'
    from datasets import load_dataset, Image as HFImage
    from huggingface_hub import HfApi
    from PIL import Image
    repo = 'ILSVRC/imagenet-1k'
    identity = output/'HF_EXPORT_SOURCE.json'
    revision = json.loads(identity.read_text())['revision'] if identity.exists() else HfApi().dataset_info(repo).sha
    source = {'repo': repo, 'revision': revision, 'split': 'train',
              'class_folders': '0000..0999; official HF label index, same canonical ImageNet order',
              'raw_encoded_bytes_preserved': True}
    write(identity, source)
    dataset = load_dataset(repo, split='train', revision=revision, streaming=True)
    dataset = dataset.cast_column('image', HFImage(decode=False))
    count = 0
    for index, example in enumerate(dataset):
        label = int(example['label'])
        if not 0 <= label < 1000:
            raise ValueError('Unexpected training label')
        image = example['image']
        body = image.get('bytes')
        if body is None:
            path = Path(image.get('path', ''))
            if not path.is_file():
                raise ValueError('HF did not supply original image bytes or an existing local image')
            body = path.read_bytes()
        with Image.open(io.BytesIO(body)) as pil:
            suffix = {'JPEG':'.JPEG', 'PNG':'.png'}.get(pil.format)
        if suffix is None:
            raise ValueError('Unexpected encoded image format; do not silently recompress')
        save_bytes(output/f'{label:04d}'/f'{index:08d}{suffix}', body)
        count = index+1
        if count%5000 == 0:
            write(output/'HF_EXPORT_PROGRESS.json', {**source, 'count': count})
            print(f'Exported {count}/1281167', flush=True)
    if count != 1281167:
        raise ValueError(f'Incomplete ImageNet training split: {count}')
    write(output/'HF_EXPORT_COMPLETE.json', {**source, 'count': count,
          'feature_labels': getattr(dataset.features['label'], 'names', None)})
    write(output/'DATASET_INVENTORY.json', inspect_data(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--archive', type=Path)
    group.add_argument('--hf', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.hf:
        from_hf(args.output.resolve())
    else:
        from_tar(args.archive.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
