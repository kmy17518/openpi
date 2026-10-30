import numpy as np
from openpi.training.lerobot_lowdim_test import dataset_root, _lowdim_dataset, _data_transforms, HORIZON
from openpi.training.b1k_stats_v2 import compute
from openpi.training import lerobot_lowdim
from openpi import transforms


def test_exact_statistics_exclude_all_padding_include_final_partial(dataset_root):
    dataset=_lowdim_dataset(dataset_root)
    transform=transforms.compose(_data_transforms())
    result,meta=compute(dataset,transform,HORIZON,estimator='exact')
    states=[];actions=[];padded_count=0
    for span in dataset.spans:
        raw,shared=dataset.episode_items(span)
        batch=lerobot_lowdim.transform_episode(raw,shared,transform,['state','actions'])
        mask=~np.asarray(raw['action_is_pad'])
        states.append(batch['state']);actions.append(batch['actions'][mask]);padded_count+=int((~mask).sum())
    assert padded_count>0
    for name,parts in [('state',states),('actions',actions)]:
        values=np.concatenate(parts).astype(np.float64)
        np.testing.assert_allclose(result[name].mean,values.mean(0),atol=1e-12)
        np.testing.assert_allclose(result[name].std,values.std(0),atol=1e-12)
        np.testing.assert_array_equal(result[name].q01,np.quantile(values,.01,axis=0))
        np.testing.assert_array_equal(result[name].q99,np.quantile(values,.99,axis=0))
    assert meta['frames']==len(dataset)
    assert meta['action_count']==sum(len(a) for a in actions)


def test_max_frames_is_reproducible_and_counts_valid_steps(dataset_root):
    dataset=_lowdim_dataset(dataset_root);transform=transforms.compose(_data_transforms())
    first,meta=compute(dataset,transform,HORIZON,estimator='exact',max_frames=13,seed=8)
    second,meta2=compute(dataset,transform,HORIZON,estimator='exact',max_frames=13,seed=8)
    assert meta==meta2 and meta['frames']==13
    np.testing.assert_array_equal(first['actions'].q99,second['actions'].q99)


def test_portable_manifest_validation(dataset_root,tmp_path,monkeypatch):
    import json,os,pytest
    from openpi.training.b1k_stats_v2 import validate_manifest
    monkeypatch.setenv('B1K_DATASET_REVISION','immutable-test-revision')
    dataset=_lowdim_dataset(dataset_root);transform=transforms.compose(_data_transforms())
    _,metadata=compute(dataset,transform,HORIZON,estimator='exact')
    path=tmp_path/'stats_manifest.json';path.write_text(json.dumps(metadata))
    validate_manifest(dataset.root,tmp_path)
    source=dataset.root/next(iter(metadata['source_sizes']))
    st=source.stat();os.utime(source,ns=(st.st_atime_ns,st.st_mtime_ns+12345678))
    validate_manifest(dataset.root,tmp_path)
    metadata['source_sizes'][str(source.relative_to(dataset.root))]+=1
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='source size mismatch'):validate_manifest(dataset.root,tmp_path)
    path.write_text('{}');validate_manifest(dataset.root,tmp_path)


def test_cli_selects_separate_assets_directory():
    import importlib.util,sys
    from pathlib import Path
    path=Path(__file__).parents[3]/'scripts/compute_norm_stats.py'
    spec=importlib.util.spec_from_file_location('b1k_compute_norm_cli_test',path)
    script=importlib.util.module_from_spec(spec);sys.modules[spec.name]=script
    spec.loader.exec_module(script)
    args=script.cli(['pi05_b1k','--assets-base-dir','/tmp/new-portable-assets','--estimator','exact'])
    assert args.assets_base_dir=='/tmp/new-portable-assets' and args.estimator=='exact'
