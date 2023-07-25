import sys

import multihot_criteo
import torch
import torchrec.models.dlrm

torch.set_default_device(torch.device('cuda:0'))

dense_in_features = 13
sparse_in_features = 26

dlrm = torchrec.models.dlrm.DLRM_DCN_Unsharded(
    num_embeddings_per_feature=[2]*sparse_in_features,
    dense_in_features=dense_in_features,
    dense_arch_layer_sizes=[512, 256, 128],
    dcn_num_layers=3,
    dcn_low_rank_dim=512,
    over_arch_layer_sizes=[1024, 1024, 512, 256, 1],
)
print(dlrm, file=sys.stderr)

ds = multihot_criteo.MultihotCriteo(
    num_embeddings_per_feature=[40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,3],
    data_path='fake_criteo',
    name='debug',
    pre_process=multihot_criteo.pre_process_criteo_dlrm,  # currently an identity function
)
count = ds.get_item_count()
ds.load_query_samples([0])

sample, _ = ds.get_samples([0])
dense_features = sample[0].dense_features
kjt = sample[0].sparse_features
jts = [kjt[key] for key in sorted(kjt.keys())]
sparse_features = [(jt.values(), jt.offsets()) for jt in jts]
torch.onnx.export(
    dlrm,
    (sparse_features, dense_features),
    'dlrm_unsharded.onnx',
    verbose=True,
    input_names=['features_dense']+[
        f'features_sparse{i}' for i in range(sparse_in_features)
    ],
)
