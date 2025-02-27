import os
import anndata as ad
import pandas as pd
import anndata
import requests
import subprocess
import os
import pandas as pd
import scanpy as sc
import scib
import rapids_singlecell as rsc
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
import functools

import jax
import jax.numpy as jnp
import numpy as np

from scib_metrics.utils import cdist, get_ndarray
from scib_metrics.nearest_neighbors import  NeighborsResults



@functools.partial(jax.jit, static_argnames=["k", "recall_target"])
def _euclidean_ann(qy: jnp.ndarray, db: jnp.ndarray, k: int, recall_target: float = 0.95):
    """Compute half squared L2 distance between query points and database points."""
    dists = cdist(qy, db)
    return jax.lax.approx_min_k(dists, k=k, recall_target=recall_target)


def jax_approx_min_k(
    X: np.ndarray, n_neighbors: int, recall_target: float = 0.95, chunk_size: int = 2048
) -> NeighborsResults:
    """Run approximate nearest neighbor search using jax.

    On TPU backends, this is approximate nearest neighbor search. On other backends, this is exact nearest neighbor search.

    Parameters
    ----------
    X
        Data matrix.
    n_neighbors
        Number of neighbors to search for.
    recall_target
        Target recall for approximate nearest neighbor search.
    chunk_size
        Number of query points to search for at once.
    """
    db = jnp.asarray(X)
    # Loop over query points in chunks
    neighbors = []
    dists = []
    for i in range(0, db.shape[0], chunk_size):
        start = i
        end = min(i + chunk_size, db.shape[0])
        qy = db[start:end]
        dist, neighbor = _euclidean_ann(qy, db, k=n_neighbors, recall_target=recall_target)
        neighbors.append(neighbor)
        dists.append(dist)
    neighbors = jnp.concatenate(neighbors, axis=0)
    dists = jnp.concatenate(dists, axis=0)
    return NeighborsResults(indices=get_ndarray(neighbors), distances=get_ndarray(dists))


def get_top_pairs(all_obs, percentile=10):
    # Create a binary matrix of (source_file x cell_type) presence
    ct = pd.crosstab(all_obs['source_file'], all_obs['cell_typeCZI']).astype(bool)

    # Compute pairwise Jaccard distances (1 - Jaccard similarity)
    jaccard_dist = pdist(ct.values, metric='jaccard')
    jaccard_matrix = squareform(jaccard_dist)  # Convert to square form

    # Get the unique file names
    file_names = list(ct.index)

    # Extract all distance values (excluding self-comparisons)
    num_files = len(file_names)
    pairs = []
    for i in range(num_files):
        for j in range(i + 1, num_files):  # Only consider unique pairs
            pairs.append((file_names[i], file_names[j], jaccard_matrix[i, j]))

    # Sort by distance (ascending: closest pairs first)
    pairs.sort(key=lambda x: x[2])

    # Get the top X% closest pairs
    top_n = max(1, int(len(pairs) * (percentile / 100)))  # At least 1 pair
    return pairs[:top_n]  # Return the top closest pairs

def preprocess(adata):
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.filter_genes(adata, min_count=10)
    rsc.pp.filter_cells(adata, min_count=30, qc_var='n_genes_by_counts')
    rsc.pp.normalize_total(adata,target_sum=1000)
    rsc.pp.log1p(adata)
 #   rsc.pp.highly_variable_genes(adata)
    rsc.get.anndata_to_CPU(adata)
    sc.pp.pca(adata)
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.neighbors(adata)

def get_metrics(adata, id):
    nn = jax_approx_min_k(adata.obsm['X_pca'])
    for variable in ['lib_prep', 'tech_10x', 'cell_prep', 'sra_study', 'bioproject', 'organism', 'tissue', 'disease', 'purturbation', 'cell_line', 'srx']:
        ilisi = scib_metrics.ilisi_knn(nn, adata.obs[variable], perplexity=100)
        results.loc[id, variable] = ilisi
        results.to_csv('fig_3C_ilisi.csv')
        sil = scib_metrics.silhouette_label(adata.obsm['X_pca'], adata.obs[variable])
        results_silhouttte.loc[id, variable] = sil
        results_silhouttte.to_csv('fig_3C_sil.csv')

import pandas as pd

def get_metrics_main(pair, id):
    adata1 = sc.read_h5ad(f"/processed_datasets/scRecount/cellxgene/complete_collections/GeneFull/annotated/{pair[0]}")
    if not 'observation_joinidCZI' in adata1.obs.columns:
        return 0,0,False
    adata1.obs['sample'] = pair[0]
    adata1.obs['sample'] = adata1.obs['sample'].astype('category')
    adata2 = sc.read_h5ad(f"/processed_datasets/scRecount/cellxgene/complete_collections/GeneFull/annotated/{pair[1]}")
    if not 'observation_joinidCZI' in adata2.obs.columns:
        return 0,0,False
    adata2.obs['sample'] = pair[1]
    adata2.obs['sample'] = adata2.obs['sample'].astype('category')
    adata = anndata.concat([adata1, adata2])
    if adata.shape[0] < 100:
        return 0,0,False
    if adata.obs[~adata.obs['cell_typeCZI'].isna()].shape[0] < 1000:
        return 0,0,False

    if adata.shape[1] < 100:
        return 0,0,False
    del adata1
    del adata2
  #  ct = pd.crosstab(adata.obs['cell_typeCZI'], adata.obs['sample'])
  #  cells = list(ct.index[(ct != 0).all(axis=1)])
    #adata = adata[adata.obs['cell_typeCZI'].isin(cells)]
    adata = adata[adata.obs['cell_typeCZI'] != 'nan']
    ids = adata.obs['observation_joinidCZI']
    adata.obs_names_make_unique()
    adata.write_csv(f'./matched_adatas{id}.h5ad')
    metrics_us = get_metrics(adata, labels='cell_typeCZI')
    del adata
    czi_1 = get_czi_collection(pair[0].split('_annotated.h5ad')[0])
    czi_2 = get_czi_collection(pair[1].split('_annotated.h5ad')[0])
    if  not czi_1.raw is None:
        czi_1 = czi_1.raw.to_adata()
    if  not czi_2.raw is None:
        czi_2 = czi_2.raw.to_adata()
 #   czi_1 = czi_1.raw.to_adata()
  #  czi_2 = czi_2.raw.to_adata()
    czi_1.obs['sample'] = pair[0]
    czi_1.obs['sample'] = czi_1.obs['sample'].astype('category')
    czi_2.obs['sample'] = pair[1]
    czi_2.obs['sample'] = czi_2.obs['sample'].astype('category')
    czi = anndata.concat([czi_1, czi_2])
    czi = czi[czi.obs['observation_joinid'].isin(ids)]
    if czi.shape[0] < 100:
        return 0,0,False
    del czi_2
    del czi_1
    ct = pd.crosstab(czi.obs['cell_type'], czi.obs['sample'])
  #  cells = list(ct.index[(ct != 0).all(axis=1)])
   # czi = czi[czi.obs['cell_type'].isin(cells)]
    czi.obs_names_make_unique()
    if czi.shape[0] < 100:
        return 0,0,False
    metrics_czi = get_metrics(czi, labels = 'cell_type')
    return metrics_us, metrics_czi, True

def get_czi_collection(collection):
    czi_id_map = pd.read_csv('/processed_datasets/scRecount/cellxgene/analysis/main_df.csv')
    idd = czi_id_map.loc[czi_id_map['CZI collection name'] == collection.split('.h5ad')[0], 'collection id'].values
    # our id to name mapping is incomplete.  this is necessary because sometimes it will be 0
    print(idd)
    if (len(idd) > 0):
        idd = idd[0]
        # got this code from a tutorial
        domain_name = "cellxgene.cziscience.com"
        site_url = f"https://{domain_name}"
        api_url_base = f"https://api.{domain_name}"
        collection_path = f"/curation/v1/collections/{idd}"
        collection_url = f"{api_url_base}{collection_path}"
        res = requests.get(url=collection_url)
        res.raise_for_status()
        res_content = res.json()
        print(res_content)
        # some collections have multiple datasets.  We concattenate them.  
        # I should re-work this to use the tree concat.  ATM this requires a lot of memory.  
        if len(res_content['datasets']) > 1:
            print(len(res_content['datasets']))

            url = res_content['datasets'][0]['assets'][0]['url']
            dataset_id  = url.split('/')[-1]
            #subprocess.run(('wget -q -O ref_map_data.h5ad ' + url), shell = True)
            if not os.path.exists(f'/large_storage/goodarzilab/public/cellxgene/{dataset_id}'):
                subprocess.run(f'wget -q -O /large_storage/goodarzilab/public/cellxgene/{dataset_id} {url}', shell = True)
            ref_adata = sc.read_h5ad((f'/large_storage/goodarzilab/public/cellxgene/{dataset_id}'))

            for i in range(1,len(res_content['datasets'])):
                if res_content['datasets'][i]['is_primary_data'] != [False]:
                    url = res_content['datasets'][i]['assets'][0]['url']
                    dataset_id  = url.split('/')[-1]
                    if not os.path.exists(f'/large_storage/goodarzilab/public/cellxgene/{dataset_id}'):
                        subprocess.run(f'wget -q -O /large_storage/goodarzilab/public/cellxgene/{dataset_id} {url}', shell = True)
                 #   subprocess.run(('wget -q -O ref_map_data' + str(i) + '.h5ad  ' + url), shell = True)
                    tmp = sc.read_h5ad(f'/large_storage/goodarzilab/public/cellxgene/{dataset_id}')
                    ref_adata = anndata.concat([ref_adata, tmp], join = 'outer')
              #      ref_adata= ref_adata[ref_adata['is_primary_data'] == True]
        else:
            # if there is only one dataset in the collection.  
            url = res_content['datasets'][0]['assets'][0]['url']
            subprocess.run(('wget -q -O ref_map_data.h5ad ' + url), shell = True)
            ref_adata = sc.read_h5ad(('ref_map_data.h5ad'))
       # ref_adata.X = ref_adata.layers['raw']
        return ref_adata



merged_df = pd.read_csv('unique_combinations.csv')
pairs_to_test = get_top_pairs(merged_df)
del merged_df


pairs_to_test = [x for x in pairs_to_test if x[2] != 0]
unique_tuples = set()
filtered_list = []

for index, (a, b, num) in enumerate(pairs_to_test):
    pair = tuple(sorted([a, b]))  # Sort ensures order-independent comparison
    if  pair not in unique_tuples:
        if a != b:
            ind = index
            unique_tuples.add(pair)
            filtered_list.append((a, b, num, ind))
print(len(filtered_list))
import random
#random.shuffle(filtered_list)
print(filtered_list)
import pickle
with open("batch_results2/filtered_list2.pkl", "wb") as f:
        pickle.dump(filtered_list, f)
#us, czi = get_metrics_main(pairs_to_test[0])
for pair in tqdm(filtered_list):
    if not os.path.exists(f'batch_results3/czi_{pair[3]}.csv'):
        us, czi, b = get_metrics_main(pair)
        if b:
            us.to_csv(f'batch_results3/us_{pair[3]}.csv')
            czi.to_csv(f'batch_results3/czi_{pair[3]}.csv')
    else:
        print('file seen')
