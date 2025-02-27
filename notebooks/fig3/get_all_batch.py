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
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score


def preprocess(adata):
    sc.pp.filter_genes(adata, min_cells=10)
   # sc.pp.filter_cells(adata, min_genes=30)
    sc.pp.normalize_total(adata,target_sum=1000)
    sc.pp.log1p(adata)
    scib.preprocessing.scale_batch(adata, 'sample')
    scib.preprocessing.hvg_batch(adata, 'sample')
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)

def get_silhoutte_total(adata):
    return silhouette_score(adata.obsm['X_pca'], labels=adata.obs['sample'])

def get_ilsi(adata):
    return scib.me.ilisi_graph(adata, batch_key="sample", type_="embed", use_rep="X_pca", k0 = 1500)

def get_graph_connectivity(adata, labels):
    return scib.me.graph_connectivity(adata, label_key=labels)

def get_silhoutte(adata, labels):
    result = scib.me.silhouette_batch(adata, batch_key="sample", label_key=labels, embed="X_pca")
    return result


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


def get_metrics(adata, labels):
    preprocess(adata)
    sil = get_silhoutte(adata, labels)
    sil_total = get_silhoutte_total(adata)
  #  ilsi = get_ilsi(adata)
  #  conn = get_graph_connectivity(adata, labels=labels)
    result = pd.DataFrame.from_dict({'metrics' : ['silhoutte_batch', 'silhoutte_pca', 'graph_connectivity'], 'values' : [sil, sil_total, 0]})
    return result


import pandas as pd



def get_metrics_main(pair):
    adata1 = sc.read_h5ad(f"/processed_datasets/scRecount/cellxgene/complete_collections/GeneFull/annotated/{pair[0]}")
    # some of my datasets are missing this column
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
    # check a decent portion of the data is annotated
    if adata.obs[~adata.obs['cell_typeCZI'].isna()].shape[0] / adata.shape[0] < 0.4:
        return 0,0,False
    if adata.shape[1] < 100:
        return 0,0,False
    del adata1
    del adata2
    # in a different version, I subsetted it to only shared cell types.  deleting this.
    ct = pd.crosstab(adata.obs['cell_typeCZI'], adata.obs['sample'])
    cells = list(ct.index[(ct != 0).all(axis=1)])
    adata = adata[adata.obs['cell_typeCZI'].isin(cells)]
    adata = adata[adata.obs['cell_typeCZI'] != 'nan']
    ids = adata.obs['observation_joinidCZI']
    adata.obs_names_make_unique()
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
    cells = list(ct.index[(ct != 0).all(axis=1)])
    czi = czi[czi.obs['cell_type'].isin(cells)]
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
