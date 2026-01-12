import pandas as pd
import numpy as np
import os
import sys
import warnings
from datetime import datetime

from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.linear_model import LogisticRegressionCV    
from statsmodels.stats.multitest import multipletests

try:
    from crt_logit import crt_logit
    from other_methods import desparsified_lasso, model_x_knockoff
except ImportError:
    print("WARNING: 'crt_logit' or 'other_methods' not found. Ensure they exist.")

CONFIG = {
    'N_JOBS': 14,
    'TARGET_FEATURES': 1000,  
    'FDR_LEVEL': 0.1,
    'SEED': 2305,
    'X_PATH': r'C:\Users\admin\Desktop\TCGA Cancer\X_final_sorted_full.csv',
    'Y_PATH': r'C:\Users\admin\Desktop\TCGA Cancer\y_final.csv',         
    'GT_PATH': r'C:\Users\admin\Desktop\TCGA Cancer\ground_truth_genes.csv',
    'MAX_ITER': 10000000000000,
    'OUTPUT_DIR': r'C:\Users\admin\Desktop\TCGA Cancer'
}

def log(msg):
    timestamp = datetime.now().strftime('%H:%M:%S')
    return f"[{timestamp}] {msg}"

def clean_gene_names(columns):
    return np.array([str(c).split('|')[0] for c in columns])

def load_and_process_data(config):
    print(log("Loading data..."))
    
    if not os.path.exists(config['OUTPUT_DIR']):
        try:
            os.makedirs(config['OUTPUT_DIR'])
        except:
            pass

    if not os.path.exists(config['X_PATH']):
        print(log("Cannot find dataset; Creating simulate data..."))
        n_samples, n_features = 200, 5000
        return (np.random.randn(n_samples, n_features), 
                np.random.randint(0, 2, n_samples), 
                np.array([f"Gene_{i}" for i in range(n_features)]), 
                set())

    base_dir = os.path.dirname(config['X_PATH'])
    y_path = os.path.join(base_dir, config['Y_PATH']) if not os.path.isabs(config['Y_PATH']) else config['Y_PATH']
    gt_path = os.path.join(base_dir, config['GT_PATH']) if not os.path.isabs(config['GT_PATH']) else config['GT_PATH']

    X_df = pd.read_csv(config['X_PATH'], index_col=0)
    y_df = pd.read_csv(y_path, index_col=0)
    
    gt_genes = set()
    if os.path.exists(gt_path):
        gt_raw = pd.read_csv(gt_path).iloc[:, 0].astype(str).str.strip().values
        gt_genes = set(clean_gene_names(gt_raw))
        
    common_idx = X_df.index.astype(str).intersection(y_df.index.astype(str))
    X = X_df.loc[common_idx].values.astype(float)
    y = y_df.loc[common_idx].values.ravel().astype(int)
    gene_names = clean_gene_names(X_df.columns)
    
    print(log(f"   - Samples: {len(common_idx)} | Genes: {X.shape[1]}"))

    X = SimpleImputer(strategy='mean').fit_transform(X)
    if np.min(X) < 0: X -= np.min(X) 
    X = np.log1p(X)
    
    vars_ = np.var(X, axis=0)
    mask = vars_ > 1e-6
    X = X[:, mask]
    gene_names = gene_names[mask]
    
    print(log(f"   - Variance filtered: {X.shape[1]} genes"))
    return X, y, gene_names, gt_genes

def perform_clustering(X, n_clusters):
    print(log(f" Clustering features: {n_clusters} clusters..."))
    connectivity = grid_to_graph(n_x=X.shape[1], n_y=1, n_z=1)
    agglo = FeatureAgglomeration(n_clusters=n_clusters, connectivity=connectivity, linkage='ward')
    
    X_reduced = agglo.fit_transform(X)
    X_std = StandardScaler().fit_transform(X_reduced)
    
    return X_std, agglo.labels_

def collect_and_print_results(method_name, selected_indices, pvals, labels, gene_names, gt_genes, csv_accumulator, txt_accumulator):
    header_str = f"\n RESULTS: {method_name}\n"
    header_str += f"   Number of clusterings: {len(selected_indices)} (FDR < {CONFIG['FDR_LEVEL']})"
    
    print(header_str)
    txt_accumulator.append(header_str)
    
    if len(selected_indices) == 0: 
        return

    hits = []
    for cid in selected_indices:
        members = gene_names[labels == cid]
        found_gt = [g for g in members if g in gt_genes]
        rep = found_gt[0] if found_gt else (members[0] if len(members) > 0 else f"Clust_{cid}")
        pval = pvals[cid] if (pvals is not None and cid < len(pvals)) else -1.0 
        
        hits.append({
            'gene': rep, 
            'pval': pval, 
            'is_gt': bool(found_gt), 
            'size': len(members),
            'members': ";".join(members),
            'cid': cid
        })

    hits.sort(key=lambda x: x['pval'] if x['pval'] != -1 else 0)

    table_header = f"   {'Representative Genes':<25} | {'P-value':<10} | {'Size':<5} | {'Is GT':<5}"
    sep_line = "   " + "-"*70
    
    print(table_header)
    print(sep_line)
    txt_accumulator.append(table_header)
    txt_accumulator.append(sep_line)

    for h in hits:
        pval_str = f"{h['pval']:.2e}" if h['pval'] != -1 else "N/A"
        is_gt_str = "Yes" if h['is_gt'] else "No"
        
        if h['is_gt']: 
            line_str = f"   {h['gene']:<25} | {pval_str:<10} | {h['size']:<5} | {is_gt_str:<5}"
            print(line_str)
            txt_accumulator.append(line_str)
        
        csv_accumulator.append({
            'Method': method_name,
            'Cluster_ID': h['cid'],
            'Representative_Gene': h['gene'],
            'P_Value': h['pval'] if h['pval'] != -1 else None,
            'Is_Ground_Truth': h['is_gt'],
            'Cluster_Size': h['size'],
            'All_Genes_In_Cluster': h['members']
        })

if __name__ == "__main__":
    np.random.seed(CONFIG['SEED'])
    warnings.filterwarnings("ignore")
    
    all_csv_rows = []
    all_txt_lines = []
    
    start_msg = "="*60 + "\nTCGA GLIOMA ANALYSIS - " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n" + "="*60
    print(start_msg)
    all_txt_lines.append(start_msg)

    X, y, gene_names, gt_genes = load_and_process_data(CONFIG)
    
    X_clustered, cluster_labels = perform_clustering(X, CONFIG['TARGET_FEATURES'])

    methods_results_summary = []

    try:
        test_clf = LogisticRegressionCV(
            Cs=np.logspace(-4, 1, 20), 
            penalty='l1', solver='saga', class_weight='balanced', 
            max_iter=5000, tol=1e-4, n_jobs=-1, cv=5
        )
        test_clf.fit(X_clustered, y)

        print(log(" RUNNING CRT-logit..."))
        if 'crt_logit' in globals():
            sel_crt, pvals_crt, _ = crt_logit(
                X_clustered, y, estimated_coef=test_clf.coef_[0], fdr=CONFIG['FDR_LEVEL'], 
                n_jobs=CONFIG['N_JOBS'], screening=True, centered=False, verbose=True
            )
        else:
            sel_crt, pvals_crt = [], []

        collect_and_print_results("CRT-logit", sel_crt, pvals_crt, cluster_labels, gene_names, gt_genes, all_csv_rows, all_txt_lines)
        methods_results_summary.append(("CRT-logit", len(sel_crt)))
    except Exception as e:
        err_msg = log(f" CRT-logit ERROR: {e}")
        print(err_msg)
        all_txt_lines.append(err_msg)

    try:
        print(log(" RUNNING Debiased Lasso..."))
        if 'desparsified_lasso' in globals():
            _, _, pvals_lasso = desparsified_lasso(
                X_clustered, y, centered=False, method="lasso_cv", n_jobs=CONFIG['N_JOBS']
            )
            rejected, _, _, _ = multipletests(pvals_lasso, alpha=0.1, method='fdr_bh')
            sel_lasso = np.where(rejected)[0]
        else:
             sel_lasso, pvals_lasso = [], []

        collect_and_print_results("Debiased Lasso", sel_lasso, pvals_lasso, cluster_labels, gene_names, gt_genes, all_csv_rows, all_txt_lines)
        methods_results_summary.append(("Debiased Lasso", len(sel_lasso)))
    except Exception as e:
        err_msg = log(f" Debiased Lasso ERROR: {e}")
        print(err_msg)
        all_txt_lines.append(err_msg)

    try:
        print(log(" RUNNING Model-X Knockoff..."))
        if 'model_x_knockoff' in globals():
            sel_ko, _, _, _ = model_x_knockoff(
                X_clustered, y, fdr=0.1, offset=1, 
                loss='logistic', method='equi', centered=False
            )
        else:
            sel_ko = []

        collect_and_print_results("Knockoff", sel_ko, None, cluster_labels, gene_names, gt_genes, all_csv_rows, all_txt_lines)
        methods_results_summary.append(("Knockoff", len(sel_ko)))
    except Exception as e:
        err_msg = log(f" Knockoff ERROR: {e}")
        print(err_msg)
        all_txt_lines.append(err_msg)

    summary_header = "\n" + "="*60 + "\nRESULTS SUMMARY:\n"
    print(summary_header.strip())
    all_txt_lines.append(summary_header)
    
    for name, count in methods_results_summary:
        line = f" - {name:<20}: {count} clusters"
        print(line)
        all_txt_lines.append(line)
    
    print("="*60)
    all_txt_lines.append("="*60)

    csv_filename = os.path.join(CONFIG['OUTPUT_DIR'], "analysis_results.csv")
    if all_csv_rows:
        df_results = pd.DataFrame(all_csv_rows)
        cols = ['Method', 'Representative_Gene', 'P_Value', 'Is_Ground_Truth', 'Cluster_Size', 'Cluster_ID', 'All_Genes_In_Cluster']
        df_results = df_results[cols]
        df_results.to_csv(csv_filename, index=False)
        print(f"\n[INFO] Saved detailed CSV to: {csv_filename}")
    else:
        print("\n[INFO] No significant results found to save to CSV.")

    txt_filename = os.path.join(CONFIG['OUTPUT_DIR'], "analysis_report.txt")
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(all_txt_lines))
    print(f"[INFO] Saved report text to: {txt_filename}")