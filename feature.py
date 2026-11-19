import pandas as pd
import numpy as np
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from rdkit import Chem
from rdkit.Chem import AllChem
import os



try:
    from kaleido import write_fig_sync

    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False
    print("Warning: kaleido not installed. Sankey chart PNG/PDF export disabled.")

warnings.filterwarnings('ignore')
np.random.seed(42)

# Output directories
output_dir = "ACM_BCB_41"
os.makedirs(output_dir, exist_ok=True)
result_dir = os.path.join(output_dir, "result_new")
os.makedirs(result_dir, exist_ok=True)


# ============================================================================
# 1. DATA LOADING (with fallback)
# ============================================================================
def load_and_analyze_data(filepath):
    """Load and analyze data. If file missing, generate synthetic."""
    print("=" * 80)
    print("DATA LOADING")
    print("=" * 80)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    else:
        # Generate synthetic data (same structure as your real data)
        cell_lines = ['A549', 'MCF7', 'PC3', 'HT29', 'SKOV3']
        drugs = ['Saracatinib', 'JNK-IN-8', 'Dasatinib', 'PF562271', 'Defactinib']
        concs = [0.1, 0.5, 1.0, 5.0, 10.0]
        data = []
        for cell in cell_lines:
            for drug in drugs:
                for d1 in concs:
                    for d2 in concs:
                        for d3 in concs:
                            viability = np.random.uniform(0.1, 0.9)
                            if d1 > 1 and d2 > 1 and d3 > 1:
                                viability *= 0.5
                            data.append({'Cell Name': cell, 'Drug3': drug,
                                         'Drug1 Concentration': d1, 'Drug2 Concentration': d2,
                                         'Drug3 Concentration': d3,
                                         'Mean Relative Viability': viability,
                                         'Mean Apoptosis Fraction': 1 - viability})
        df = pd.DataFrame(data)
        print(f"Synthetic data created: {df.shape}")
    # Basic cleaning
    for col in ['Mean Relative Viability', 'Mean Apoptosis Fraction']:
        if col in df.columns:
            m, s = df[col].mean(), df[col].std()
            df = df[(df[col] >= m - 4 * s) & (df[col] <= m + 4 * s)]
    all_cells = sorted(df['Cell Name'].unique())
    all_drugs = sorted(df['Drug3'].unique())
    df.to_csv(os.path.join(result_dir, 'processed_data.csv'), index=False)
    return df, all_drugs, [], all_cells, all_drugs

# ============================================================================
# 2. SYNERGY SCORES (Bliss, HSA, Loewe)
# ============================================================================
def calculate_synergy_scores(df, universal_drugs, limited_drugs):
    """Calculate Bliss, HSA, Loewe synergy scores."""
    print("\nCalculating synergy scores...")
    results = []
    for cell in df['Cell Name'].unique():
        for drug3 in df['Drug3'].unique():
            subset = df[(df['Cell Name'] == cell) & (df['Drug3'] == drug3)]
            if len(subset) == 0: continue
            mono1 = subset[(subset['Drug2 Concentration'] == 0) & (subset['Drug3 Concentration'] == 0)]
            mono2 = subset[(subset['Drug1 Concentration'] == 0) & (subset['Drug3 Concentration'] == 0)]
            mono3 = subset[(subset['Drug1 Concentration'] == 0) & (subset['Drug2 Concentration'] == 0)]
            if len(mono1) == 0 or len(mono2) == 0 or len(mono3) == 0: continue
            e1 = mono1['Mean Relative Viability'].mean()
            e2 = mono2['Mean Relative Viability'].mean()
            e3 = mono3['Mean Relative Viability'].mean()
            combos = subset[(subset['Drug1 Concentration'] > 0) & (subset['Drug2 Concentration'] > 0) & (
                        subset['Drug3 Concentration'] > 0)]
            for _, row in combos.iterrows():
                d1, d2, d3 = row['Drug1 Concentration'], row['Drug2 Concentration'], row['Drug3 Concentration']
                e_obs = row['Mean Relative Viability']
                e_bliss = e1 * e2 * e3
                synergy_bliss = e_bliss - e_obs
                e_hsa = max(e1, e2, e3)
                synergy_hsa = e_hsa - e_obs
                # Simplified Loewe = HSA
                synergy_loewe = synergy_hsa
                # Bootstrap for p-value
                boot = []
                for _ in range(100):
                    e1b = np.random.normal(e1, 0.05)
                    e2b = np.random.normal(e2, 0.05)
                    e3b = np.random.normal(e3, 0.05)
                    e_obsb = np.random.normal(e_obs, 0.05)
                    boot.append((e1b * e2b * e3b) - e_obsb)
                pval = np.mean(np.array(boot) <= 0)
                # Classification
                if synergy_bliss > 0.6 and pval < 0.05:
                    cls = 'Strong Synergy'
                elif synergy_bliss > 0.4 and pval < 0.05:
                    cls = 'Moderate Synergy'
                elif synergy_bliss > 0:
                    cls = 'Weak Synergy'
                elif synergy_bliss < -0.2 and pval < 0.05:
                    cls = 'Strong Antagonism'
                elif synergy_bliss < -0.1 and pval < 0.05:
                    cls = 'Moderate Antagonism'
                elif synergy_bliss < -0.05:
                    cls = 'Weak Antagonism'
                else:
                    cls = 'Additive'
                results.append({'Cell_Line': cell, 'Drug3': drug3, 'D1_conc': d1, 'D2_conc': d2, 'D3_conc': d3,
                                'Observed_Viability': e_obs, 'Bliss_Expected': e_bliss, 'Bliss_Synergy': synergy_bliss,
                                'HSA_Expected': e_hsa, 'HSA_Synergy': synergy_hsa, 'Loewe_Synergy': synergy_loewe,
                                'Synergy_p_value': pval, 'Synergy_Class': cls})
    synergy_df = pd.DataFrame(results)
    if len(synergy_df) > 0:
        _, pcorr, _, _ = multipletests(synergy_df['Synergy_p_value'].values, method='fdr_bh')
        synergy_df['Synergy_p_adj'] = pcorr
    synergy_df.to_csv(os.path.join(result_dir, 'synergy_results.csv'), index=False)
    print(f"Calculated {len(synergy_df)} synergy entries.")
    return synergy_df

# ============================================================================
# 3. FEATURE ENGINEERING (PCA for cell lines, Morgan fingerprints for drugs)
# ============================================================================

def load_cell_line_features(expression_file='cellline_expression.csv', n_components=50):
    """Load RNA-seq logTPM, perform PCA to n_components (auto-reduce if needed), return dict cell->features"""
    if not os.path.exists(expression_file):
        print(f"Warning: {expression_file} not found. Using dummy cell features.")
        return {}
    expr_df = pd.read_csv(expression_file, index_col=0)  # rows: cell lines, columns: genes
    print(f"Loaded expression data shape: {expr_df.shape}")
    n_samples, n_features = expr_df.shape
    # 调整 PCA 组件数
    max_components = min(n_samples, n_features)
    if n_components > max_components:
        print(
            f"Warning: Requested n_components={n_components} exceeds max possible {max_components}. Reducing to {max_components}.")
        n_components = max_components
    if n_components <= 0:
        print("Error: No components can be extracted. Returning zeros.")
        return {cell: np.zeros(1) for cell in expr_df.index}
    name_col = expr_df.columns[0]
    id_col = expr_df.columns[1]
    feature_cols = expr_df.columns[2:]
    X = expr_df[feature_cols].values
    scaler = StandardScaler()
    expr_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    expr_pca = pca.fit_transform(expr_scaled)
    cell_features = {cell: expr_pca[i] for i, cell in enumerate(expr_df[name_col].values)}
    print(f"Loaded cell line features: {len(cell_features)} cell lines, {n_components} PCA components")
    return cell_features

def load_drug_morgan(smiles_file, radius=2, nBits=1024):
    """Load SMILES and return dict {drug: Morgan fingerprint array}."""
    if os.path.exists(smiles_file):
        smiles_df = pd.read_csv(smiles_file)
        drug_to_fp = {}
        for _, row in smiles_df.iterrows():
            drug = row['Drug3']
            smiles = row['Drug3_smiles']
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
            arr = np.zeros(nBits, dtype=np.float32)
            for i in range(nBits):
                arr[i] = fp.GetBit(i)
            drug_to_fp[drug] = arr
        print(f"Loaded {len(drug_to_fp)} drug fingerprints.")
        return drug_to_fp
    else:
        print("SMILES file not found. Using random drug features.")
        return {}

def build_feature_matrix(df, synergy_df, cell_features, drug_features):
    """Combine concentration, log, interactions, PCA cell features, Morgan drug features."""
    # Merge
    ml = pd.merge(df, synergy_df[['Cell_Line', 'Drug3', 'D1_conc', 'D2_conc', 'D3_conc',
                                  'Bliss_Synergy', 'Observed_Viability']],
                  left_on=['Cell Name', 'Drug3', 'Drug1 Concentration', 'Drug2 Concentration', 'Drug3 Concentration'],
                  right_on=['Cell_Line', 'Drug3', 'D1_conc', 'D2_conc', 'D3_conc'], how='inner')
    features = []
    # Concentrations
    for col in ['Drug1 Concentration', 'Drug2 Concentration', 'Drug3 Concentration']:
        ml[f'log_{col}'] = np.log10(ml[col] + 1e-9)
        features.extend([f'log_{col}', col])
    # Interactions
    ml['D1xD2'] = ml['Drug1 Concentration'] * ml['Drug2 Concentration']
    ml['D1xD3'] = ml['Drug1 Concentration'] * ml['Drug3 Concentration']
    ml['D2xD3'] = ml['Drug2 Concentration'] * ml['Drug3 Concentration']
    ml['D1xD2xD3'] = ml['Drug1 Concentration'] * ml['Drug2 Concentration'] * ml['Drug3 Concentration']
    features.extend(['D1xD2', 'D1xD3', 'D2xD3', 'D1xD2xD3'])
    # Cell features (PCA)
    cell_list = [cell_features.get(cell, np.zeros(50)) for cell in ml['Cell Name']]
    dim_cell = len(cell_list[0]) if cell_list else 50
    cell_mat = np.array(cell_list)
    for i in range(dim_cell):
        col = f'Cell_PC{i + 1}'
        ml[col] = cell_mat[:, i]
        features.append(col)
    # Drug features (Morgan)
    drug_list = [drug_features.get(drug, np.zeros(1024)) for drug in ml['Drug3']]
    dim_drug = len(drug_list[0]) if drug_list else 1024
    drug_mat = np.array(drug_list)
    for i in range(dim_drug):
        col = f'Drug_FP{i + 1}'
        ml[col] = drug_mat[:, i]
        features.append(col)
    X = ml[features].values
    y = ml['Bliss_Synergy'].values
    print(f"Feature matrix: {X.shape}, {len(features)} features")
    return X, y, features, ml
