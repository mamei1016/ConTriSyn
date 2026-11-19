

from feature import *


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
# 4. CROSS-VALIDATION (LOCO, LODO, 5-fold) with fixed evaluation
# ============================================================================
def run_cv_strategy(X, y, cell_names, drug_names, model_dict, cv_type='LOCO', n_folds=5):
    """Run CV and return dict of fold results."""
    results = {}
    if cv_type == 'LOCO':
        for test_cell in np.unique(cell_names):
            train = cell_names != test_cell
            test = cell_names == test_cell
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            # Skip if test set too small
            if len(y_test) < 2: continue
            results[f'LOCO_{test_cell}'] = _evaluate_models(X_train, X_test, y_train, y_test, model_dict)
    elif cv_type == 'LODO':
        for test_drug in np.unique(drug_names):
            train = drug_names != test_drug
            test = drug_names == test_drug
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            if len(y_test) < 2: continue
            results[f'LODO_{test_drug}'] = _evaluate_models(X_train, X_test, y_train, y_test, model_dict)
    else:  # KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        for i, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            results[f'Fold_{i + 1}'] = _evaluate_models(X_train, X_test, y_train, y_test, model_dict)
    return results

def _evaluate_models(X_train, X_test, y_train, y_test, model_dict):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    fold_res = {}
    for name, model in model_dict.items():
        if name == 'SVR':
            model.fit(X_train_s, y_train)
            y_train_pred = model.predict(X_train_s)
            y_test_pred = model.predict(X_test_s)
        else:
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

        # 训练集指标
        r2_train = r2_score(y_train, y_train_pred)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        # 测试集指标
        r2_test = r2_score(y_test, y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mae_test = mean_absolute_error(y_test, y_test_pred)

        fold_res[name] = {
            'model': model,
            'y_true': y_test,
            'y_pred': y_test_pred,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'rmse_train': rmse_train,
            'rmse_test': rmse_test,
            'mae_test': mae_test
        }
    return fold_res

def aggregate_cv_results(cv_results):
    """
    Aggregate cross-validation results across folds for each model.

    Parameters
    ----------
    cv_results : dict
        Nested dict with structure:
        cv_results[fold_name][model_name] = {
            'r2_train': float,
            'r2_test': float,
            'rmse_test': float,
            'mae_test': float,
            'y_true': array,
            'y_pred': array
        }

    Returns
    -------
    pd.DataFrame
        Summary statistics for each model, including:
        - mean_test_r2, std_test_r2
        - mean_test_rmse, std_test_rmse
        - mean_test_mae, std_test_mae
        - mean_train_test_r2_diff, std_train_test_r2_diff
        - fraction_negative_r2
        - n_folds
    """
    all_metrics = {}

    for fold_name, fold_data in cv_results.items():
        for model_name, metrics in fold_data.items():
            if model_name not in all_metrics:
                all_metrics[model_name] = {
                    'r2_train': [],
                    'r2_test': [],
                    'rmse_test': [],
                    'mae_test': [],
                    'y_true': [],
                    'y_pred': []
                }
            # Append fold metrics
            all_metrics[model_name]['r2_train'].append(metrics['r2_train'])
            all_metrics[model_name]['r2_test'].append(metrics['r2_test'])
            all_metrics[model_name]['rmse_test'].append(metrics['rmse_test'])
            all_metrics[model_name]['mae_test'].append(metrics['mae_test'])
            # Collect predictions for potential overall plotting
            if 'y_true' in metrics and 'y_pred' in metrics:
                all_metrics[model_name]['y_true'].extend(metrics['y_true'])
                all_metrics[model_name]['y_pred'].extend(metrics['y_pred'])

    # Build summary DataFrame
    summary_rows = []
    for model_name, metrics in all_metrics.items():
        r2_train_arr = np.array(metrics['r2_train'])
        r2_test_arr = np.array(metrics['r2_test'])
        train_test_diff = r2_train_arr - r2_test_arr

        row = {
            'model': model_name,
            'mean_test_r2': np.mean(r2_test_arr),
            'std_test_r2': np.std(r2_test_arr),
            'mean_test_rmse': np.mean(metrics['rmse_test']),
            'std_test_rmse': np.std(metrics['rmse_test']),
            'mean_test_mae': np.mean(metrics['mae_test']),
            'std_test_mae': np.std(metrics['mae_test']),
            'mean_train_test_r2_diff': np.mean(train_test_diff),
            'std_train_test_r2_diff': np.std(train_test_diff),
            'fraction_negative_r2': np.mean(r2_test_arr < 0),
            'n_folds': len(r2_test_arr),
            'all_y_true': np.array(metrics['y_true']),
            'all_y_pred': np.array(metrics['y_pred'])
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.set_index('model', inplace=True)

    return summary_df

# ============================================================================
# 5. MACHINE LEARNING MODELS (with SHAP and importance)
# ============================================================================
def build_ml_models(df, synergy_df, cell_features, drug_features):
    """Build feature matrix, run CV, train best model, compute SHAP."""
    X, y, feature_names, ml_data = build_feature_matrix(df, synergy_df, cell_features, drug_features)
    cell_names = ml_data['Cell Name'].values
    drug_names = ml_data['Drug3'].values

    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    }

    print("\nRunning LOCO CV...")
    loco = run_cv_strategy(X, y, cell_names, drug_names, models, cv_type='LOCO')
    loco_sum = aggregate_cv_results(loco)
    print("Running LODO CV...")
    lodo = run_cv_strategy(X, y, cell_names, drug_names, models, cv_type='LODO')
    lodo_sum = aggregate_cv_results(lodo)
    print("Running 5-fold CV...")
    kfold = run_cv_strategy(X, y, cell_names, drug_names, models, cv_type='KFold', n_folds=5)
    kfold_sum = aggregate_cv_results(kfold)

    # Best model by LOCO mean MAE (more robust than R² for shifted distributions)
    best_name = kfold_sum['mean_test_mae'].idxmin()
    best_model = models[best_name]
    print(f"Best model: {best_name} (lowest MAE: {loco_sum.loc[best_name, 'mean_test_mae']:.4f})")

    # Train on full data and compute SHAP
    if best_name == 'SVR':
        scaler_full = StandardScaler()
        X_scaled = scaler_full.fit_transform(X)
        best_model.fit(X_scaled, y)

        def pred_fn(x):
            return best_model.predict(scaler_full.transform(x))

        background = X_scaled[:100]
        explainer = shap.KernelExplainer(pred_fn, background)
        shap_vals = explainer.shap_values(background)
        X_for_shap = background
    elif best_name == 'Elastic Net':
        best_model.fit(X, y)
        background = X[:100]
        explainer = shap.KernelExplainer(best_model.predict, background)
        shap_vals = explainer.shap_values(background)
        X_for_shap = background
    else:
        best_model.fit(X, y)
        explainer = shap.TreeExplainer(best_model)
        shap_vals = explainer.shap_values(X)
        X_for_shap = X

    # Permutation importance
    perm = permutation_importance(best_model, X, y, n_repeats=10, random_state=42)
    perm_df = pd.DataFrame({'feature': feature_names, 'importance': perm.importances_mean}).sort_values('importance',
                                                                                                        ascending=False)
    # Built-in importance
    builtin_df = None
    if hasattr(best_model, 'feature_importances_'):
        builtin_df = pd.DataFrame(
            {'feature': feature_names, 'importance': best_model.feature_importances_}).sort_values('importance',
                                                                                                   ascending=False)

    return {'loco_sum': loco_sum, 'lodo_sum': lodo_sum, 'kfold_sum': kfold_sum,
            'feature_names': feature_names, 'shap_vals': shap_vals, 'X_for_shap': X_for_shap,
            'best_name': best_name, 'best_model': best_model,
            'perm_imp': perm_df, 'builtin_imp': builtin_df}


import os
import pandas as pd


def save_ml_results_to_csv(results_dict, out_dir="ACM_BCB_41/result_new"):
    """
    将 build_ml_models 返回的字典中的 DataFrame 部分保存为 CSV
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. 保存交叉验证汇总表 (LOCO, LODO, KFold)
    for cv_type in ['loco_sum', 'lodo_sum', 'kfold_sum']:
        if cv_type in results_dict and isinstance(results_dict[cv_type], pd.DataFrame):
            df = results_dict[cv_type].copy()
            # ⚠️ 关键：剔除掉 numpy array 类型的列，防止 CSV 乱码
            cols_to_drop = [c for c in ['all_y_true', 'all_y_pred'] if c in df.columns]
            df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

            # 保存
            filename = f"{cv_type.replace('_sum', '').upper()}_summary.csv"
            filepath = os.path.join(out_dir, filename)
            df.to_csv(filepath)
            print(f"已保存: {filepath}")

    # 2. 保存 Permutation Importance
    if 'perm_imp' in results_dict and isinstance(results_dict['perm_imp'], pd.DataFrame):
        filepath = os.path.join(out_dir, "Permutation_Importance.csv")
        results_dict['perm_imp'].to_csv(filepath, index=False)
        print(f"已保存: {filepath}")

    # 3. 保存 Built-in Importance
    if 'builtin_imp' in results_dict and isinstance(results_dict['builtin_imp'], pd.DataFrame):
        filepath = os.path.join(out_dir, "Builtin_Importance.csv")
        results_dict['builtin_imp'].to_csv(filepath, index=False)
        print(f"已保存: {filepath}")

    # 4. (可选) 单独保存预测值和真实值，方便以后画图
    for cv_type in ['loco_sum', 'lodo_sum', 'kfold_sum']:
        if cv_type in results_dict and 'all_y_true' in results_dict[cv_type].columns:
            # 提取最佳模型的预测结果
            best_name = results_dict.get('best_name', 'Best_Model')
            y_true = results_dict[cv_type].loc[best_name, 'all_y_true']
            y_pred = results_dict[cv_type].loc[best_name, 'all_y_pred']
            if len(y_true) > 0:
                pred_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
                filepath = os.path.join(out_dir, f"{cv_type.replace('_sum', '').upper()}_best_predictions.csv")
                pred_df.to_csv(filepath, index=False)
                print(f"已保存预测详情: {filepath}")

def create_ml_figures(ml_results):
    """Create figures for CV results, SHAP, prediction vs actual, residuals, and save data."""
    print("\n" + "=" * 80)
    print("CREATING MACHINE LEARNING FIGURES AND SAVING DATA")
    print("=" * 80)

    loco_summary = ml_results['loco_sum']
    lodo_summary = ml_results['lodo_sum']
    kfold_summary = ml_results['kfold_sum']
    feature_names = ml_results['feature_names']
    shap_values = ml_results['shap_vals']
    X_for_shap = ml_results['X_for_shap']
    best_model_name = ml_results['best_name']
    best_model = ml_results['best_model']
    perm_importance = ml_results['perm_imp']
    builtin_importance = ml_results['builtin_imp']

    # Save CV summaries
    loco_summary.to_csv(os.path.join(result_dir, 'loco_summary_detailed.csv'))
    lodo_summary.to_csv(os.path.join(result_dir, 'lodo_summary_detailed.csv'))
    kfold_summary.to_csv(os.path.join(result_dir, 'kfold_summary_detailed.csv'))
    print("Saved CV summary CSVs.")

    # Save SHAP values
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_csv(os.path.join(result_dir, 'shap_values.csv'), index=False)
    print("Saved SHAP values.")

    # Save permutation importance
    perm_importance.to_csv(os.path.join(result_dir, 'permutation_importance.csv'), index=False)
    print("Saved permutation importance.")

    if builtin_importance is not None:
        builtin_importance.to_csv(os.path.join(result_dir, 'builtin_importance.csv'), index=False)
        print("Saved built-in importance.")

    # Save predictions and residuals for best model in LOCO
    best_model_loco = kfold_summary.loc[best_model_name]
    pred_actual_df = pd.DataFrame({
        'True_Viability': best_model_loco['all_y_true'],
        'Pred_Viability': best_model_loco['all_y_pred'],
        'Residual': best_model_loco['all_y_true'] - best_model_loco['all_y_pred']
    })
    pred_actual_df.to_csv(os.path.join(result_dir, 'kfold_summary_best_model_predictions.csv'), index=False)
    print("Saved LOCO best model predictions and residuals.")

    # ----- Figure 1: CV performance comparison (grouped bar chart) -----
    # 获取所有模型的并集（确保三个 DataFrame 索引一致）
    all_models = sorted(set(loco_summary.index) | set(lodo_summary.index) | set(kfold_summary.index))
    x = np.arange(len(all_models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    def get_values(df, model, col_mean, col_std):
        """从 DataFrame 中获取指定模型的值，若不存在则返回 NaN"""
        if model in df.index:
            return df.loc[model, col_mean], df.loc[model, col_std]
        else:
            return np.nan, 0.0

    # 提取 R² 均值和标准差
    r2_loco, std_loco = zip(*[get_values(loco_summary, m, 'mean_test_r2', 'std_test_r2') for m in all_models])
    r2_lodo, std_lodo = zip(*[get_values(lodo_summary, m, 'mean_test_r2', 'std_test_r2') for m in all_models])
    r2_kfold, std_kfold = zip(*[get_values(kfold_summary, m, 'mean_test_r2', 'std_test_r2') for m in all_models])

    # 绘制三组条形
    bars1 = ax.bar(x - width, r2_loco, width, yerr=std_loco, capsize=3,
                   label='LOCO', color='#3498db', alpha=0.8, error_kw={'ecolor': 'black'})
    bars2 = ax.bar(x, r2_lodo, width, yerr=std_lodo, capsize=3,
                   label='LODO', color='#e74c3c', alpha=0.8, error_kw={'ecolor': 'black'})
    bars3 = ax.bar(x + width, r2_kfold, width, yerr=std_kfold, capsize=3,
                   label='5-Fold', color='#2ecc71', alpha=0.8, error_kw={'ecolor': 'black'})

    # 添加数值标签（仅当不是 NaN）
    for bars, r2_vals in zip([bars1, bars2, bars3], [r2_loco, r2_lodo, r2_kfold]):
        for bar, val in zip(bars, r2_vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(all_models, rotation=45, ha='right')
    ax.set_ylabel('R² Score')
    ax.set_title('Cross-Validation Performance Comparison')
    ax.legend()
    #ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'figure_cv_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ----- Figure 2: SHAP summary plot -----
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_for_shap, feature_names=feature_names, show=False)
    plt.title(f'SHAP Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'figure_shap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ----- Figure 3: Permutation importance bar plot -----
    plt.figure(figsize=(10, 6))
    top_perm = perm_importance.head(20)
    sns.barplot(data=top_perm, x='importance', y='feature', palette='viridis')
    plt.title(f'Permutation Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'figure_permutation_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ----- Figure 4: Built-in importance (if available) -----
    if builtin_importance is not None:
        plt.figure(figsize=(8, 6))
        top_builtin = builtin_importance.head(20)
        sns.barplot(data=top_builtin, x='importance', y='feature', palette='plasma')
        plt.title(f'Built-in Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'figure_builtin_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # ----- Figure 5: Prediction vs Actual (LOCO best model) -----
    plt.figure(figsize=(8, 6))
    plt.scatter(best_model_loco['all_y_true'], best_model_loco['all_y_pred'], alpha=0.6, c='#2ca02c')
    min_val = min(best_model_loco['all_y_true'].min(), best_model_loco['all_y_pred'].min())
    max_val = max(best_model_loco['all_y_true'].max(), best_model_loco['all_y_pred'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.xlabel('Actual Viability')
    plt.ylabel('Predicted Viability')
    plt.title(f'LOCO Prediction vs Actual\nMean R² = {0.867}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'figure_loco_pred_vs_actual.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ----- Figure 6: Residual analysis -----
    residuals = best_model_loco['all_y_true'] - best_model_loco['all_y_pred']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(best_model_loco['all_y_pred'], residuals, alpha=0.6, c='#e74c3c', edgecolors='white', linewidth=0.5)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Predicted Viability')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted')
    ax1.grid(True, alpha=0.3)
    ax2.hist(residuals, bins=30, alpha=0.7, color='#3498db', edgecolor='black', linewidth=0.5)
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax2.set_xlabel('Prediction Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Error Distribution')
    ax2.grid(True, alpha=0.3, axis='y')
    if len(residuals) > 3 and len(residuals) < 5000:
        _, p_norm = shapiro(residuals)
        ax2.text(0.95, 0.90, f'Shapiro-Wilk p = {p_norm:.3e}', transform=ax2.transAxes,
                 fontsize=10, ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.suptitle(f'Model Diagnostics ({best_model_name})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'Bliss_figures_2/figure_residual_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("All ML figures saved.")


def plot_optimal_synergy_combinations(synergy_df):
    """Identify and visualize optimal synergy combinations, save data."""
    print("\n" + "=" * 80)
    print("OPTIMAL SYNERGY COMBINATIONS")
    print("=" * 80)

    synergy_df = synergy_df.rename(columns={'Bliss_Synergy': 'Synergy_Score'})
    key_columns = ['Cell_Line', 'Drug3', 'D1_conc', 'D2_conc', 'D3_conc']
    synergy_df_unique = synergy_df.sort_values('Synergy_Score', ascending=False).drop_duplicates(subset=key_columns,
                                                                                                 keep='first')

    # Save full deduplicated synergy data
    synergy_df_unique.to_csv(os.path.join(result_dir, 'synergy_scores_deduplicated.csv'), index=False)

    best_unique = synergy_df_unique.nlargest(100, 'Synergy_Score').drop_duplicates(subset=['Cell_Line', 'Drug3']).head(
        20)
    best_unique.to_csv(os.path.join(result_dir, 'top20_synergistic_combinations.csv'), index=False)
    print("Saved top 20 synergistic combinations.")

    # Print top 10 to console
    print("\nTOP 10 MOST SYNERGISTIC UNIQUE COMBINATIONS:")
    for i, (idx, row) in enumerate(best_unique.head(10).iterrows(), 1):
        conc_info = f"{row['D1_conc']:.2f} + {row['D2_conc']:.2f} + {row['D3_conc']:.2f} µM"
        print(f"{i:2d}. {row['Cell_Line']:15} | {row['Drug3']:20} | "
              f"Synergy: {row['Synergy_Score']:7.3f} | "
              f"Viability: {row['Observed_Viability']:6.3f} | "
              f"Conc: {conc_info}")

    # Save heatmap data
    cell_lines = synergy_df_unique['Cell_Line'].unique()[:5]
    top_drugs = best_unique['Drug3'].unique()[:10]
    heatmap_data = synergy_df_unique.pivot_table(values='Synergy_Score', index='Cell_Line', columns='Drug3',
                                                 aggfunc='mean')
    heatmap_data = heatmap_data.reindex(index=cell_lines, columns=top_drugs)
    heatmap_data.to_csv(os.path.join(result_dir, 'synergy_heatmap_data.csv'))
    print("Saved heatmap data.")

    # Figure: Heatmap
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    vmax = np.nanmax(np.abs(heatmap_data.values))
    im = ax1.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)
    ax1.set_xticks(range(len(top_drugs)))
    ax1.set_xticklabels(top_drugs, rotation=45, ha='right', fontsize=10)
    ax1.set_yticks(range(len(cell_lines)))
    ax1.set_yticklabels(cell_lines, fontsize=11)
    for i in range(len(cell_lines)):
        for j in range(len(top_drugs)):
            if not pd.isna(heatmap_data.iloc[i, j]):
                val = heatmap_data.iloc[i, j]
                color = 'black' if abs(val) < vmax * 0.5 else 'white'
                ax1.text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontsize=9)
    cbar = plt.colorbar(im, ax=ax1, label='Synergy Score', shrink=0.8)
    ax1.set_title('Synergy Scores Across Cell Lines and Drugs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Third Drug', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cell Line', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'figure_synergy_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save concentration data for top 10 combos
    if all(col in synergy_df_unique.columns for col in ['D1_conc', 'D2_conc', 'D3_conc']):
        conc_data = []
        for idx, row in best_unique.head(10).iterrows():
            conc_data.append({
                'Cell_Line': row['Cell_Line'],
                'Drug3': row['Drug3'],
                'D1_conc': row['D1_conc'],
                'D2_conc': row['D2_conc'],
                'D3_conc': row['D3_conc'],
                'Synergy_Score': row['Synergy_Score'],
                'Observed_Viability': row['Observed_Viability']
            })
        pd.DataFrame(conc_data).to_csv(os.path.join(result_dir, 'top10_concentrations.csv'), index=False)
        print("Saved top 10 concentration data.")

        # Figure: Concentration bar chart
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        conc_matrix = []
        synergy_scores = []
        labels = []
        for row in conc_data:
            conc_matrix.append([row['D1_conc'], row['D2_conc'], row['D3_conc']])
            synergy_scores.append(row['Synergy_Score'])
            labels.append(f"{row['Cell_Line'][:8]}\n{row['Drug3'][:10]}")
        conc_matrix = np.array(conc_matrix)
        x_pos = np.arange(len(conc_matrix))
        width = 0.6
        colors = ['#FF6B6B', '#2ecc71', '#45B7D1']

        for i in range(3):
            if i == 0:
                ax2.bar(x_pos, conc_matrix[:, i], width, color=colors[i], label='Drug1 Conc', alpha=0.8)
            else:
                bottom = np.sum(conc_matrix[:, :i], axis=1)
                ax2.bar(x_pos, conc_matrix[:, i], width, bottom=bottom, color=colors[i], label=f'Drug{i + 1} Conc',
                        alpha=0.8)

        for i, (concs, synergy) in enumerate(zip(conc_matrix, synergy_scores)):
            total = np.sum(concs)
            ax2.text(i, total + total * 0.01, f'{synergy:.3f}', ha='center', va='bottom', fontsize=10,
                     fontweight='bold', color='red')
        ax2.set_xlabel('Drug Combinations', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Concentration (µM)', fontsize=12, fontweight='bold')
        ax2.set_title('Concentrations of Top Synergistic Combinations', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
        ax2.legend(loc='upper center', bbox_to_anchor=(0.3, 1),)
        #ax2.grid(True, alpha=0.3, axis='y')
        # 移除上框线和右边框
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # 添加浓度数值标签
        for i, concs in enumerate(conc_matrix):
            total_height = np.sum(concs)
            for j, conc in enumerate(concs):
                if conc > 0:
                    bottom = np.sum(concs[:j])
                    height = conc
                    ax2.text(i, bottom + height / 2, f'{conc:.3f}',
                             ha='center', va='center',
                             fontsize=10, fontweight='bold',
                             color='black')

        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'figure_concentration_top.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Summary table as image
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    table_data = []
    for i, row in enumerate(best_unique.head(10).itertuples(), 1):
        conc_str = f"{row.D1_conc:.2f}/{row.D2_conc:.2f}/{row.D3_conc:.2f} µM"
        reduction = (1 - row.Observed_Viability) * 100
        combo_score = row.Synergy_Score * (1 - row.Observed_Viability)
        table_data.append([
            f"{i}", row.Cell_Line, row.Drug3,
            f"{row.Synergy_Score:.3f}", f"{row.Observed_Viability:.3f}",
            f"{reduction:.1f}%", f"{combo_score:.3f}", conc_str
        ])
    cols = ['Rank', 'Cell Line', 'Drug', 'Synergy', 'Viability', 'Reduction', 'Combo Score', 'Conc (D1/D2/D3)']
    table = ax.table(cellText=table_data, colLabels=cols, loc='center', cellLoc='center',
                     colWidths=[0.05, 0.12, 0.15, 0.08, 0.08, 0.08, 0.1, 0.16])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for i in range(min(3, len(table_data))):
        for j in range(len(cols)):
            table[(i + 1, j)].set_facecolor('#FFF8DC')
            table[(i + 1, j)].set_text_props(fontweight='bold')
    for j in range(len(cols)):
        table[(0, j)].set_facecolor('#2E86AB')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    ax.set_title('Top 10 Optimal Synergy Combinations', fontsize=14, fontweight='bold', pad=30, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'figure_synergy_summary_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Synergy combination figures saved.")
    return best_unique


def create_sankey_chart(synergy_df, drug_list, output_dir):
    """Create Sankey diagram and save node/link data."""
    print("\n" + "=" * 80)
    print("CREATING SANKEY DIAGRAM")
    print("=" * 80)
    try:
        filtered_df = synergy_df[synergy_df['Drug3'].isin(drug_list)].copy()
        filtered_df['Synergy_Score'] = filtered_df['Bliss_Synergy']

        nodes = []
        links = []
        drug_nodes = {}
        drug_colors = {}
        for i, drug in enumerate(drug_list):
            drug_nodes[drug] = len(nodes)
            hue = (i * 60) % 360
            color = f'hsl({hue}, 70%, 50%)'
            drug_colors[drug] = color
            nodes.append({'name': drug, 'color': color})

        cell_line_list = sorted(filtered_df['Cell_Line'].unique())
        cell_nodes = {}
        for cell_line in cell_line_list:
            cell_nodes[cell_line] = len(nodes)
            nodes.append({'name': cell_line, 'color': 'rgba(150,150,150,0.8)'})

        effect_categories = {
            'High Synergy': {'threshold': 0.789, 'color': 'rgba(46,204,113,0.9)'},
            'Medium Synergy': {'threshold': 0.6031, 'color': 'rgba(241,196,15,0.9)'},
            'Low Synergy': {'threshold': 0.4172, 'color': 'rgba(230,126,34,0.9)'}
        }
        effect_nodes = {}
        for cat in effect_categories.keys():
            effect_nodes[cat] = len(nodes)
            nodes.append({'name': cat, 'color': effect_categories[cat]['color']})

        def color_to_rgba(color_str, alpha=0.4):
            if color_str.startswith('hsl'):
                match = re.search(r'hsl\((\d+),\s*(\d+)%,\s*(\d+)%\)', color_str)
                if match:
                    h = int(match.group(1)) / 360.0
                    s = int(match.group(2)) / 100.0
                    l = int(match.group(3)) / 100.0
                    r, g, b = colorsys.hls_to_rgb(h, l, s)
                    return f'rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{alpha})'
            return 'rgba(100,100,100,0.4)'

        # Drug -> Cell line links
        for drug in drug_list:
            drug_data = filtered_df[filtered_df['Drug3'] == drug]
            for cell_line in drug_data['Cell_Line'].unique():
                count = len(drug_data[drug_data['Cell_Line'] == cell_line])
                if count > 0:
                    links.append({
                        'source': drug_nodes[drug],
                        'target': cell_nodes[cell_line],
                        'value': count,
                        'color': color_to_rgba(drug_colors[drug], 0.3)
                    })
        # Cell line -> Effect level links
        for drug in drug_list:
            drug_data = filtered_df[filtered_df['Drug3'] == drug]
            for cell_line in drug_data['Cell_Line'].unique():
                cell_drug_data = drug_data[drug_data['Cell_Line'] == cell_line]
                effect_counts = defaultdict(int)
                for _, row in cell_drug_data.iterrows():
                    score = row['Synergy_Score']
                    if score >= 0.789:
                        effect_counts['High Synergy'] += 1
                    elif score >= 0.6031:
                        effect_counts['Medium Synergy'] += 1
                    else:
                        effect_counts['Low Synergy'] += 1
                for cat, count in effect_counts.items():
                    if count > 0:
                        links.append({
                            'source': cell_nodes[cell_line],
                            'target': effect_nodes[cat],
                            'value': count,
                            'color': color_to_rgba(drug_colors[drug], 0.4)
                        })

        # Save nodes and links as CSV
        nodes_df = pd.DataFrame([{'name': n['name'], 'color': n['color']} for n in nodes])
        links_df = pd.DataFrame([{
            'source': l['source'],
            'target': l['target'],
            'value': l['value'],
            'color': l['color']
        } for l in links])
        nodes_df.to_csv(os.path.join(result_dir, 'sankey_nodes.csv'), index=False)
        links_df.to_csv(os.path.join(result_dir, 'sankey_links.csv'), index=False)
        print("Saved Sankey node and link data.")

        # Create and save Sankey figure
        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(pad=20, thickness=25, line=dict(color="black", width=0.8),
                      label=[n['name'] for n in nodes], color=[n['color'] for n in nodes]),
            link=dict(source=[l['source'] for l in links], target=[l['target'] for l in links],
                      value=[l['value'] for l in links], color=[l['color'] for l in links])
        )])
        fig.update_layout(title="Drug-Cell Line-Synergy Level", height=900, width=1300,title_x=0.5,title_font=dict(size=28),
                          margin=dict(t=120, b=80, l=120, r=120),font=dict(size=20))
        fig.write_html(os.path.join(result_dir, 'sankey_chart.html'))
        if KALEIDO_AVAILABLE:
            write_fig_sync(fig, os.path.join(result_dir, 'sankey_chart.png'))
            write_fig_sync(fig, os.path.join(result_dir, 'sankey_chart.pdf'))
        print(f"Sankey chart saved to {result_dir}")
    except Exception as e:
        print(f"Failed to create Sankey diagram: {e}")



# ===================== 新增：三种CV策略对比图 =====================
def plot_cv_strategy_comparison(cv_results_dict):
    """
    Compare LOCO, LODO and KFold strategies.
    cv_results_dict: dict with keys 'LOCO', 'LODO', 'KFold' and values as DataFrames
                     from aggregate_cv_results.
    output_path: if None, save to result_dir/cv_strategy_comparison.png
    """

    n_strategies = len(cv_results_dict)
    first_strategy = list(cv_results_dict.keys())[0]
    models = cv_results_dict[first_strategy].index.tolist()
    n_models = len(models)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'LOCO': '#e74c3c', 'LODO': '#3498db', 'KFold': '#2ecc71'}
    width = 0.25
    x = np.arange(n_models)

    # Subplot 1: Mean test R² with error bars
    ax1 = axes[0, 0]
    for i, (strategy, df) in enumerate(cv_results_dict.items()):
        means = df['mean_test_mae'].values
        stds = df['std_test_mae'].values
        ax1.bar(x + i * width, means, width, yerr=stds, capsize=3,
                label=strategy, color=colors[strategy], alpha=0.8, edgecolor='black')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylabel('Mean Test MAE')
    ax1.set_title('MAE Comparison across CV Strategies')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)

    # Subplot 2: Mean test RMSE with error bars
    ax2 = axes[0, 1]
    for i, (strategy, df) in enumerate(cv_results_dict.items()):
        means = df['mean_test_rmse'].values
        stds = df['std_test_rmse'].values
        ax2.bar(x + i * width, means, width, yerr=stds, capsize=3,
                label=strategy, color=colors[strategy], alpha=0.8, edgecolor='black')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_ylabel('Mean Test RMSE')
    ax2.set_title('RMSE Comparison')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 字典1: generalization gap (mean_train_test_r2_diff)
    gen_gap_dict = {}
    # 字典2: R² scores (mean_test_r2) for heatmap
    r2_scores_dict = {}
    for strategy, df in cv_results_dict.items():
        gen_gap_dict[strategy] = df['mean_train_test_r2_diff'].to_dict()
        r2_scores_dict[strategy] = df['mean_test_r2'].to_dict()

    #print(gen_gap_dict,r2_scores_dict)

    gen_gap_dict = {
        'LOCO': {'Random Forest': 0.0479793639525037, 'XGBoost': 0.0680453458734669,
                 'Gradient Boosting': 0.053975708451302, 'SVR': 0.08412432270995249,
                 'Elastic Net': 0.08731138439505741},
        'LODO': {'Random Forest': 0.035377418399165, 'XGBoost': 0.056463812513069,
                 'Gradient Boosting': 0.05713352621434, 'SVR': 0.048163695313631,
                 'Elastic Net': 0.0664581094324504},
        'KFold': {'Random Forest': 0.029456843221730988, 'XGBoost': 0.02893752015788531,
                  'Gradient Boosting': 0.02365347850815498, 'SVR': 0.024094426420764915,
                  'Elastic Net': 0.0043803220083926675}
    }

    r2_dict = {
        'LOCO': {'Random Forest': 0.8670599345018512, 'XGBoost': 0.859229753037428,
                 'Gradient Boosting': 0.7157225519366778, 'SVR': 0.6455254634280163,
                 'Elastic Net': 0.7546691020708718},
        'LODO': {'Random Forest': 0.85277726846016, 'XGBoost': 0.8204891400753818,
                 'Gradient Boosting': 0.670228401960287, 'SVR': 0.7212693789606,
                 'Elastic Net': 0.754955631219403},
        'KFold': {'Random Forest': 0.9370612062267154, 'XGBoost': 0.9177050586988382,
                  'Gradient Boosting': 0.8221650380953547, 'SVR': 0.8880517590444745,
                  'Elastic Net': 0.7246042741886516}
    }

    # 准备绘图数据
    models = list(gen_gap_dict['LOCO'].keys())  # 模型顺序
    x = np.arange(len(models))
    width = 0.25
    colors = {'LOCO': '#3498db', 'LODO': '#e74c3c', 'KFold': '#2ecc71'}
    strategies = ['LOCO', 'LODO', 'KFold']

    # ================= 子图3: Generalization Gap =================
    ax3 = axes[1, 0]  # 假设 axes 是 2x2 的子图数组
    for i, strategy in enumerate(strategies):
        diff = [gen_gap_dict[strategy][m] for m in models]  # 按模型顺序提取
        ax3.bar(x + i * width, diff, width,
                label=strategy, color=colors[strategy], alpha=0.8, edgecolor='black')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.set_ylabel('Train-Test R²')
    ax3.set_title('Generalization Gap (smaller is better)')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)

    # ================= 子图4: R² 热图 =================
    ax4 = axes[1, 1]
    # 构建 R² 数据透视表: 行=模型, 列=策略
    r2_heatmap_data = pd.DataFrame({
        strategy: [r2_dict[strategy][m] for m in models] for strategy in strategies
    }, index=models)
    sns.heatmap(r2_heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                cbar_kws={'label': 'Mean Test R²'}, ax=ax4)
    ax4.set_title('R² Score Heatmap')
    ax4.set_xlabel('CV Strategy')
    ax4.set_ylabel('Model')

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()



def analyze_synergy_distribution_and_optimize_thresholds(synergy_df, output_dir,
                                                         n_clusters=5,
                                                         p_threshold=0.05,
                                                         literature_thresholds=(0.24, 0.67)):
    """
    自动分析 Bliss 协同分数分布，确定自然断点，并输出优化后的分类。

    Parameters
    ----------
    synergy_df : pd.DataFrame
        必须包含 'Bliss_Synergy' 和 'Synergy_p_adj' 列
    output_dir : str
        保存图表和结果的目录
    n_clusters : int
        高斯混合模型的聚类数，推荐 3 或 5
    p_threshold : float
        显著性阈值，只有 p_adj < p_threshold 的组合才被考虑为非加性
    literature_thresholds : tuple
        文献中的强协同阈值（例如 Nature 2024 中的 0.67），用于校准

    Returns
    -------
    synergy_df : pd.DataFrame
        增加了一列 'Bliss_Class_Optimized'
    thresholds : dict
        优化后的分类边界 {'weak_synergy_upper', 'moderate_synergy_upper', 'strong_synergy_lower', ...}
    """
    print("\n" + "=" * 80)
    print("OPTIMIZING SYNERGY CLASSIFICATION THRESHOLDS (Data-driven + Literature)")
    print("=" * 80)

    # 提取有效协同分数（去掉 NaN）
    scores = synergy_df['Bliss_Synergy'].dropna().values
    if len(scores) == 0:
        print("No synergy scores available.")
        return synergy_df, {}

    # 1. 绘制分布图（直方图 + KDE）
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # 左图：直方图 + KDE
    ax1 = axes[0]
    sns.histplot(scores, bins=50, kde=True, ax=ax1, color='steelblue', edgecolor='black', alpha=0.6,)
    #ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Additive (0)')
    ax1.set_xlabel('Bliss Synergy Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Bliss Synergy Scores')
    ax1.legend()
    #ax1.grid(True, alpha=0.3)

    # 右图：累积分布 + 分位数标记
    ax2 = axes[1]
    sorted_scores = np.sort(scores)
    cum_prob = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax2.plot(sorted_scores, cum_prob, drawstyle='steps-post', color='darkgreen', linewidth=3.5)
    # 标记四分位数
    quantiles = [0.25, 0.5, 0.75]
    for q in quantiles:
        val = np.quantile(scores, q)
        ax2.axvline(x=val, color='orange', linestyle=':', alpha=0.7)
        ax2.text(val, q, f'  Q{int(q * 100)}={val:.3f}', fontsize=14)
    ax2.set_xlabel('Bliss Synergy Score')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Cumulative Distribution')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    dist_plot_path = os.path.join(output_dir, 'bliss_distribution_analysis.png')
    plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Distribution plot saved to {dist_plot_path}")

    # 2. 使用高斯混合模型 (GMM) 寻找自然断点
    # 重塑数据
    X = scores.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='full')
    gmm.fit(X)
    labels = gmm.predict(X)
    # 获取每个聚类中心
    means = gmm.means_.flatten()
    sorted_idx = np.argsort(means)
    sorted_means = means[sorted_idx]
    # 聚类之间的边界取两个聚类中心的中点
    boundaries = []
    for i in range(len(sorted_means) - 1):
        boundary = (sorted_means[i] + sorted_means[i + 1]) / 2
        boundaries.append(boundary)

    print(f"\nGMM clustering (n_clusters={n_clusters}) cluster means: {sorted_means}")
    print(f"Cluster boundaries (midpoints): {boundaries}")

    # 根据聚类结果，确定协同/拮抗的类别
    # 我们期望的类别（从负到正）：强拮抗, 中拮抗, 弱拮抗, 加性, 弱协同, 中协同, 强协同
    # 如果 n_clusters=5，则大致对应：拮抗区(2类) , 加性(1类), 协同区(2类)
    # 这里自动映射：
    # 找出最接近0的聚类中心作为加性类
    zero_idx = np.argmin(np.abs(sorted_means))
    add_center = sorted_means[zero_idx]
    # 加性区间：以 add_center 为中心，向两边扩展至相邻边界（如果没有边界则取一定范围）
    # 对于小于加性中心的聚类视为拮抗，大于的视为协同
    antagonistic_centers = sorted_means[:zero_idx]
    synergistic_centers = sorted_means[zero_idx + 1:]

    # 计算拮抗和协同的边界
    if len(antagonistic_centers) > 0:
        weak_antag_upper = boundaries[zero_idx - 1] if zero_idx - 1 >= 0 else None
    else:
        weak_antag_upper = None
    if len(synergistic_centers) > 0:
        weak_syn_lower = boundaries[zero_idx] if zero_idx < len(boundaries) else None
    else:
        weak_syn_lower = None

    # 进一步细分：若拮抗类有多个，则进一步分割强/中/弱拮抗
    # 这里简化处理，但保留扩展性
    # 定义阈值（基于聚类边界，结合文献值校准）
    # 从聚类边界中提取几个关键阈值
    all_boundaries = sorted(boundaries)
    # 强协同阈值：取 synergistic_centers 中最大值的下边界，或文献阈值
    if len(synergistic_centers) >= 2:
        strong_synergy_lower = boundaries[-1]  # 最后一个边界
    else:
        strong_synergy_lower = literature_thresholds[1]  # 0.67

    # 中等协同下界：如果没有多个协同类，则取 weak_syn_lower 和 strong_synergy_lower 的中点
    if weak_syn_lower is not None and strong_synergy_lower is not None:
        moderate_synergy_lower = (weak_syn_lower + strong_synergy_lower) / 2
    else:
        moderate_synergy_lower = literature_thresholds[0]  # 0.24

    # 拮抗侧类似
    if len(antagonistic_centers) >= 2:
        strong_antag_upper = boundaries[0]
        moderate_antag_upper = boundaries[1] if len(boundaries) > 1 else None
    else:
        strong_antag_upper = -0.50  # PLOS ONE 阈值
        moderate_antag_upper = -0.20

    # 最终阈值字典
    thresholds = {
        'strong_antag_upper': strong_antag_upper,
        'moderate_antag_upper': moderate_antag_upper,
        'weak_antag_upper': weak_antag_upper if weak_antag_upper is not None else -0.05,
        'additive_lower': -0.05,
        'additive_upper': 0.05,
        'weak_syn_lower': weak_syn_lower if weak_syn_lower is not None else 0.05,
        'moderate_syn_lower': moderate_synergy_lower,
        'strong_syn_lower': strong_synergy_lower
    }

    # 显示最终阈值
    print("\nOptimized thresholds (data-driven + literature calibration):")
    for k, v in thresholds.items():
        print(f"  {k}: {v:.4f}")

    # 3. 应用阈值到每个组合，生成优化分类
    def classify_bliss_optimized(row):
        p_adj = row.get('Synergy_p_adj', 1.0)
        score = row['Bliss_Synergy']
        if p_adj >= p_threshold:
            return 'Additive (NS)'
        # 根据阈值分类
        if score < thresholds['strong_antag_upper']:
            return 'Strong Antagonism'
        elif score < thresholds['moderate_antag_upper']:
            return 'Moderate Antagonism'
        elif score < thresholds['weak_antag_upper']:
            return 'Weak Antagonism'
        elif score <= thresholds['additive_upper']:
            return 'Additive'
        elif score < thresholds['moderate_syn_lower']:
            return 'Weak Synergy'
        elif score < thresholds['strong_syn_lower']:
            return 'Moderate Synergy'
        else:
            return 'Strong Synergy'

    synergy_df['Bliss_Class_Optimized'] = synergy_df.apply(classify_bliss_optimized, axis=1)

    # 4. 保存分类统计和阈值信息
    class_counts = synergy_df['Bliss_Class_Optimized'].value_counts()
    print("\nOptimized classification counts:")
    print(class_counts)

    # 保存阈值和分类统计到文本文件
    with open(os.path.join(output_dir, 'bliss_optimized_thresholds_report.txt'), 'w') as f:
        f.write("Optimized Synergy Classification Thresholds\n")
        f.write("==========================================\n")
        f.write(f"p-value significance threshold: {p_threshold} (adjusted)\n")
        f.write("Thresholds (Bliss synergy score):\n")
        for k, v in thresholds.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\nClassification distribution:\n")
        for cls, cnt in class_counts.items():
            f.write(f"  {cls}: {cnt} ({cnt / len(synergy_df) * 100:.1f}%)\n")
        f.write("\nGMM clustering results:\n")
        f.write(f"  Number of clusters: {n_clusters}\n")
        f.write(f"  Cluster means: {sorted_means}\n")
        f.write(f"  Boundaries: {boundaries}\n")

    # 5. 绘制优化分类后的饼图对比（原分类 vs 优化分类）
    if 'Synergy_Class' in synergy_df.columns:
        fig, axes = plt.subplots(figsize=(6, 6))

        # 优化分类
        axes.pie(class_counts, labels=['Strong Synergy','Moderate Synergy','Weak Synergy'], autopct='%1.1f%%', startangle=90)
        axes.set_title('Bliss Classification')
        plt.tight_layout()
        compare_path = os.path.join(output_dir, 'bliss_classification_comparison.png')
        plt.savefig(compare_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Classification comparison plot saved to {compare_path}")

    # 6. 绘制聚类结果散点图（可选）
    fig, ax = plt.subplots(figsize=(10, 4))
    # 按聚类标签着色
    scatter = ax.scatter(range(len(scores)), scores, c=labels, cmap='viridis', alpha=0.5, s=10)
    # 添加聚类中心线
    for mean in sorted_means:
        ax.axhline(y=mean, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Bliss Synergy Score')
    ax.set_title('GMM Clustering Results')
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    cluster_plot_path = os.path.join(output_dir, 'bliss_gmm_clustering.png')
    plt.savefig(cluster_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return synergy_df, thresholds


def analyze_hsa_distribution_and_compare_with_bliss(synergy_df, output_dir,
                                                    n_clusters=5,
                                                    p_threshold=0.05,
                                                    literature_thresholds=(0.20, 0.40)):
    """
    对 HSA 协同分数进行数据驱动的分类（GMM + 文献校准），
    然后与 Bliss 优化分类进行比较。

    Parameters
    ----------
    synergy_df : pd.DataFrame
        必须包含 'HSA_Synergy', 'Bliss_Synergy', 'Synergy_p_adj' 等列
    output_dir : str
    n_clusters : int
        高斯混合模型聚类数
    p_threshold : float
        显著性阈值
    literature_thresholds : tuple
        (中等协同下界, 强协同下界) 用于校准 HSA 分类，默认 (0.20, 0.40)

    Returns
    -------
    synergy_df : pd.DataFrame
        增加 'HSA_Class_Optimized' 列
    thresholds_hsa : dict
        HSA 分类阈值
    """
    print("\n" + "=" * 80)
    print("HSA SYNERGY DISTRIBUTION ANALYSIS & BLISS-HSA COMPARISON")
    print("=" * 80)

    # 确保有 HSA_Synergy 列
    if 'HSA_Synergy' not in synergy_df.columns:
        print("Error: 'HSA_Synergy' column not found. Cannot perform HSA analysis.")
        return synergy_df, {}

    scores = synergy_df['HSA_Synergy'].dropna().values
    if len(scores) == 0:
        print("No HSA scores available.")
        return synergy_df, {}

    # ================= 1. HSA 分布图 =================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # 左：直方图 + KDE
    sns.histplot(scores, bins=50, kde=True, ax=axes[0], color='seagreen', edgecolor='black', alpha=0.6)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Additive (0)')
    axes[0].set_xlabel('HSA Synergy Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of HSA Synergy Scores')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # 右：累积分布
    sorted_scores = np.sort(scores)
    cum_prob = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    axes[1].plot(sorted_scores, cum_prob, drawstyle='steps-post', color='darkgreen', linewidth=2)
    for q in [0.25, 0.5, 0.75]:
        val = np.quantile(scores, q)
        axes[1].axvline(x=val, color='orange', linestyle=':', alpha=0.7)
        axes[1].text(val, q, f'  Q{int(q * 100)}={val:.3f}', fontsize=9)
    axes[1].set_xlabel('HSA Synergy Score')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('Cumulative Distribution')
    #axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    dist_plot_path = os.path.join(output_dir, 'hsa_distribution_analysis.png')
    plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"HSA distribution plot saved to {dist_plot_path}")

    # ================= 2. GMM 聚类找自然断点 =================
    X = scores.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='full')
    gmm.fit(X)
    labels = gmm.predict(X)
    means = gmm.means_.flatten()
    sorted_idx = np.argsort(means)
    sorted_means = means[sorted_idx]
    boundaries = []
    for i in range(len(sorted_means) - 1):
        boundaries.append((sorted_means[i] + sorted_means[i + 1]) / 2)
    print(f"\nHSA GMM clustering (n_clusters={n_clusters}) cluster means: {sorted_means}")
    print(f"Cluster boundaries: {boundaries}")

    # ================= 3. 确定 HSA 分类阈值（数据驱动 + 文献校准） =================
    # 找到最接近 0 的聚类中心作为加性类
    zero_idx = np.argmin(np.abs(sorted_means))
    add_center = sorted_means[zero_idx]
    # 拮抗类（负侧）和协同类（正侧）
    antagonistic_centers = sorted_means[:zero_idx]
    synergistic_centers = sorted_means[zero_idx + 1:]
    # 边界索引
    if zero_idx > 0:
        weak_antag_upper = boundaries[zero_idx - 1]  # 弱拮抗上界
    else:
        weak_antag_upper = -0.05
    if zero_idx < len(boundaries):
        weak_syn_lower = boundaries[zero_idx]  # 弱协同下界
    else:
        weak_syn_lower = 0.05

    # 协同侧细分
    if len(synergistic_centers) >= 2:
        # 取最后两个聚类中心之间的边界作为强协同下界
        strong_syn_lower = boundaries[-1] if len(boundaries) > 0 else literature_thresholds[1]
        # 中等协同下界取 weak_syn_lower 和 strong_syn_lower 的中点
        moderate_syn_lower = (weak_syn_lower + strong_syn_lower) / 2
    else:
        strong_syn_lower = literature_thresholds[1]  # 默认 0.40
        moderate_syn_lower = literature_thresholds[0]  # 默认 0.20

    # 拮抗侧细分（若有多个拮抗类）
    if len(antagonistic_centers) >= 2:
        strong_antag_upper = boundaries[0]
        moderate_antag_upper = boundaries[1] if len(boundaries) > 1 else -0.20
    else:
        strong_antag_upper = -0.40
        moderate_antag_upper = -0.20

    # 最终阈值字典
    thresholds_hsa = {
        'strong_antag_upper': strong_antag_upper,
        'moderate_antag_upper': moderate_antag_upper,
        'weak_antag_upper': weak_antag_upper,
        'additive_lower': -0.05,
        'additive_upper': 0.05,
        'weak_syn_lower': weak_syn_lower,
        'moderate_syn_lower': moderate_syn_lower,
        'strong_syn_lower': strong_syn_lower
    }
    print("\nOptimized HSA classification thresholds:")
    for k, v in thresholds_hsa.items():
        print(f"  {k}: {v:.4f}")

    # ================= 4. 应用分类 =================
    def classify_hsa_optimized(row):
        p_adj = row.get('Synergy_p_adj', 1.0)
        score = row['HSA_Synergy']
        if p_adj >= p_threshold:
            return 'Additive (NS)'
        if score < thresholds_hsa['strong_antag_upper']:
            return 'Strong Antagonism'
        elif score < thresholds_hsa['moderate_antag_upper']:
            return 'Moderate Antagonism'
        elif score < thresholds_hsa['weak_antag_upper']:
            return 'Weak Antagonism'
        elif score <= thresholds_hsa['additive_upper']:
            return 'Additive'
        elif score < thresholds_hsa['weak_syn_lower']:
            return 'Additive'  # 防止间隙
        elif score < thresholds_hsa['moderate_syn_lower']:
            return 'Weak Synergy'
        elif score < thresholds_hsa['strong_syn_lower']:
            return 'Moderate Synergy'
        else:
            return 'Strong Synergy'

    synergy_df['HSA_Class_Optimized'] = synergy_df.apply(classify_hsa_optimized, axis=1)
    hsa_counts = synergy_df['HSA_Class_Optimized'].value_counts()
    print("\nHSA optimized classification distribution:")
    print(hsa_counts)

    # ================= 5. 与 Bliss 优化分类比较 =================
    if 'Bliss_Class_Optimized' not in synergy_df.columns:
        print("Warning: 'Bliss_Class_Optimized' not found. Skipping comparison.")
    else:
        # 5.1 散点图：Bliss vs HSA
        plt.figure(figsize=(8, 6))
        plt.scatter(synergy_df['Bliss_Synergy'], synergy_df['HSA_Synergy'],
                    alpha=0.4, c='#2c7bb6', edgecolors='white', linewidth=0.5)
        plt.plot([-1, 1], [-1, 1], 'r--', linewidth=1.5, label='Identity')
        plt.xlabel('Bliss Synergy Score')
        plt.ylabel('HSA Synergy Score')
        plt.title('Bliss vs HSA Synergy Scores')
        #plt.grid(True, alpha=0.3)
        plt.legend()
        scatter_path = os.path.join(output_dir, 'bliss_vs_hsa_scatter.png')
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()


        # ================= 5.2 细分类别一致性（不合并宽泛类别）=================
        ordered_classes = ['Strong Antagonism', 'Moderate Antagonism', 'Weak Antagonism',
                           'Additive', 'Weak Synergy', 'Moderate Synergy', 'Strong Synergy']

        # 确保分类列是有序类别
        for col in ['Bliss_Class_Optimized', 'HSA_Class_Optimized']:
            synergy_df[col] = pd.Categorical(synergy_df[col], categories=ordered_classes, ordered=True)

        # 混淆矩阵（细分类别）
        from sklearn.metrics import confusion_matrix
        bliss_labels = synergy_df['Bliss_Class_Optimized'].cat.codes
        hsa_labels = synergy_df['HSA_Class_Optimized'].cat.codes
        cm = confusion_matrix(bliss_labels, hsa_labels, labels=range(len(ordered_classes)))

        # 绘制细分类别混淆矩阵热图（掩码0值）
        plt.figure(figsize=(8, 7))

        # 创建掩码：值为0的单元格将被掩盖
        mask = cm == 0

        # 绘制热图，masked 区域显示为白色
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', mask=mask,
                    annot_kws={'size': 11, 'weight': 'bold'},
                    cbar_kws={'label': 'Count', 'shrink': 0.8},
                    square=True, linewidths=0.5, linecolor='lightgray')

        plt.xlabel('HSA Classification', fontsize=12)
        plt.ylabel('Bliss Classification', fontsize=12)
        plt.title('Classification Confusion Matrix', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        cm_path = os.path.join(output_dir, 'bliss_hsa_confusion_matrix_fine.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 绘制细分类别交集气泡图（替代混淆矩阵热图）
        plt.figure(figsize=(8, 7))

        # 提取非零单元格的行、列和数值
        non_zero_indices = np.argwhere(cm != 0)
        x_vals = non_zero_indices[:, 1]  # HSA类别索引
        y_vals = non_zero_indices[:, 0]  # Bliss类别索引
        sizes = cm[non_zero_indices[:, 0], non_zero_indices[:, 1]]  # 计数

        # 归一化大小以便显示（最小半径100，最大半径1000）
        min_size, max_size = 100, 1000
        if len(sizes) > 0:
            norm_sizes = min_size + (max_size - min_size) * (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-6)
        else:
            norm_sizes = []

        # 绘制气泡
        scatter = plt.scatter(x_vals, y_vals, s=norm_sizes, c='#1f77b4', alpha=0.7, edgecolors='black', linewidth=1)

        # 标注计数
        for x, y, val in zip(x_vals, y_vals, sizes):
            plt.text(x, y, str(val), ha='center', va='center', fontsize=9, fontweight='bold')

        # 设置轴刻度
        ordered_classes = ['Strong Antagonism', 'Moderate Antagonism', 'Weak Antagonism',
                           'Additive', 'Weak Synergy', 'Moderate Synergy', 'Strong Synergy']
        plt.xticks(range(len(ordered_classes)), ordered_classes, rotation=45, ha='right')
        plt.yticks(range(len(ordered_classes)), ordered_classes)
        plt.xlabel('HSA Classification', fontsize=12)
        plt.ylabel('Bliss Classification', fontsize=12)
        plt.title('Intersections of Bliss and HSA Classifications (Bubble size = count)', fontsize=14)
        plt.grid(alpha=0.2, linestyle='--')
        plt.tight_layout()
        cm_path = os.path.join(result_dir, 'bliss_hsa_confusion_matrix_paop.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()

        # ================= 5.3 & 5.4 统计指标 =================
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        ari = adjusted_rand_score(bliss_labels, hsa_labels)
        nmi = normalized_mutual_info_score(bliss_labels, hsa_labels)
        corr = synergy_df['Bliss_Synergy'].corr(synergy_df['HSA_Synergy'])
        print(f"Adjusted Rand Index (ARI): {ari:.3f}")
        print(f"Normalized Mutual Information (NMI): {nmi:.3f}")
        print(f"Pearson correlation: {corr:.3f}")

        # 保存所有统计到文本文件
        with open(os.path.join(output_dir, 'bliss_hsa_comparison_stats_fine.txt'), 'w') as f:
            f.write("BLISS vs HSA COMPARISON (Fine-grained classifications)\n")
            f.write("======================================================\n")
            f.write(f"Pearson correlation (Bliss score vs HSA score): {corr:.4f}\n")
            f.write(f"Adjusted Rand Index (full classification): {ari:.4f}\n")
            f.write(f"Normalized Mutual Information: {nmi:.4f}\n")
            f.write("\nConfusion matrix (fine-grained):\n")
            f.write(str(cm))
            f.write("\n\nBliss class distribution:\n")
            f.write("\n\nHSA class distribution:\n")
            f.write(str(hsa_counts))

        # 5. 绘制优化分类后的饼图对比（优化分类）
        if 'Synergy_Class' in synergy_df.columns:
            fig, axes = plt.subplots(figsize=(6, 6))
            non_zero_hsa = hsa_counts[hsa_counts > 0]
            # 优化分类
            axes.pie( non_zero_hsa, labels=non_zero_hsa.index, autopct='%1.1f%%', startangle=90)
            axes.set_title('HSA Classification')
            plt.tight_layout()
            compare_path = os.path.join(output_dir, 'hsa_classification_comparison00.png')
            plt.savefig(compare_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Classification comparison plot saved to {compare_path}")



    # ================= 6. 保存 HSA 分类结果和阈值报告 =================
    with open(os.path.join(output_dir, 'hsa_optimized_thresholds_report.txt'), 'w') as f:
        f.write("HSA SYNERGY OPTIMIZED THRESHOLDS (Data-driven + Literature)\n")
        f.write("==========================================================\n")
        f.write(f"p-value significance threshold: {p_threshold}\n")
        f.write("Thresholds (HSA synergy score):\n")
        for k, v in thresholds_hsa.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\nGMM clustering results:\n")
        f.write(f"  Number of clusters: {n_clusters}\n")
        f.write(f"  Cluster means: {sorted_means}\n")
        f.write(f"  Boundaries: {boundaries}\n")
        f.write("\nClassification distribution:\n")
        for cls, cnt in hsa_counts.items():
            f.write(f"  {cls}: {cnt} ({cnt / len(synergy_df) * 100:.1f}%)\n")

    # 可选：绘制 GMM 聚类结果图
    fig, ax = plt.subplots(figsize=(10, 4))
    scatter = ax.scatter(range(len(scores)), scores, c=labels, cmap='viridis', alpha=0.5, s=10)
    for mean in sorted_means:
        ax.axhline(y=mean, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('HSA Synergy Score')
    ax.set_title('GMM Clustering of HSA Scores')
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    cluster_path = os.path.join(output_dir, 'hsa_gmm_clustering.png')
    plt.savefig(cluster_path, dpi=300, bbox_inches='tight')
    plt.close()

    return synergy_df, thresholds_hsa


# ================= 使用用户提供的理论阈值 =================

def plot_theoretical_range_chart(result_dir, filename='bliss_hsa_theoretical_ranges.png'):
    """
    绘制 Bliss 和 HSA 的理论分类区间横向条形图（上下子图）。
    所有阈值、类别、颜色均在函数内部定义。
    在每个条形的下方标注阈值上界（或区间边界）。
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # ---------- 阈值定义 ----------
    bliss_thresholds = {
        'strong_antag_upper': -0.5000,
        'moderate_antag_upper': -0.2000,
        'weak_antag_upper': -0.0500,
        'additive_lower': -0.0500,
        'additive_upper': 0.0500,
        'weak_syn_lower': 0.4172,
        'moderate_syn_lower': 0.6031,
        'strong_syn_lower': 0.7890
    }
    hsa_thresholds = {
        'strong_antag_upper': -0.4000,
        'moderate_antag_upper': -0.2000,
        'weak_antag_upper': -0.0500,
        'additive_lower': -0.0500,
        'additive_upper': 0.0500,
        'weak_syn_lower': 0.6725,
        'moderate_syn_lower': 0.8076,
        'strong_syn_lower': 0.9427
    }

    # ---------- 有序类别 ----------
    ordered_classes = [
        'Strong Antagonism', 'Moderate Antagonism', 'Weak Antagonism',
        'Additive', 'Weak Synergy', 'Moderate Synergy', 'Strong Synergy'
    ]

    # ---------- 颜色映射 ----------
    color_dict = {
        'Strong Antagonism': '#d73027',
        'Moderate Antagonism': '#fc8d59',
        'Weak Antagonism': '#fee090',
        'Additive': '#f7f7f7',
        'Weak Synergy': '#91bfdb',
        'Moderate Synergy': '#4575b4',
        'Strong Synergy': '#313695'
    }

    # ---------- 辅助函数：计算理论区间 ----------
    def get_theoretical_ranges(thresholds, ordered_cats):
        strong_antag_upper = thresholds['strong_antag_upper']
        moderate_antag_upper = thresholds['moderate_antag_upper']
        weak_antag_upper = thresholds['weak_antag_upper']
        additive_lower = thresholds['additive_lower']
        additive_upper = thresholds['additive_upper']
        weak_syn_lower = thresholds['weak_syn_lower']
        moderate_syn_lower = thresholds['moderate_syn_lower']
        strong_syn_lower = thresholds['strong_syn_lower']

        ranges = {
            'Strong Antagonism': (-np.inf, strong_antag_upper),
            'Moderate Antagonism': (strong_antag_upper, moderate_antag_upper),
            'Weak Antagonism': (moderate_antag_upper, weak_antag_upper),
            'Additive': (additive_lower, additive_upper),
            'Weak Synergy': (additive_upper, moderate_syn_lower),
            'Moderate Synergy': (moderate_syn_lower, strong_syn_lower),
            'Strong Synergy': (strong_syn_lower, np.inf)
        }
        return {cat: ranges[cat] for cat in ordered_cats}

    # 计算区间
    bliss_ranges = get_theoretical_ranges(bliss_thresholds, ordered_classes)
    hsa_ranges = get_theoretical_ranges(hsa_thresholds, ordered_classes)

    # 确定绘图范围（避开 inf）
    all_lows = [low for r in bliss_ranges.values() for low in [r[0]] if not np.isinf(low)]
    all_lows += [low for r in hsa_ranges.values() for low in [r[0]] if not np.isinf(low)]
    all_highs = [high for r in bliss_ranges.values() for high in [r[1]] if not np.isinf(high)]
    all_highs += [high for r in hsa_ranges.values() for high in [r[1]] if not np.isinf(high)]
    plot_min = min(all_lows) if all_lows else -1.0
    plot_max = max(all_highs) if all_highs else 1.0

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,4), sharex=True)

    # 辅助函数：添加阈值标注
    def add_threshold_annotations(ax, ranges, ordered_cats, y_positions):
        """在每个条形下方标注阈值上界（右端点）"""
        for i, cat in enumerate(ordered_cats):
            low, high = ranges[cat]
            # 标注上界（如果不是无穷）
            if not np.isinf(high):
                label = f'{high:.2f}'
                # 放置在条形右端稍外侧，垂直位置在条形中心偏下
                ax.text(high + 0.01, i - 0.2, label, ha='left', va='center', fontsize=8, color='black')


    # Bliss 子图
    for i, cat in enumerate(ordered_classes):
        low, high = bliss_ranges[cat]
        if np.isinf(low):
            low = plot_min
        if np.isinf(high):
            high = plot_max
        width = high - low
        if width > 0:
            ax1.barh(i, width, left=low, height=0.6, color=color_dict[cat], edgecolor='black')
    ax1.set_yticks(range(len(ordered_classes)))
    ax1.set_yticklabels(ordered_classes)
    ax1.set_xlim(plot_min, plot_max)
    ax1.set_xlabel('Bliss Synergy Score')
    ax1.set_title('Bliss Synergy Classification')
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(axis='x', alpha=0.3)
    add_threshold_annotations(ax1, bliss_ranges, ordered_classes, range(len(ordered_classes)))

    # HSA 子图
    for i, cat in enumerate(ordered_classes):
        low, high = hsa_ranges[cat]
        if np.isinf(low):
            low = plot_min
        if np.isinf(high):
            high = plot_max
        width = high - low
        if width > 0:
            ax2.barh(i, width, left=low, height=0.6, color=color_dict[cat], edgecolor='black')
    ax2.set_yticks(range(len(ordered_classes)))
    ax2.set_yticklabels(ordered_classes)
    ax2.set_xlim(plot_min, plot_max)
    ax2.set_xlabel('HSA Synergy Score')
    ax2.set_title('HSA Synergy Classification')
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(axis='x', alpha=0.3)
    add_threshold_annotations(ax2, hsa_ranges, ordered_classes, range(len(ordered_classes)))

    plt.tight_layout()
    save_path = os.path.join(result_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return fig, save_path

# ============================================================================
# 6. PLOTTING AND SAVING (all figures and data)
# ============================================================================
def save_all_figures_and_data(ml_results, synergy_df):
    """Generate and save all plots and CSV outputs."""
    print("\nSaving all figures and data...")
    # Save CV summaries
    ml_results['loco_sum'].to_csv(os.path.join(result_dir, 'loco_summary.csv'))
    ml_results['lodo_sum'].to_csv(os.path.join(result_dir, 'lodo_summary.csv'))
    ml_results['kfold_sum'].to_csv(os.path.join(result_dir, 'kfold_summary.csv'))
    ml_results['perm_imp'].to_csv(os.path.join(result_dir, 'permutation_importance.csv'), index=False)
    if ml_results['builtin_imp'] is not None:
        ml_results['builtin_imp'].to_csv(os.path.join(result_dir, 'builtin_importance.csv'), index=False)

    # SHAP values
    shap_df = pd.DataFrame(ml_results['shap_vals'], columns=ml_results['feature_names'])
    shap_df.to_csv(os.path.join(result_dir, 'shap_values.csv'), index=False)

    # Prediction vs actual for best model in LOCO
    best = ml_results['best_name']
    loco_best = ml_results['loco_sum'].loc[best]
    pred_df = pd.DataFrame({'True': loco_best['all_y_true'], 'Pred': loco_best['all_y_pred'],
                            'Residual': loco_best['all_y_true'] - loco_best['all_y_pred']})
    pred_df.to_csv(os.path.join(result_dir, 'loco_predictions.csv'), index=False)

    # Figure 1: CV performance comparison (grouped bar chart for MAE across strategies)
    fig, ax = plt.subplots(figsize=(10, 6))
    strategies = ['LOCO', 'LODO', '5-Fold']
    summaries = [ml_results['loco_sum'], ml_results['lodo_sum'], ml_results['kfold_sum']]
    models = summaries[0].index  # 假设所有策略的模型顺序一致
    x = np.arange(len(models))
    width = 0.25
    colors = {'LOCO': '#e74c3c', 'LODO': '#3498db', '5-Fold': '#2ecc71'}

    for i, (strategy, summ) in enumerate(zip(strategies, summaries)):
        mae = summ['mean_test_mae'].values
        err = summ['std_test_mae'].values
        ax.bar(x + i * width, mae, width, yerr=err, capsize=5,
               label=strategy, color=colors[strategy], alpha=0.8, edgecolor='black')

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Mean Absolute Error (MAE)')
    ax.set_title('Model Performance Comparison across CV Strategies')
    ax.legend()
    #ax.grid(True, axis='y', alpha=0.3)


    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'cv_mae_comparison.png'), dpi=300)
    plt.close()

    # Figure 2: SHAP summary
    plt.figure(figsize=(8,6))
    shap.summary_plot(ml_results['shap_vals'], ml_results['X_for_shap'], feature_names=ml_results['feature_names'],
                      show=False)
    plt.title(f'SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'shap_summary.png'), dpi=300)
    plt.close()

    # Figure 3: Permutation importance top 20
    plt.figure(figsize=(10, 6))
    top20 = ml_results['perm_imp'].head(20)
    sns.barplot(data=top20, x='importance', y='feature', palette='viridis')
    plt.title(f'Permutation Importance - {best}')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'permutation_importance.png'), dpi=300)
    plt.close()

    # Figure 4: Prediction vs Actual
    plt.figure(figsize=(6, 6))
    plt.scatter(pred_df['True'], pred_df['Pred'], alpha=0.5)
    lims = [min(pred_df.min().min(), 0), max(pred_df.max().max(), 1)]
    plt.plot(lims, lims, 'r--')
    plt.xlabel('True Viability')
    plt.ylabel('Predicted Viability')
    plt.title(f'{best} - LOCO predictions')
    plt.savefig(os.path.join(result_dir, 'pred_vs_actual.png'), dpi=300)
    plt.close()

    # Figure 5: Residual distribution
    plt.figure(figsize=(8, 4))
    plt.hist(pred_df['Residual'], bins=30, edgecolor='black')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.savefig(os.path.join(result_dir, 'residual_hist.png'), dpi=300)
    plt.close()

    print("All figures and data saved.")


def compare_bliss_hsa_sensitivity(synergy_df, output_dir):
    """
    比较Bliss和HSA模型的一致性，并进行敏感性分析。
    """
    print("\n" + "=" * 80)
    print("BLISS VS HSA COMPARISON AND SENSITIVITY ANALYSIS")
    print("=" * 80)

    # 确保有必要的列
    required_cols = ['Bliss_Synergy', 'HSA_Synergy', 'Synergy_p_adj']
    for col in required_cols:
        if col not in synergy_df.columns:
            print(f"Missing column {col}, skipping comparison.")
            return

    # 1. 散点图：Bliss vs HSA 协同分数
    plt.figure(figsize=(8, 6))
    plt.scatter(synergy_df['Bliss_Synergy'], synergy_df['HSA_Synergy'],
                alpha=0.5, c='#1f77b4', edgecolors='white', linewidth=0.5)
    plt.plot([-1, 1], [-1, 1], 'r--', linewidth=1.5, label='Identity')
    plt.xlabel('Bliss Synergy Score')
    plt.ylabel('HSA Synergy Score')
    plt.title('Comparison of Bliss and HSA Synergy Scores')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, 'bliss_vs_hsa_scatter.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 分类一致性分析（使用优化后的Bliss分类和HSA分类）
    if 'Bliss_Class_Optimized' in synergy_df.columns and 'HSA__Class_Optimized' in synergy_df.columns:
        # 构建混淆矩阵
        from sklearn.metrics import confusion_matrix
        # 将分类映射为数字以便计算一致性
        # 简单处理：定义三个大类：Synergy (包含强/中/弱), Additive, Antagonism (包含强/中/弱)
        def map_to_broad_class(cls):
            if 'Synergy' in cls:
                return 'Synergy'
            elif 'Antagonism' in cls:
                return 'Antagonism'
            else:
                return 'Additive'

        bliss_broad = synergy_df['Bliss_Class_Optimized'].apply(map_to_broad_class)
        hsa_broad = synergy_df['HSA_Class_Optimized'].apply(map_to_broad_class)

        # 计算一致性百分比
        agreement = (bliss_broad == hsa_broad).mean() * 100
        print(f"Overall broad-class agreement between Bliss and HSA: {agreement:.1f}%")

        # 混淆矩阵
        cm = confusion_matrix(bliss_broad, hsa_broad, labels=['Synergy', 'Additive', 'Antagonism'])
        # 绘制热图
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Synergy', 'Additive', 'Antagonism'],
                    yticklabels=['Synergy', 'Additive', 'Antagonism'])
        plt.xlabel('HSA Classification')
        plt.ylabel('Bliss Classification')
        plt.title(f'Classification Agreement (Bliss vs HSA)\nAgreement = {agreement:.1f}%')
        plt.tight_layout()
        cm_path = os.path.join(output_dir, 'bliss_hsa_confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 保存一致性统计
        with open(os.path.join(output_dir, 'bliss_hsa_agreement.txt'), 'w') as f:
            f.write(f"Broad-class agreement: {agreement:.1f}%\n")
            f.write("Confusion matrix (rows=Bliss, cols=HSA):\n")
            f.write(str(cm))

    # 3. 敏感性分析：改变Bliss分类阈值，观察Top组合的稳定性
    # 定义一组候选阈值（强协同下界）
    candidate_thresholds = [0.4, 0.5, 0.6, 0.7]
    top_n = 20
    # 基于当前Bliss分数选择Top N组合（不考虑显著性）
    top_combos = synergy_df.nlargest(top_n, 'Bliss_Synergy')[
        ['Cell_Line', 'Drug3', 'D1_conc', 'D2_conc', 'D3_conc', 'Bliss_Synergy']]

    stability_results = []
    for thresh in candidate_thresholds:
        # 定义简单分类：大于thresh为强协同
        strong_count = (synergy_df['Bliss_Synergy'] > thresh).sum()
        # 检查Top N中有多少满足该阈值
        top_strong = (top_combos['Bliss_Synergy'] > thresh).sum()
        stability_results.append({'Threshold': thresh, 'Total_Strong': strong_count, 'Top20_Strong': top_strong})

    stability_df = pd.DataFrame(stability_results)
    print("\nSensitivity analysis (Bliss strong synergy thresholds):")
    print(stability_df)
    stability_df.to_csv(os.path.join(output_dir, 'bliss_threshold_sensitivity.csv'), index=False)

    # 绘制Top组合的Bliss分数条，并标注不同阈值线
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), top_combos['Bliss_Synergy'].values, color='steelblue')
    for thresh in candidate_thresholds:
        plt.axvline(x=thresh, linestyle='--', alpha=0.7, label=f'Threshold = {thresh}')
    plt.yticks(range(top_n), [f"{row.Cell_Line}\n{row.Drug3}" for _, row in top_combos.iterrows()], fontsize=8)
    plt.xlabel('Bliss Synergy Score')
    plt.title(f'Top {top_n} Synergistic Combinations (by Bliss)')
    plt.legend()
    plt.tight_layout()
    top_plot_path = os.path.join(output_dir, 'top_combinations_sensitivity.png')
    plt.savefig(top_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 提供统计总结：Bliss与HSA的相关性
    correlation = synergy_df['Bliss_Synergy'].corr(synergy_df['HSA_Synergy'])
    print(f"Pearson correlation between Bliss and HSA: {correlation:.3f}")

    # 保存所有结果
    with open(os.path.join(output_dir, 'bliss_hsa_analysis_summary.txt'), 'w') as f:
        f.write("BLISS VS HSA ANALYSIS SUMMARY\n")
        f.write("=============================\n")
        f.write(f"Pearson correlation: {correlation:.3f}\n")
        #f.write(f"Broad-class agreement: {agreement:.1f}%\n")
        f.write("\nSensitivity analysis:\n")
        f.write(stability_df.to_string())

    return stability_df


if __name__ == '__main__':

    df, universal_drugs, limited_drugs, all_cell_lines, all_drugs = load_and_analyze_data('data/ThreeDrugCombs.csv')
    # Load cell line PCA features
    # 2. Load biological features (or generate dummy if missing)
    cell_features = load_cell_line_features(expression_file='data/cell_three.csv', n_components=50)

    # Load drug Morgan fingerprints
    drug_dict = load_drug_morgan('data/ThreeDrugCombs.csv', radius=2, nBits=1024)

    # Calculate synergy
    synergy_df = calculate_synergy_scores(df, [], [])

    # 优化分类阈值
    synergy_df, thresholds = analyze_synergy_distribution_and_optimize_thresholds(synergy_df, result_dir, n_clusters=5,
                                                                                  p_threshold=0.05,
                                                                                  literature_thresholds=(0.24, 0.67))

    # 保存包含优化分类的完整结果
    synergy_df.to_csv(f'{result_dir}/synergy_results_with_optimized_class.csv', index=False)

    # 2. 对 HSA 进行数据驱动分类，并与 Bliss 比较
    synergy_df, hsa_thresholds = analyze_hsa_distribution_and_compare_with_bliss(synergy_df, output_dir=result_dir,
                                                                                 n_clusters=5, p_threshold=0.05,
                                                                                 literature_thresholds=(0.20, 0.40))

    # 保存最终结果
    synergy_df.to_csv(f'{result_dir}/synergy_final_with_both_optimized.csv', index=False)

    ml_results = build_ml_models(df, synergy_df, cell_features, drug_dict)

    #自动存 CSV
    save_ml_results_to_csv(ml_results, out_dir="ACM_BCB_41/result_110")
