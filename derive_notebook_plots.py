import os, glob, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# --- CONFIGURATION (Mapped to your active D:\ drive dataset) ---
DATASET_PATH = r"D:\projects\sanjay10new\Dataset_3_NCM_NCA_battery"
OUTPUT_DIR   = r"D:\projects\sanjay10new\Notebook_Plots_Derivation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLUMN_MAP = {
    'time': 'time/s', 'voltage': 'Ecell/V', 'current': '<I>/mA',
    'q_discharge': 'Q discharge/mA.h', 'q_charge': 'Q charge/mA.h', 'cycle': 'cycle number'
}

V_MAX, V_MIN = 4.25, 2.40
RISK_MODERATE, RISK_HIGH = 0.40, 0.70
W_DV, W_TEMP, W_PWR, W_R, W_SOC, W_DQ = 0.25, 0.20, 0.20, 0.15, 0.10, 0.10

def save_fig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved Graph: {path}')

def aggregate_cycles(df_raw, source_label):
    vc, ic, qdc, qcc = COLUMN_MAP['voltage'], COLUMN_MAP['current'], COLUMN_MAP['q_discharge'], COLUMN_MAP['q_charge']
    tc, cyc = COLUMN_MAP['time'], COLUMN_MAP['cycle']
    
    grp = df_raw.groupby(cyc)
    agg = grp.agg(
        final_voltage=(vc, 'last'), mean_voltage=(vc, 'mean'), min_voltage=(vc, 'min'), max_voltage=(vc, 'max'), std_voltage=(vc, 'std'),
        mean_current=(ic, 'mean'), max_current=(ic, 'max'), min_current=(ic, 'min'), std_current=(ic, 'std'),
        Q_discharge=(qdc, 'max'), Q_charge=(qcc, 'max'),
        duration_s=(tc, lambda x: x.max() - x.min()), n_points=(tc, 'count'),
    ).reset_index().rename(columns={cyc: 'cycle_number'})
    
    agg['Q_fade'] = agg['Q_charge'].max() - agg['Q_charge']
    agg['coulombic_eff'] = (agg['Q_discharge'] / (agg['Q_charge'] + 1e-9)).clip(0, 1.05)
    
    charge_mask = df_raw[ic] > 0
    discharge_mask = df_raw[ic] < 0
    v_chg = df_raw[charge_mask].groupby(cyc)[vc].mean().rename('mean_v_charge')
    v_dch = df_raw[discharge_mask].groupby(cyc)[vc].mean().rename('mean_v_discharge')
    
    agg = agg.merge(v_chg.reset_index().rename(columns={cyc:'cycle_number'}), on='cycle_number', how='left')
    agg = agg.merge(v_dch.reset_index().rename(columns={cyc:'cycle_number'}), on='cycle_number', how='left')
    
    agg['source_file'] = source_label
    try:
        agg['c_rate'] = float(source_label.replace('CY25-05_','').split('-')[0])
    except:
        agg['c_rate'] = 1.0
    return agg

def process_and_plot():
    files = glob.glob(os.path.join(DATASET_PATH, "*.csv"))
    all_cycle_dfs = []
    
    print("Loading data for plots...")
    for f in files:
        label = os.path.basename(f).replace('.csv', '')
        raw = pd.read_csv(f, usecols=list(COLUMN_MAP.values()))
        all_cycle_dfs.append(aggregate_cycles(raw, label))
    
    df = pd.concat(all_cycle_dfs, ignore_index=True)
    df.sort_values(['source_file', 'cycle_number'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- 1. Basic EDA Feature Distributions ---
    plot_cols = ['final_voltage', 'mean_voltage', 'Q_discharge', 'Q_charge', 'Q_fade', 'coulombic_eff']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flatten(), plot_cols):
        ax.hist(df[col].dropna(), bins=50, color='steelblue', edgecolor='white', alpha=0.85)
        ax.set_title(col, fontsize=10, fontweight='bold')
    plt.suptitle('Aggregated Feature Distributions', y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('01_feature_distributions.png')

    # --- 2. Degradation by C-rate ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    cmap = plt.cm.tab10
    for idx, (src, grp) in enumerate(df.groupby('source_file')):
        col, lbl = cmap(idx % 10), src
        axes[0,0].plot(grp['cycle_number'], grp['mean_voltage'], color=col, alpha=0.75, label=lbl)
        axes[0,1].plot(grp['cycle_number'], grp['Q_discharge'], color=col, alpha=0.75, label=lbl)
        axes[1,0].plot(grp['cycle_number'], grp['Q_fade'], color=col, alpha=0.75, label=lbl)
        axes[1,1].plot(grp['cycle_number'], grp['coulombic_eff'], color=col, alpha=0.75, label=lbl)
    
    labels = ['Voltage (V)', 'Q_discharge (mAh)', 'Q_fade (mAh)', 'CE']
    for ax, ylabel in zip(axes.flatten(), labels):
        ax.set_ylabel(ylabel); ax.set_xlabel('Cycle Number'); ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.25)
    plt.suptitle('Battery Degradation By Cycle', fontsize=14, fontweight='bold')
    save_fig('02_degradation_by_crate.png')

    # --- 3A. C-rate Avg Degradation Comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for crate, grp in df.groupby('c_rate'):
        avg = grp.groupby('cycle_number')[['Q_charge','Q_fade']].mean()
        axes[0].plot(avg.index, avg['Q_charge'], label=f'{crate}C', lw=2)
        axes[1].plot(avg.index, avg['Q_fade'], label=f'{crate}C', lw=2)
    axes[0].set(title='Charge Capacity vs Cycle', xlabel='Cycle', ylabel='Q_charge (mAh)')
    axes[1].set(title='Capacity Fade vs Cycle', xlabel='Cycle', ylabel='Q_fade (mAh)')
    for ax in axes: ax.legend(title='C-rate'); ax.grid(alpha=0.25)
    plt.suptitle('Capacity Degradation by C-rate', fontweight='bold')
    plt.tight_layout()
    save_fig('03_crate_comparison.png')

    # --- 3B. Charge vs Discharge Voltage ---
    if 'mean_v_charge' in df.columns and 'mean_v_discharge' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for crate, grp in df.groupby('c_rate'):
            avg = grp.groupby('cycle_number')[['mean_v_charge','mean_v_discharge']].mean()
            axes[0].plot(avg.index, avg['mean_v_charge'], label=f'{crate}C', lw=2)
            axes[1].plot(avg.index, avg['mean_v_discharge'], label=f'{crate}C', lw=2)
        axes[0].set(title='Mean Charge Voltage', xlabel='Cycle', ylabel='V (V)')
        axes[1].set(title='Mean Discharge Voltage', xlabel='Cycle', ylabel='V (V)')
        for ax in axes: ax.legend(); ax.grid(alpha=0.25)
        plt.suptitle('Voltage Degradation', fontweight='bold')
        plt.tight_layout()
        save_fig('04_charge_discharge_voltage.png')

    # --- 5. Feature Correlation Heatmap ---
    heat_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in ['cycle_number', 'c_rate']]
    corr = df[heat_cols].corr()
    plt.figure(figsize=(14, 11))
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)), annot=True, fmt='.2f', cmap='RdYlGn', center=0)
    plt.title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
    save_fig('05_correlation_heatmap.png')

    # --- Feature Engineering for Risk Score ---
    df['dV_dt'] = df.groupby('source_file')['final_voltage'].diff().fillna(0)
    df['dQ_dt'] = df.groupby('source_file')['Q_charge'].diff().fillna(0)
    
    dI = df.groupby('source_file')['mean_current'].diff().fillna(0)
    df['internal_resistance'] = (df['dV_dt'].abs() / (dI.abs() + 1e-9)).clip(0, 100)
    
    df['joule_heat'] = (df['mean_current'].abs()**2) * df['internal_resistance'] / 1e6
    df['temp_rise'] = df.groupby('source_file')['joule_heat'].diff().fillna(0).clip(lower=0)
    df['power'] = df['mean_voltage'] * df['mean_current'].abs()
    
    q_max = df.groupby('source_file')['Q_charge'].transform('max')
    df['SOC_est'] = (df['Q_charge'] / (q_max + 1e-9)).clip(0, 1)
    df['soc_stress_score'] = np.where(df['SOC_est'] > 0.80, df['SOC_est'] - 0.80, np.where(df['SOC_est'] < 0.20, 0.20 - df['SOC_est'], 0))

    # --- 4. Engineered Features Dist ---
    eng_vars = ['dV_dt', 'dQ_dt', 'internal_resistance', 'temp_rise', 'power', 'soc_stress_score']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, feat in zip(axes.flatten(), eng_vars):
        ax.hist(df[feat].dropna(), bins=50, color='darkgreen', edgecolor='white')
        ax.set_title(feat, fontweight='bold')
    plt.suptitle('Engineered Thermal Indicators', fontsize=14, fontweight='bold')
    save_fig('06_engineered_features.png')

    # --- Risk Score ---
    def pct_norm(series):
        lo, hi = np.percentile(series, 1), np.percentile(series, 99)
        return np.clip((series - lo) / (hi - lo + 1e-9), 0, 1)

    df['n_dV'], df['n_dQ'], df['n_temp'] = pct_norm(df['dV_dt'].abs()), pct_norm(df['dQ_dt'].abs()), pct_norm(df['temp_rise'])
    df['n_pwr'], df['n_R'], df['n_soc'] = pct_norm(df['power']), pct_norm(df['internal_resistance']), pct_norm(df['soc_stress_score'])
    
    df['risk_score'] = (W_DV*df['n_dV'] + W_TEMP*df['n_temp'] + W_PWR*df['n_pwr'] + W_R*df['n_R'] + W_SOC*df['n_soc'] + W_DQ*df['n_dQ'])
    
    ALL_LABELS = {0: 'Safe (0)', 1: 'Moderate (1)', 2: 'High Risk (2)'}
    df['risk_label'] = pd.cut(df['risk_score'], bins=[-np.inf, RISK_MODERATE, RISK_HIGH, np.inf], labels=[0, 1, 2]).astype(int)

    # --- 5. Risk Graph ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df['risk_score'], bins=70, color='crimson', edgecolor='white')
    axes[0].axvline(RISK_MODERATE, color='orange', lw=2, ls='--', label=f'Mod ({RISK_MODERATE})')
    axes[0].axvline(RISK_HIGH, color='red', lw=2, ls='--', label=f'High ({RISK_HIGH})')
    axes[0].set_title('Risk Score Distribution'); axes[0].legend()
    
    lc = df['risk_label'].value_counts().sort_index()
    active_labels_list = [ALL_LABELS[i] for i in lc.index]
    active_colors_list = [{'0': '#2ecc71', '1': '#f39c12', '2': '#e74c3c'}[str(i)] for i in lc.index]
    axes[1].pie(lc, labels=active_labels_list, autopct='%1.1f%%', colors=active_colors_list, startangle=140)
    axes[1].set_title('Class Distribution')
    
    plt.suptitle('Thermal Runaway Risk Analysis', fontsize=13, fontweight='bold')
    save_fig('07_risk_score_analysis.png')
    
    # --- 8. Percentile Analysis ---
    pctls = [5, 10, 25, 50, 75, 90, 95, 99]
    vals  = np.percentile(df['risk_score'], pctls)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df['risk_score'], bins=80, density=True, color='steelblue', alpha=0.7, label='PDF')
    for p, v in zip(pctls, vals): axes[0].axvline(v, linestyle=':', alpha=0.7, label=f'P{p}={v:.2f}')
    axes[0].axvline(RISK_MODERATE, color='orange', lw=2, label='Moderate')
    axes[0].axvline(RISK_HIGH, color='red', lw=2, label='High')
    axes[0].set_title('Risk Score PDF'); axes[0].legend(fontsize=6); axes[0].grid(alpha=0.25)
    
    for lid, col in zip([0,1,2], ['#2ecc71','#f39c12','#e74c3c']):
        sub = df[df['risk_label']==lid]['risk_score']
        if len(sub) > 0: axes[1].boxplot(sub, positions=[lid], patch_artist=True, boxprops=dict(facecolor=col))
    axes[1].set_xticks(list(ALL_LABELS.keys())); axes[1].set_xticklabels(list(ALL_LABELS.values()))
    axes[1].set_title('Risk Distribution per Class'); axes[1].grid(alpha=0.25)
    plt.suptitle('Percentile Analysis', fontweight='bold')
    plt.tight_layout()
    save_fig('08_percentile_analysis.png')
    
    print("\nAll EDA graphical derivations successfully completed!")

if __name__ == "__main__":
    process_and_plot()
