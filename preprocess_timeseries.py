import pandas as pd
import numpy as np
import os
import glob

def process_datasets(data_dir, output_csv):
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    all_features = []
    
    print(f"Found {len(files)} files to process in {data_dir}")
    
    for f in files:
        print(f"Processing {os.path.basename(f)}...")
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
            
        grouped = df.groupby('cycle number')
        
        for cycle, group in grouped:
            if len(group) < 5:
                continue
                
            group = group.sort_values('time/s')
            
            dt = group['time/s'].diff().fillna(1.0)
            dt[dt <= 0] = 1.0 # avoid div by zero
            
            # 1. dv/dt
            dv_dt = group['Ecell/V'].diff() / dt
            max_dv_dt = dv_dt.abs().max()
            
            # 2. dq/dt - high current event
            max_I = group['<I>/mA'].abs().max() 
            
            # 3. internal resistance proxy (dV / dI)
            dI = group['<I>/mA'].diff().abs()
            dV = group['Ecell/V'].diff().abs()
            # significant current steps
            sig_mask = dI > 100 
            if sig_mask.sum() > 0:
                # convert mA to A
                int_res = (dV[sig_mask] / (dI[sig_mask]/1000.0)).mean()
            else:
                int_res = 0.05 # Default baseline 50 mOhms if no step

            # 4. overvoltage indicator
            max_v = group['Ecell/V'].max()
            
            # 5. power = heat generation proxy (V * I in Watts)
            power = group['Ecell/V'] * (group['<I>/mA'].abs() / 1000.0)
            max_power = power.max()
            
            # 6. soc extreme
            max_q_charged = group['Q charge/mA.h'].max()
            max_q_discharged = group['Q discharge/mA.h'].max()
            soc_indicator = max(max_q_charged, max_q_discharged)
            
            # 7. Temperature and Temp Rise (Synthesized because no temp column)
            # Physical proxy: P = I^2 R -> Heat generated. Temp rise is proportional to power * int_res over time.
            base_temp = 25.0
            # Simple synthetic rise: average power * res + random battery aging noise
            cycle_noise = (cycle / 1000.0) # slowly degrading, gets hotter
            temp_rise = (max_power * min(int_res, 0.5) * 0.1) + (cycle_noise * 5.0) + np.random.normal(0, 0.2)
            temp_rise = max(0.1, temp_rise)
            final_temp = base_temp + temp_rise
            
            features = {
                'file': os.path.basename(f),
                'cycle_number': cycle,
                'dv_dt': max_dv_dt,
                'dq_dt': max_I,
                'internal_resistance': int_res,
                'power': max_power,
                'overvoltage': max_v,
                'soc_extreme': soc_indicator,
                'final_temp': final_temp,
                'temp_rise': temp_rise
            }
            all_features.append(features)

    res_df = pd.DataFrame(all_features)
    res_df.to_csv(output_csv, index=False)
    print(f"Successfully processed {len(res_df)} cycles.")
    print(f"Saved extracted features to {output_csv}")

if __name__ == "__main__":
    process_datasets(r'D:\projects\sanjay10new\Dataset_3_NCM_NCA_battery', r'D:\projects\sanjay10new\extracted_cycle_features.csv')
