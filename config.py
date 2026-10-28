import json

import numpy as np

from ckpt_root import ARTIFACTS as _CKPT_ROOT, TB_ROOT as _TB_ROOT

CONFIG = {
    "seed": 0,
    "data": {
        "n_rev": 16,                   
        "n_probes": 8,                 
        "probe_layout": "real8",
        "probe_angles_deg": None,
        "f_rot_hz": 35.0,              
        "f_rot_range": [25.0, 45.0],  
        "rate_ref_hz": 35.0,           
        "regime": "fluct",             # fluct/probe(constant speed)
        "speed_fluct_mode": "rw",       #rw/iid
        "speed_fluct": 0.003,          
        "speed_fluct_sample": False,    
        "speed_fluct_range": [0.0, 0.001],
        "speed_fluct_iid": 0.001,      
        "speed_fluct_iid_sample": False,
        "speed_fluct_iid_range": [0.0, 0.001],
        "speed_ramp_mode": "rel", #rel     
        "speed_ramp": 0.0015,          
        "speed_ramp_sample": False,     
        "speed_ramp_range": [0.0, 0.0015],
        "speed_ramp_rpm": -0.035,
        "speed_ramp_rpm_sample": True,
        "speed_ramp_rpm_range": [-0.04, 0.04],
        "freq_drift": False,           
        "drift_coupling": 0.5,         
        "drift_coupling_range": [0.0, 5.0],
    },

    "signal": {
        "num_components": 6,          
        "k_range": None,            
        "min_sep_hz": 4.0,             
        "f_lo": 5.0,                   
        "f_hi": 495.0,                 
        "distance": "uniform",         
        "amplitude": "normal_floor",   
        "floor_amplitude": 0.1,
    },

    "frequency": {
        "grid_step_hz": 2.0,
        "f_min": 0.0,                
        "f_max": 500.0,
    },

    "encoder": {
        "type": "nudft_ftf_anchor",
        "rate_inject": True,           
        "film_input": "summary",       
        "ftf": {
            "d_model": 128,
            "n_heads": 8,
            "n_layers": 4,
            "ffn_dim": 256,
            "dropout": 0.0,
        },
        "anchor": {
            "obj_sigma_hz": None,      
            "obj_sigma_bins": 1.0,     
        },
    },

    "training": {
        "batch_size": 256,
        "num_workers": 12,             
        "train_snr_choices": {"min": 0.0, "max": 50.0},
        "stage1": {
            "iters": 60000,
            "lr": 3e-4,
            "anneal_start": 40000,
            "anneal_min_factor": 0.01,
        },
        "stage2": {
            "iters": 20000,
            "lr": 3e-4,
            "anneal_start": 6000,      
            "anneal_min_factor": 0.01,
        },
    },

    "eval": {
        "eval_every": 2000,
        "eval_batches": 8,             
        "monitor_snr_db": 20.0,        
        "eval_snr_db": [-5.0, 0.0, 5.0, 10.0, 20.0, 30.0],   
        "final_recall_dL": [0, 2],
        "basin_hz": 2.0,               
        "nms_distance_hz": 2.7,        
    },

    "crosswin": {
        "n_rev_seq": 104,               
        "hop_rev": 8,                 
        "time_axis": {
            "n_heads": 4,
            "ffn_dim": 256,
        },
        "train": {
            "iters": 8000,
            "batch_size": 32,
            "lr": 1e-3,
            "anneal_start": 4800,
            "anneal_min_factor": 0.01,
            "beta_gnll": 0.0,         
        },
        "eval": {
            "eval_every": 500,
            "monitor_snr_db": 20.0,    
            "eval_size": 256,          
        },
    },

    "checkpoint": {
        "dir": _CKPT_ROOT,
    },

    "tb_dir": _TB_ROOT,        
}


PROBE_LAYOUTS = {
    "real8": [0.0, 9.47, 29.00, 43.12, 70.08, 80.66, 94.06, 112.54],
}


def resolve_probe_angles(cfg):
    layout = cfg["data"]["probe_layout"]
    if layout == "custom":
        ang = cfg["data"]["probe_angles_deg"]
        if not ang:
            raise ValueError("probe_layout='custom' requires probe_angles_deg")
        return list(ang)
    if layout not in PROBE_LAYOUTS:
        raise ValueError(f"unknown probe_layout={layout!r}")
    return list(PROBE_LAYOUTS[layout])


def grid_size(cfg):
    fcfg = cfg["frequency"]
    return int(round((fcfg["f_max"] - fcfg["f_min"]) / fcfg["grid_step_hz"])) + 1


def signal_dim(cfg):
    d = cfg["data"]
    return int(d["n_probes"]) * int(d["n_rev"])


def load_ckpt_config(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
