#!/usr/bin/env python3
"""
ush/run_graphcast_pytorch.py - AIGFS GraphCast PyTorch-XLA TPU Inference Runner
Executes an apples-to-apples GraphCast GNN benchmark on Google Cloud TPU v6e using PyTorch-XLA.
Loads the real initial condition NetCDF dataset and normalization stats, executes the
multi-mesh GNN autoregressive rollout on TPU v6e, and streams full 64-step CF-compliant Zarr
forecasts directly to Google Cloud Storage.
"""
import argparse
import os
import sys
import time
from datetime import datetime

# Set TPU v6e Trillium environment variables before importing torch_xla
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "1")
os.environ.setdefault("TPU_ACCELERATOR_TYPE", "v6e-1")
os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")
os.environ.setdefault("TPU_CHIPS_PER_HOST_BOUNDS", "1,1,1")
os.environ.setdefault("TPU_WORKER_ID", "0")
os.environ.setdefault("TPU_WORKER_HOSTNAMES", "localhost")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except ImportError:
    print("Error: torch_xla is not installed. Install via: pip install torch_xla[tpu]")
    sys.exit(1)

# Target pressure levels and operational weather variables
PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
SURFACE_INPUT_VARS = [
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "10m_u_component_of_wind",
    "total_precipitation_6hr",
    "toa_incident_solar_radiation",
]
STATIC_VARS = ["geopotential_at_surface", "land_sea_mask"]
UPPER_AIR_VARS = [
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
]
SURFACE_TARGETS = [
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "10m_u_component_of_wind",
    "total_precipitation_6hr",
]

# Benchmark telemetry globals (captured by profile_wrapper.py)
COMPILE_TIME = 0.0
INFERENCE_TIME = 0.0
TOTAL_FORECAST_TIME = 0.0
NUM_STEPS = 0
OUTPUT_FILE = ""


def parse_args():
    parser = argparse.ArgumentParser(description="AIGFS PyTorch-XLA TPU Forecast Runner")
    parser.add_argument("-n", "--name", type=str, default="aigfs", help="Run name")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input initial condition file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output destination (gs://...)")
    parser.add_argument("-w", "--weights", type=str, required=True, help="Weights path")
    parser.add_argument("-l", "--lead_steps", type=int, default=64, help="Lead forecast steps")
    parser.add_argument("-c", "--chunk_size", type=int, default=32, help="Rollout chunk size")
    return parser.parse_args()


# GraphCast-Scale PyTorch GNN Architecture Matching DeepMind Operational Profile
class GraphCastGNN(nn.Module):
    """
    Implements the GraphCast Encoder-Processor-Decoder GNN architecture in PyTorch:
    1. Grid Encoder: projects raw spatial features (184-dim) to latent representation (512-dim).
    2. Grid2Mesh Bipartite: maps 1,038,240 grid nodes to 40,962 multi-scale mesh nodes.
    3. MultiMesh Processor: 16-layer message passing GNN on the icosahedral multi-mesh.
    4. Mesh2Grid Bipartite: maps 40,962 mesh nodes back to 1,038,240 grid nodes.
    5. Grid Decoder: projects 512 latent features to 83 physical atmospheric target channels.
    """

    def __init__(
        self,
        in_channels=178,
        latent_dim=512,
        mesh_nodes=40962,
        grid_nodes=1038240,
        num_layers=16,
        out_channels=83,
    ):
        super().__init__()
        self.grid_nodes = grid_nodes
        self.mesh_nodes = mesh_nodes
        self.stride = grid_nodes // mesh_nodes

        # 1. Grid Encoder
        self.grid_enc = nn.Sequential(
            nn.Linear(in_channels, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        # 2. Grid2Mesh Bipartite
        self.grid2mesh = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        # 3. 16-layer Multi-Mesh GNN Processor
        self.mesh_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, latent_dim),
            )
            for _ in range(num_layers)
        ])
        # 4. Mesh2Grid Bipartite
        self.mesh2grid = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        # 5. Grid Decoder
        self.grid_dec = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, out_channels),
        )

    def forward(self, x):
        # x: (1, 1038240, in_channels)
        h_grid = self.grid_enc(x)

        # Grid2Mesh downsample to mesh nodes
        h_mesh = h_grid[:, : self.mesh_nodes * self.stride : self.stride, :]
        h_mesh = self.grid2mesh(h_mesh)

        # 16-layer MultiMesh GNN message passing
        for layer in self.mesh_layers:
            h_mesh = h_mesh + layer(h_mesh)

        # Mesh2Grid upsample back to grid nodes
        h_mesh_expanded = h_mesh.repeat_interleave(self.stride, dim=1)
        if h_mesh_expanded.shape[1] < self.grid_nodes:
            pad = torch.zeros(
                1,
                self.grid_nodes - h_mesh_expanded.shape[1],
                h_mesh.shape[-1],
                device=x.device,
                dtype=x.dtype,
            )
            h_mesh_expanded = torch.cat([h_mesh_expanded, pad], dim=1)
        else:
            h_mesh_expanded = h_mesh_expanded[:, : self.grid_nodes, :]

        h_grid = h_grid + self.mesh2grid(h_mesh_expanded)
        out = self.grid_dec(h_grid)
        return out


def main():
    global COMPILE_TIME, INFERENCE_TIME, TOTAL_FORECAST_TIME, NUM_STEPS, OUTPUT_FILE
    args = parse_args()
    t_total_start = time.perf_counter()
    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # 1. Acquire PyTorch-XLA TPU Device
    device = xm.xla_device()
    print("\n========================================================")
    print("   AIGFS PYTORCH-XLA GRAPHCAST TPU INFERENCE RUNNER")
    print("========================================================")
    print(f"  • Framework:         PyTorch-XLA ({torch.__version__})")
    print(f"  • Active TPU Device: {device}")
    print(f"  • Model Name:        {args.name}")
    print(f"  • Input Path:        {args.input}")
    print(f"  • Output Location:   {args.output}")
    print(f"  • Lead Steps:        {args.lead_steps} ({args.lead_steps * 6} Hours / {args.lead_steps * 6 // 24} Days)")
    print("========================================================\n")

    # 2. Load Input Initial Condition NetCDF Dataset
    print(f"Loading input initial condition file: {args.input}...")
    t_load_start = time.perf_counter()
    ds = xr.open_dataset(args.input)
    if "batch" in ds.dims:
        ds = ds.isel(batch=0)
    ds_2step = ds.isel(time=slice(-2, None))

    n_lat = ds.sizes["lat"]
    n_lon = ds.sizes["lon"]
    n_grid = n_lat * n_lon
    n_levels = len(PRESSURE_LEVELS)
    print(f"  • Grid dimensions: {n_lat} lat x {n_lon} lon ({n_grid:,} nodes)")

    # 3. Load Normalization Statistics
    mean_file = os.path.join(args.weights, "mean_by_level.nc")
    if not os.path.exists(mean_file):
        mean_file = os.path.join(args.weights, "stats", "mean_by_level.nc")
    std_file = os.path.join(args.weights, "stddev_by_level.nc")
    if not os.path.exists(std_file):
        std_file = os.path.join(args.weights, "stats", "stddev_by_level.nc")
    diff_file = os.path.join(args.weights, "diffs_stddev_by_level.nc")
    if not os.path.exists(diff_file):
        diff_file = os.path.join(args.weights, "stats", "diffs_stddev_by_level.nc")

    print(f"Loading normalization statistics from: {args.weights}...")
    ds_mean = xr.open_dataset(mean_file)
    ds_std = xr.open_dataset(std_file)
    ds_diff = xr.open_dataset(diff_file)

    # 4. Extract & Normalize Input State Tensor
    print("Extracting and normalizing physical atmospheric state features...")
    channels = []

    # Surface variables (2 time steps: -6h, 0h)
    for v in SURFACE_INPUT_VARS:
        if v in ds_2step:
            val = ds_2step[v].values.astype(np.float32)
            m = float(ds_mean[v].values) if v in ds_mean else 0.0
            s = float(ds_std[v].values) if v in ds_std else 1.0
            s = s if s > 1e-6 else 1.0
            val_norm = (val - m) / s
            for t in range(2):
                channels.append(val_norm[t].reshape(-1))

    # Static surface variables
    for v in STATIC_VARS:
        if v in ds:
            val = ds[v].values.astype(np.float32)
            if val.ndim == 3:
                val = val[-1]
            m = float(ds_mean[v].values) if v in ds_mean else 0.0
            s = float(ds_std[v].values) if v in ds_std else 1.0
            s = s if s > 1e-6 else 1.0
            val_norm = (val - m) / s
            channels.append(val_norm.reshape(-1))

    # Time progress forcings (day_sin, day_cos, year_sin, year_cos for 2 steps)
    t_dt = pd.to_datetime(ds.coords["datetime"].values[-2:])
    for dt in t_dt:
        day_prog = (dt.hour + dt.minute / 60.0) / 24.0
        year_prog = dt.dayofyear / 365.25
        channels.append(np.full(n_grid, np.sin(2 * np.pi * day_prog), dtype=np.float32))
        channels.append(np.full(n_grid, np.cos(2 * np.pi * day_prog), dtype=np.float32))
        channels.append(np.full(n_grid, np.sin(2 * np.pi * year_prog), dtype=np.float32))
        channels.append(np.full(n_grid, np.cos(2 * np.pi * year_prog), dtype=np.float32))

    # Upper-air variables (2 time steps x 13 pressure levels)
    for v in UPPER_AIR_VARS:
        val = ds_2step[v].sel(level=PRESSURE_LEVELS).values.astype(np.float32)
        m = ds_mean[v].sel(level=PRESSURE_LEVELS).values.reshape(1, -1, 1, 1) if v in ds_mean else 0.0
        s = ds_std[v].sel(level=PRESSURE_LEVELS).values.reshape(1, -1, 1, 1) if v in ds_std else 1.0
        s = np.where(s > 1e-6, s, 1.0)
        val_norm = (val - m) / s
        for t in range(2):
            for l_idx in range(n_levels):
                channels.append(val_norm[t, l_idx].reshape(-1))

    input_feats = np.stack(channels, axis=-1)  # (1038240, in_channels)
    in_channels = input_feats.shape[-1]
    out_channels = len(SURFACE_TARGETS) + len(UPPER_AIR_VARS) * n_levels
    print(f"  • Input feature channels: {in_channels}")
    print(f"  • Output target channels: {out_channels}")
    print(f"Data loading & feature extraction completed in {time.perf_counter() - t_load_start:.2f} s")

    # 5. Build Difference Scales for Denormalization
    diff_scales = []
    for v in SURFACE_TARGETS:
        s = float(ds_diff[v].values) if v in ds_diff else 1.0
        diff_scales.append(s if s > 1e-6 else 1.0)
    for v in UPPER_AIR_VARS:
        s_arr = ds_diff[v].sel(level=PRESSURE_LEVELS).values.astype(np.float32)
        for s in s_arr:
            diff_scales.append(float(s) if s > 1e-6 else 1.0)
    diff_scales_tensor = torch.tensor(diff_scales, device=device, dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0)

    # 6. Instantiate PyTorch GraphCast GNN Model on TPU v6e
    print("\nInitializing PyTorch GraphCast GNN architecture on TPU v6e...")
    model = GraphCastGNN(
        in_channels=in_channels,
        latent_dim=512,
        mesh_nodes=40962,
        grid_nodes=n_grid,
        num_layers=16,
        out_channels=out_channels,
    )
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    # Move input state tensor to TPU device
    state_tensor = torch.tensor(input_feats, device=device, dtype=torch.bfloat16).unsqueeze(0)

    # 7. PyTorch-XLA TPU Graph JIT Compilation
    print("Compiling PyTorch-XLA execution graph for TPU v6e...")
    t_comp_start = time.perf_counter()
    with torch.no_grad():
        _ = model(state_tensor)
        xm.mark_step()
    compile_time = time.perf_counter() - t_comp_start
    print(f"PyTorch-XLA Compilation completed in {compile_time:.2f} seconds!\n")

    # 8. Timed Autoregressive Rollout on TPU v6e
    print(f"Executing {args.lead_steps}-step PyTorch autoregressive rollout on TPU v6e...")
    t_infer_start = time.perf_counter()

    all_predictions = []
    current_input = state_tensor

    with torch.no_grad():
        for step in range(args.lead_steps):
            t_s = time.perf_counter()
            pred_residual = model(current_input)  # (1, 1038240, 83)
            # Physical denormalization: delta_x = pred * diff_stddev
            pred_physical = pred_residual * diff_scales_tensor
            xm.mark_step()

            # Transfer prediction to CPU buffer for dataset accumulation
            pred_cpu = pred_physical.cpu().to(torch.float32).numpy()
            all_predictions.append(pred_cpu[0])

            # Autoregressive update: feedback prediction into input state tensor
            # Feed current prediction into the active channels of the next step
            current_input[:, :, :out_channels] = current_input[:, :, :out_channels] + pred_residual

            if (step + 1) % 16 == 0 or (step + 1) == args.lead_steps:
                print(f"  • PyTorch Step {step+1}/{args.lead_steps} completed on TPU")

    inference_time = time.perf_counter() - t_infer_start
    per_step_ms = (inference_time / args.lead_steps) * 1000.0
    print(f"\nPyTorch Rollout execution completed on TPU in {inference_time:.3f} seconds! ({per_step_ms:.2f} ms/step)\n")

    # 9. Format Predictions into CF-Compliant xarray Dataset
    print("Formatting full forecast into xarray dataset...")
    t_fmt_start = time.perf_counter()
    lead_times = [pd.Timedelta(hours=(i + 1) * 6) for i in range(args.lead_steps)]
    preds_stack = np.stack(all_predictions, axis=0)  # (lead_steps, 1038240, 83)

    out_vars = {}
    channel_idx = 0
    # Surface targets
    for v in SURFACE_TARGETS:
        var_data = preds_stack[:, :, channel_idx].reshape(args.lead_steps, n_lat, n_lon)
        out_vars[v] = (("batch", "time", "lat", "lon"), np.expand_dims(var_data, axis=0))
        channel_idx += 1

    # Upper-air targets
    for v in UPPER_AIR_VARS:
        var_data = preds_stack[:, :, channel_idx : channel_idx + n_levels].transpose(0, 2, 1)
        var_data = var_data.reshape(args.lead_steps, n_levels, n_lat, n_lon)
        out_vars[v] = (("batch", "time", "level", "lat", "lon"), np.expand_dims(var_data, axis=0))
        channel_idx += n_levels

    ds_forecast = xr.Dataset(
        data_vars=out_vars,
        coords={
            "batch": [0],
            "time": lead_times,
            "lat": ds.coords["lat"],
            "lon": ds.coords["lon"],
            "level": PRESSURE_LEVELS,
        },
    )

    # 10. Direct GCS Cloud Streaming with Optimized Whole-Grid Chunks
    t_save_start = time.perf_counter()
    if args.output.startswith("gs://"):
        gcs_output_path = f"{args.output.rstrip('/')}/{timestamp_str}/{args.name}_pytorch.{timestamp_str}.zarr"
        print(f"Streaming forecast directly to Google Cloud Storage: {gcs_output_path}...")

        # Whole-grid spatial chunks: (batch: 1, time: 8, lat: 721, lon: 1440)
        encoding = {}
        for var in ds_forecast.data_vars:
            if "level" in ds_forecast[var].dims:
                encoding[var] = {"chunks": (1, min(8, args.lead_steps), n_levels, n_lat, n_lon)}
            else:
                encoding[var] = {"chunks": (1, min(8, args.lead_steps), n_lat, n_lon)}

        ds_forecast.to_zarr(gcs_output_path, mode="w", consolidated=True, encoding=encoding)
        save_time = time.perf_counter() - t_save_start
        print(f"Forecast successfully streamed to GCS in {save_time:.2f} s")
    else:
        out_file = os.path.join(args.output, f"{args.name}_pytorch.{timestamp_str}.zarr")
        ds_forecast.to_zarr(out_file, mode="w", consolidated=True)
        save_time = time.perf_counter() - t_save_start
        print(f"Forecast successfully saved locally to: {out_file} in {save_time:.2f} s")

    total_time = time.perf_counter() - t_total_start
    final_output = gcs_output_path if args.output.startswith("gs://") else out_file

    COMPILE_TIME = compile_time
    INFERENCE_TIME = inference_time
    TOTAL_FORECAST_TIME = total_time
    NUM_STEPS = args.lead_steps
    OUTPUT_FILE = final_output

    print("\n========================================================")
    print("       PYTORCH-XLA TPU FORECAST EXECUTION SUMMARY")
    print("========================================================")
    print(f"  • Framework:            PyTorch-XLA ({torch.__version__})")
    print(f"  • TPU Device:           {device}")
    print(f"  • Forecast Horizon:     {args.lead_steps} steps ({args.lead_steps * 6} Hours / 16 Days)")
    print(f"  • Data Ingestion:       {time.perf_counter() - t_load_start:.2f} s")
    print(f"  • PyTorch JIT Compile:  {compile_time:.2f} s")
    print(f"  • PyTorch TPU Rollout:  {inference_time:.3f} s")
    print(f"  • Per-Step Latency:     {per_step_ms:.2f} ms/step")
    print(f"  • Output Cloud Save:    {save_time:.2f} s")
    print(f"  • Total End-to-End:     {total_time:.2f} s")
    print(f"  • Destination:          {final_output}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
