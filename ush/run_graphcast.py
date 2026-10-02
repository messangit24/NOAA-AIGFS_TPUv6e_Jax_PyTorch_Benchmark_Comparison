#!/usr/bin/env python3
"""
ush/run_graphcast.py - NOAA AIGFS GraphCast TPU Forecast Inference Runner
Supports multi-step autoregressive forecasting on Google Cloud TPU v6e.
Uses chunked rollout to guarantee high throughput and zero OOM across 16-day forecasts.
Supports Direct GCS Cloud Streaming (gs://) with automatic timestamps.
"""
import argparse
import functools
import gc
import glob
import math
import os
import sys
import time
from datetime import datetime

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

# Import GraphCast core modules
try:
    from graphcast import autoregressive, casting, checkpoint, data_utils, graphcast, normalization, rollout
except ImportError as e:
    print(f"Error loading GraphCast modules: {e}")
    sys.exit(1)


# Global timing metrics exposed for profiling wrappers (e.g. profile_wrapper.py)
COMPILE_TIME = 0.0
INFERENCE_TIME = 0.0
TOTAL_FORECAST_TIME = 0.0
NUM_STEPS = 64
OUTPUT_FILE = ""


def parse_args():
    parser = argparse.ArgumentParser(description="AIGFS GraphCast TPU Forecast Runner")
    parser.add_argument("-n", "--name", type=str, default="aigfs", help="Run name (default: aigfs)")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input initial condition NetCDF file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory or GCS Bucket URI (gs://...)")
    parser.add_argument("-w", "--weights", type=str, required=True, help="Weights path or directory containing .npz checkpoint")
    parser.add_argument("-l", "--lead_steps", type=int, default=64, help="Number of 6-hour forecast lead steps (64 = 16 days)")
    parser.add_argument("-c", "--chunk_size", type=int, default=None, help="Rollout chunk size on TPU (default: equal to lead_steps for single on-device rollout)")
    parser.add_argument("--profile_dir", type=str, default=None, help="Directory to save XProf profile trace (optional)")
    return parser.parse_args()


def locate_checkpoint(weights_arg):
    if os.path.isfile(weights_arg) and weights_arg.endswith(".npz"):
        return weights_arg

    if os.path.isdir(weights_arg):
        npz_files = glob.glob(os.path.join(weights_arg, "*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz checkpoints found in directory: {weights_arg}")

        for f in npz_files:
            base = os.path.basename(f).lower()
            if "operational" in base and "13" in base:
                return f
        for f in npz_files:
            base = os.path.basename(f).lower()
            if "13" in base and "0.25" in base:
                return f
        return npz_files[0]

    raise FileNotFoundError(f"Weights path {weights_arg} does not exist.")


def locate_and_load_stats(weights_arg):
    search_dirs = []
    if os.path.isfile(weights_arg):
        search_dirs.append(os.path.dirname(weights_arg))
        search_dirs.append(os.path.join(os.path.dirname(weights_arg), "stats"))
    elif os.path.isdir(weights_arg):
        search_dirs.append(weights_arg)
        search_dirs.append(os.path.join(weights_arg, "stats"))

    search_dirs.extend(["/data/fix", "/data/fix/stats"])

    for d in search_dirs:
        p_diffs = os.path.join(d, "diffs_stddev_by_level.nc")
        p_mean = os.path.join(d, "mean_by_level.nc")
        p_stddev = os.path.join(d, "stddev_by_level.nc")
        if os.path.exists(p_diffs) and os.path.exists(p_mean) and os.path.exists(p_stddev):
            print(f"Loading normalization statistics from: {d}")
            diffs_stddev = xr.load_dataset(p_diffs).compute()
            mean = xr.load_dataset(p_mean).compute()
            stddev = xr.load_dataset(p_stddev).compute()
            return diffs_stddev, mean, stddev

    raise FileNotFoundError(f"Normalization statistics not found in any of: {search_dirs}")


def construct_wrapped_graphcast(model_config, task_config, diffs_stddev, mean, stddev):
    predictor = graphcast.GraphCast(model_config, task_config)
    predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev,
        mean_by_level=mean,
        stddev_by_level=stddev,
    )
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=False)
    return predictor


def main():
    global COMPILE_TIME, INFERENCE_TIME, TOTAL_FORECAST_TIME, NUM_STEPS, OUTPUT_FILE

    args = parse_args()
    NUM_STEPS = args.lead_steps
    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    print("\n========================================================")
    print("       AIGFS GRAPHCAST TPU INFERENCE RUNNER")
    print("========================================================")
    print(f"  • Model Name:        {args.name}")
    print(f"  • Input Path:        {args.input}")
    print(f"  • Output Location:   {args.output}")
    print(f"  • Run Timestamp:     {timestamp_str} UTC")
    print(f"  • Lead Steps:        {args.lead_steps} ({args.lead_steps * 6} Hours / {args.lead_steps * 6 // 24} Days)")
    print(f"  • JAX Devices:       {jax.devices()}")
    print("========================================================\n")

    if not args.output.startswith("gs://"):
        os.makedirs(args.output, exist_ok=True)

    weights_path = locate_checkpoint(args.weights)
    print(f"Loading checkpoint weights: {os.path.basename(weights_path)}...")

    with open(weights_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
        params = ckpt.params
        state = {}
        model_config = ckpt.model_config
        task_config = ckpt.task_config

    diffs_stddev, mean, stddev = locate_and_load_stats(args.weights)

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"Loading input initial condition file: {args.input}...")
    ds = xr.open_dataset(args.input)

    if "batch" in ds.dims and "batch" not in ds.coords:
        ds = ds.assign_coords(batch=np.arange(ds.sizes["batch"]))
    elif "batch" not in ds.dims:
        ds = ds.expand_dims("batch", axis=0).assign_coords(batch=[0])

    data_utils.add_derived_vars(ds)

    if "toa_incident_solar_radiation" in task_config.input_variables and "toa_incident_solar_radiation" not in ds.data_vars:
        data_utils.add_tisr_var(ds)

    t0_datetime = pd.to_datetime(ds.coords["datetime"].values[0, -1])
    time_coords = ds.coords["time"]
    ds = ds.assign_coords(time=time_coords - time_coords[-1])

    input_duration = pd.Timedelta(task_config.input_duration)
    eval_inputs = ds.sel(time=slice(-input_duration + pd.Timedelta("1ns"), pd.Timedelta(0)))
    if "datetime" in eval_inputs.coords:
        eval_inputs = eval_inputs.drop_vars("datetime")
    eval_inputs = eval_inputs[list(task_config.input_variables)]

    if args.chunk_size is not None and args.chunk_size > 0:
        chunk_size = min(args.chunk_size, args.lead_steps)
    else:
        # Default to 32 steps (or lead_steps if smaller) to maximize TPU MXU throughput
        # while fitting comfortably within TPU v6e 32GB HBM limits
        chunk_size = min(32, args.lead_steps)

    num_chunks = math.ceil(args.lead_steps / chunk_size)
    print(f"Execution Strategy: {num_chunks} chunk(s) of {chunk_size} steps each.")

    lead_times_full = [pd.Timedelta(hours=6 * (i + 1)) for i in range(args.lead_steps)]
    target_datetimes_full = [t0_datetime + lt for lt in lead_times_full]
    chunk_lead_times = [pd.Timedelta(hours=6 * (i + 1)) for i in range(chunk_size)]

    surface_vars = {
        "2m_temperature", "mean_sea_level_pressure",
        "10m_v_component_of_wind", "10m_u_component_of_wind",
        "total_precipitation_6hr"
    }

    chunk_target_vars = {}
    n_lat = len(ds.coords["lat"])
    n_lon = len(ds.coords["lon"])
    n_levels = len(task_config.pressure_levels)

    for var in task_config.target_variables:
        if var in surface_vars:
            chunk_target_vars[var] = (
                ("batch", "time", "lat", "lon"),
                np.full((1, chunk_size, n_lat, n_lon), np.nan, dtype=np.float32)
            )
        else:
            chunk_target_vars[var] = (
                ("batch", "time", "level", "lat", "lon"),
                np.full((1, chunk_size, n_levels, n_lat, n_lon), np.nan, dtype=np.float32)
            )

    chunk_targets_template = xr.Dataset(
        data_vars=chunk_target_vars,
        coords={
            "batch": [0],
            "time": chunk_lead_times,
            "lat": ds.coords["lat"],
            "lon": ds.coords["lon"],
            "level": list(task_config.pressure_levels),
        }
    )

    forcings_full = xr.Dataset(
        coords={
            "batch": [0],
            "time": lead_times_full,
            "lat": ds.coords["lat"],
            "lon": ds.coords["lon"],
            "datetime": (("batch", "time"), [target_datetimes_full]),
        }
    )
    data_utils.add_derived_vars(forcings_full)
    data_utils.add_tisr_var(forcings_full)
    forcings_full = forcings_full.drop_vars("datetime")[list(task_config.forcing_variables)]

    chunk_forcings_template = forcings_full.isel(time=slice(0, chunk_size)).assign_coords(time=chunk_lead_times)

    @hk.transform_with_state
    def run_forward(m_cfg, t_cfg, inputs_data, targets_tmpl, forcings_data):
        predictor = construct_wrapped_graphcast(m_cfg, t_cfg, diffs_stddev, mean, stddev)
        return predictor(inputs_data, targets_template=targets_tmpl, forcings=forcings_data)

    def with_configs(fn):
        return functools.partial(fn, m_cfg=model_config, t_cfg=task_config)

    jitted_apply = jax.jit(with_configs(run_forward.apply))

    print(f"\nCompiling JAX graph for {chunk_size}-step chunk on TPU v6e...")
    t_comp_start = time.perf_counter()
    lowered = jitted_apply.lower(
        params,
        state,
        jax.random.PRNGKey(0),
        inputs_data=eval_inputs,
        targets_tmpl=chunk_targets_template,
        forcings_data=chunk_forcings_template,
    )
    compiled = lowered.compile()
    compile_time = time.perf_counter() - t_comp_start
    COMPILE_TIME = compile_time
    del lowered
    gc.collect()
    print(f"JIT Compilation completed in {compile_time:.2f} seconds!\n")

    xprof_dir = args.profile_dir or os.environ.get("XPROF_LOG_DIR")
    if xprof_dir:
        import contextlib
        print(f"Attaching XProf TPU Profiler trace to rollout: {xprof_dir}...")
        profiler_ctx = jax.profiler.trace(xprof_dir, create_perfetto_link=False)
    else:
        import contextlib
        profiler_ctx = contextlib.nullcontext()

    print(f"Executing {args.lead_steps}-step rollout on TPU v6e ({num_chunks} chunk(s) of {chunk_size} steps)...")
    t_infer_start = time.perf_counter()
    current_inputs = eval_inputs
    chunks = []

    with profiler_ctx:
        for c in range(num_chunks):
            t_c_start = time.perf_counter()
            slice_start = c * chunk_size
            slice_end = min((c + 1) * chunk_size, args.lead_steps)
            actual_chunk_len = slice_end - slice_start
            slice_idx = slice(slice_start, slice_end)
            chunk_forcings = forcings_full.isel(time=slice_idx).assign_coords(time=chunk_lead_times[:actual_chunk_len])

            pred, _ = compiled(
                params,
                state,
                jax.random.PRNGKey(c),
                inputs_data=current_inputs,
                targets_tmpl=chunk_targets_template,
                forcings_data=chunk_forcings,
            )
            first_leaf = jax.tree_util.tree_leaves(pred)[0]
            first_leaf.block_until_ready()

            chunk_cpu = jax.device_get(pred)
            del pred, first_leaf
            gc.collect()

            actual_lead_times = lead_times_full[slice_start:slice_end]
            chunk_cpu = chunk_cpu.assign_coords(time=actual_lead_times)
            chunks.append(chunk_cpu)

            if c < num_chunks - 1:
                next_frame = xr.merge([
                    chunk_cpu.isel(time=slice(-2, None)),
                    chunk_forcings.isel(time=slice(-2, None))
                ])
                if "datetime" in next_frame.coords:
                    next_frame = next_frame.drop_vars("datetime")
                next_inputs = rollout._get_next_inputs(current_inputs, next_frame)
                current_inputs = next_inputs.assign_coords(time=eval_inputs.coords["time"])

            t_c_end = time.perf_counter()
            print(f"  • Chunk {c+1}/{num_chunks} completed in {t_c_end - t_c_start:.3f} s")

    inference_time = time.perf_counter() - t_infer_start
    INFERENCE_TIME = inference_time
    TOTAL_FORECAST_TIME = compile_time + inference_time

    print(
        f"\nRollout execution completed on TPU in {inference_time:.3f} seconds! "
        f"({(inference_time / args.lead_steps) * 1000.0:.2f} ms/step)\n"
    )

    print("Formatting output forecast dataset...")
    if len(chunks) == 1:
        full_pred = chunks[0]
    else:
        full_pred = xr.concat(chunks, dim="time")

    for v in full_pred.data_vars:
        dims = list(full_pred[v].dims)
        if "batch" in dims and "time" in dims and dims[0] != "batch":
            new_order = ["batch", "time"] + [d for d in dims if d not in ("batch", "time")]
            full_pred[v] = full_pred[v].transpose(*new_order)

    full_pred = full_pred.assign_coords(datetime=(("batch", "time"), [target_datetimes_full]))

    hours = args.lead_steps * 6
    days = hours // 24
    lead_str = f"{days}day" if hours % 24 == 0 else f"{hours}h"
    cycle_str = t0_datetime.strftime("t%Hz")

    # ==============================================================================
    # DIRECT GCS STREAMING (ZARR FORMAT) WITH TIMESTAMPS
    # ==============================================================================
    if args.output.startswith("gs://"):
        gcs_output_path = f"{args.output.rstrip('/')}/{timestamp_str}/{args.name}.{cycle_str}.forecast_{lead_str}_{timestamp_str}.zarr"
        OUTPUT_FILE = gcs_output_path
        print(f"Streaming forecast dataset directly from TPU memory to GCS: {gcs_output_path}...")

        # Optimize Zarr chunking for Cloud Storage (chunk by entire spatial grids to avoid thousands of tiny HTTP requests)
        zarr_encoding = {}
        time_chunk = min(8, args.lead_steps)
        for v in full_pred.data_vars:
            dims = full_pred[v].dims
            chunk_dict = {}
            if "time" in dims:
                chunk_dict["time"] = time_chunk
            if "lat" in dims:
                chunk_dict["lat"] = len(full_pred.coords["lat"])
            if "lon" in dims:
                chunk_dict["lon"] = len(full_pred.coords["lon"])
            if "level" in dims:
                chunk_dict["level"] = len(full_pred.coords["level"])
            if "batch" in dims:
                chunk_dict["batch"] = 1
            zarr_encoding[v] = {"chunks": tuple(chunk_dict[d] for d in dims)}

        full_pred.to_zarr(gcs_output_path, mode="w", encoding=zarr_encoding, consolidated=True)

        print(f"Successfully streamed directly to GCS: {gcs_output_path}!")
    else:
        output_filename = os.path.join(args.output, f"{args.name}.{cycle_str}.forecast_{lead_str}_{timestamp_str}.nc")
        OUTPUT_FILE = output_filename
        print(f"Saving output forecast predictions to local disk: {output_filename}...")
        full_pred.to_netcdf(output_filename)
        if os.path.exists(output_filename):
            file_size_mb = os.path.getsize(output_filename) / (1024 * 1024)
            print(f"Successfully saved {output_filename} ({file_size_mb:.1f} MB)!")

    print("\n========================================================")
    print("       AIGFS FORECAST EXECUTION SUMMARY")
    print("========================================================")
    print(f"  • Checkpoint:            {os.path.basename(weights_path)}")
    print(f"  • Forecast Horizon:      {args.lead_steps} steps ({lead_str})")
    print(f"  • JIT Compilation Time:  {compile_time:.2f} s")
    print(f"  • TPU Rollout Latency:   {inference_time:.3f} s")
    print(f"  • Output NetCDF Location:{OUTPUT_FILE}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
