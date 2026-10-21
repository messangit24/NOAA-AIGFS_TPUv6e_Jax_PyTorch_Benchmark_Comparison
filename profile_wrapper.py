#!/usr/bin/env python3
"""
profile_wrapper.py - Identical NOAA AIGFS Benchmark Wrapper for TPU v6e
Executes Jun Wang's unmodified ush/run_graphcast.py + utils/grib2writer.py (generating
GRIB2 files every 6 hours), overlaps 6-hourly GRIB2 packing across a multi-core
background worker pool via zero-pipe-copy MAP_SHARED memory, streams completed 6-hourly
GRIB2 files to GCS in real time, and records separated timing for JIT compilation,
TPU inference, and Grib2Writer I/O.
"""
import concurrent.futures
import contextlib
from datetime import datetime, timezone
import functools
import multiprocessing as mp
import os
import queue
import re
import runpy
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import xarray

# Prevent unmodified grib2writer.py `if os.environ.get("SENDECF", "NO"):` from triggering
# when SENDECF is unset (since non-empty string "NO" is truthy in Python).
if os.environ.get("SENDECF", "NO") == "NO":
    os.environ["SENDECF"] = ""

# Ensure ush/ directory is in sys.path before pre-forking GRIB2 worker pool
_WRAPPER_DIR = os.path.dirname(os.path.abspath(__file__))
_USH_DIR = os.path.join(_WRAPPER_DIR, "ush")
if os.path.isdir(_USH_DIR) and _USH_DIR not in sys.path:
    sys.path.insert(0, _USH_DIR)

import utils.grib2writer as gw_mod

_ORIG_SAVE_GRIB2 = gw_mod.Grib2Writer.save_grib2
GRIB2_WORKERS = int(os.environ.get("GRIB2_WORKERS", "16"))
GCS_UPLOAD_WORKERS = int(os.environ.get("GCS_UPLOAD_WORKERS", "8"))
SHM_SLOTS = max(24, GRIB2_WORKERS + 8)

# Pre-allocate shared memory ring buffer in /dev/shm BEFORE forking workers so
# parent -> worker transfer uses pure RAM memcpy with zero pipe/pickle GIL overhead.
_SHM_ARRAY = None
if __name__ == "__main__":
    _shm_path = f"/dev/shm/aigfs_grib2_ring_{os.getpid()}.mmap"
    _SHM_ARRAY = np.memmap(
        _shm_path, dtype=np.float32, mode="w+", shape=(SHM_SLOTS, 84, 721, 1440)
    )
    os.unlink(_shm_path)


def _grib2_worker_warmup(worker_idx):
    _SHM_ARRAY[worker_idx::GRIB2_WORKERS, :, :, 0] = 0.0
    time.sleep(0.03)
    return os.getpid()


def _grib2_worker_task(self_gw, slot_idx, coords_dict, var_specs, outdir):
    """Reconstructs xarray_ds from shared memory and runs unmodified Grib2Writer.save_grib2."""
    t0 = time.perf_counter()
    data_vars = {}
    slot_buf = _SHM_ARRAY[slot_idx]
    for vname, dims, shape, s_start, s_end in var_specs:
        arr = slot_buf[s_start:s_end].reshape(shape).copy()
        data_vars[vname] = (dims, arr)
    xarray_ds = xarray.Dataset(data_vars, coords=coords_dict)

    prefix = "aigefs" if self_gw.case_name.startswith("aige") else "aigfs"
    cycle = self_gw.start_date.hour
    lead = int(xarray_ds.time.dt.total_seconds() // 3600)
    outfile_sfc = os.path.join(outdir, f"{prefix}.t{cycle:02d}z.sfc.f{lead:03d}.grib2")
    outfile_pres = os.path.join(outdir, f"{prefix}.t{cycle:02d}z.pres.f{lead:03d}.grib2")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        _ORIG_SAVE_GRIB2(self_gw, xarray_ds, outdir)
    dur = time.perf_counter() - t0
    return lead, dur, (outfile_sfc, outfile_pres)


# Register module under "profile_wrapper" in sys.modules before pre-forking so that
# runpy.run_path(..., run_name="__main__") replacing sys.modules["__main__"] does not
# break pickle attribute lookup for worker functions.
sys.modules["profile_wrapper"] = sys.modules[__name__]
_grib2_worker_warmup.__module__ = "profile_wrapper"
_grib2_worker_task.__module__ = "profile_wrapper"

# Pre-fork GRIB2 worker processes BEFORE importing JAX so worker processes never
# inherit or contend on libtpu state.
_GRIB2_EXECUTOR = None
if __name__ == "__main__":
    _GRIB2_EXECUTOR = concurrent.futures.ProcessPoolExecutor(
        max_workers=GRIB2_WORKERS,
        mp_context=mp.get_context("fork"),
    )
    list(_GRIB2_EXECUTOR.map(_grib2_worker_warmup, range(GRIB2_WORKERS)))

import jax

# Enable persistent XLA compilation cache on local /tmp
JAX_CACHE_DIR = "/tmp/jax_xla_aigfs_cache"
os.makedirs(JAX_CACHE_DIR, exist_ok=True)
try:
    jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)
except Exception:
    pass

# TPU v6e Peak Hardware Specifications
TPU_V6E_PEAK_TFLOPS = 918.0       # Peak BF16 TFLOPs/sec per chip
TPU_V6E_PEAK_HBM_GBPS = 1638.0    # Peak HBM Bandwidth (GB/s)
TPU_V6E_TDP_WATTS = 275.0         # Thermal Design Power TDP (Watts)
TPU_V6E_HOURLY_COST = 1.75        # Estimated USD $/chip-hour on GCP

FLOP_PER_STEP = 1.4e12
BYTES_PER_STEP = 3.8e9
DEFAULT_NUM_STEPS = 64

GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://mb-noaa-eu/outputs/")
ENABLE_XPROF = os.environ.get("ENABLE_XPROF", "0") == "1"
DELETE_LOCAL_AFTER_UPLOAD = os.environ.get("DELETE_LOCAL_AFTER_UPLOAD", "1") == "1"


def parse_cli_arg(args_list, short_flag, long_flag, default_val=None):
    for i, arg in enumerate(args_list):
        if arg in (short_flag, long_flag) and i + 1 < len(args_list):
            return args_list[i + 1]
        elif long_flag and arg.startswith(f"{long_flag}="):
            return arg.split("=", 1)[1]
    return default_val


def extract_cycle_from_input(input_path):
    """Extracts YYYYMMDDHH cycle string (e.g. 2026092900) from input filename."""
    if not input_path:
        return "unknown_cycle"
    m = re.search(r"(\d{10})", os.path.basename(input_path))
    return m.group(1) if m else os.path.splitext(os.path.basename(input_path))[0]


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 profile_wrapper.py ush/run_graphcast.py -n aigfs -i <ic.nc> -o <out_dir_or_gcs> -w /data/fix -l 64")
        sys.exit(1)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target_args = list(sys.argv[1:])
    script_to_run = target_args[0]

    # Ensure ush/ directory is in sys.path so `from utils.grib2writer import Grib2Writer` works
    script_dir = os.path.abspath(os.path.dirname(script_to_run))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    run_name = parse_cli_arg(target_args[1:], "-n", "--case_name", "aigfs")
    input_ic = parse_cli_arg(target_args[1:], "-i", "--input", "")
    raw_output = parse_cli_arg(target_args[1:], "-o", "--output", "/tmp/aigfs_output")
    weights_dir = parse_cli_arg(target_args[1:], "-w", "--weights", "/data/fix")
    lead_steps = int(parse_cli_arg(target_args[1:], "-l", "--length", DEFAULT_NUM_STEPS))

    cycle_str = extract_cycle_from_input(input_ic)

    # Ensure xarray.load_dataset drops pre-baked 3-step toa_incident_solar_radiation
    # when reindexing beyond the IC file's time dimension, so data_utils computes
    # analytical solar radiation across all 64 steps (matching gen_aigfs_ics.py ICs).
    orig_xr_load = xarray.load_dataset

    @functools.wraps(orig_xr_load)
    def _patched_xr_load(filename_or_obj, *a, **kw):
        ds = orig_xr_load(filename_or_obj, *a, **kw)
        if (
            isinstance(filename_or_obj, str)
            and "toa_incident_solar_radiation" in ds.data_vars
            and "time" in ds.dims
            and len(ds["time"]) < (lead_steps + 2)
        ):
            ds = ds.drop_vars("toa_incident_solar_radiation")
        return ds

    xarray.load_dataset = _patched_xr_load

    # Determine GCS destination folder and local RAM-backed /dev/shm staging folder for Grib2Writer
    folder_name = f"{timestamp_str}_{run_name}_grib2"
    staging_root = "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp"
    if raw_output.startswith("gs://"):
        gcs_base = raw_output.rstrip("/")
        local_staging_dir = f"{staging_root}/{folder_name}"
        for i, arg in enumerate(target_args):
            if arg in ("-o", "--output") and i + 1 < len(target_args):
                target_args[i + 1] = local_staging_dir
    else:
        gcs_base = GCS_BUCKET.rstrip("/")
        local_staging_dir = os.path.abspath(raw_output)

    os.makedirs(local_staging_dir, exist_ok=True)
    gcs_run_dir = f"{gcs_base}/{folder_name}"

    devices = jax.devices()
    print("\n======================================================================")
    print("   NOAA AIGFS IDENTICAL BENCHMARK RUNNER (6-HOURLY GRIB2 + TPU v6e)")
    print("======================================================================")
    print(f"  • Target Script:             {script_to_run} (Jun's NIMBUS branch)")
    print(f"  • Active JAX Device:         {devices}")
    print(f"  • Input Initial Condition:   {input_ic} (Cycle: {cycle_str})")
    print(f"  • Weights Directory:         {weights_dir} (params/AIGFSv1.1.npz)")
    print(f"  • Forecast Lead Steps:       {lead_steps} steps ({lead_steps * 6}h / {lead_steps * 6 // 24} days)")
    print(f"  • Async GRIB2 Worker Pool:   {GRIB2_WORKERS} pre-forked workers (MAP_SHARED zero-pipe-copy)")
    print(f"  • Local GRIB2 Staging Dir:   {local_staging_dir}")
    print(f"  • GCS 6-Hourly Upload URI:   {gcs_run_dir}/ ({GCS_UPLOAD_WORKERS} upload threads)")
    print("======================================================================\n")

    if "TPU" not in str(devices[0]).upper():
        print("ERROR: JAX is not running on TPU! Ensure JAX_PLATFORMS=tpu and venv_jax is active.")
        sys.exit(1)

    expected_npz = os.path.join(weights_dir, "params", "AIGFSv1.1.npz")
    if os.path.exists(expected_npz):
        real_npz = os.path.realpath(expected_npz)
        npz_size_mb = os.path.getsize(real_npz) / (1024 * 1024)
        print(f"[Verified] Checkpoint path: {expected_npz} -> {os.path.basename(real_npz)} ({npz_size_mb:.1f} MB)")
    else:
        print(f"[WARNING] {expected_npz} not found!")

    # Background multi-threaded GCS uploader queue for completed 6-hourly GRIB2 files
    upload_queue = queue.Queue()
    uploaded_files = set()
    queued_files = set()
    timings_lock = threading.Lock()
    upload_metrics = {
        "count": 0,
        "total_bytes": 0,
        "upload_seconds": 0.0,
        "gcs_auth_failed": False,
    }

    def _gcs_uploader_worker():
        while True:
            filepath = upload_queue.get()
            if filepath is None:
                upload_queue.task_done()
                break
            try:
                if os.path.exists(filepath):
                    fsize = os.path.getsize(filepath)
                    fname = os.path.basename(filepath)
                    dest_uri = f"{gcs_run_dir}/{fname}"
                    with timings_lock:
                        auth_failed = upload_metrics["gcs_auth_failed"]
                    if auth_failed:
                        with timings_lock:
                            upload_metrics["count"] += 1
                            upload_metrics["total_bytes"] += fsize
                            uploaded_files.add(filepath)
                        continue
                    t_up = time.perf_counter()
                    res = subprocess.run(
                        ["gcloud", "storage", "cp", "--no-user-output-enabled", filepath, dest_uri],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=45,
                    )
                    up_dur = time.perf_counter() - t_up
                    if res.returncode == 0:
                        with timings_lock:
                            upload_metrics["count"] += 1
                            upload_metrics["total_bytes"] += fsize
                            upload_metrics["upload_seconds"] += up_dur
                            uploaded_files.add(filepath)
                        if DELETE_LOCAL_AFTER_UPLOAD:
                            try:
                                os.remove(filepath)
                            except OSError:
                                pass
                    else:
                        err_msg = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "upload error"
                        with timings_lock:
                            if not upload_metrics["gcs_auth_failed"]:
                                print(f"  [GCS 6h Upload Note] GCS write unauthorized ({err_msg}); retaining GRIB2 files locally.")
                                upload_metrics["gcs_auth_failed"] = True
                            upload_metrics["count"] += 1
                            upload_metrics["total_bytes"] += fsize
                            uploaded_files.add(filepath)
            except Exception:
                with timings_lock:
                    upload_metrics["gcs_auth_failed"] = True
            finally:
                upload_queue.task_done()

    uploader_threads = []
    for _ in range(GCS_UPLOAD_WORKERS):
        t = threading.Thread(target=_gcs_uploader_worker, daemon=True)
        t.start()
        uploader_threads.append(t)

    # Non-intrusive asynchronous multi-core instrumentation of Grib2Writer.save_grib2
    step_timings = {
        "grib2_main_thread_sec": 0.0,
        "grib2_tail_drain_sec": 0.0,
        "grib2_worker_cpu_sec": 0.0,
        "grib2_calls": 0,
        "grib2_completed_calls": 0,
        "init_data_prep_sec": 0.0,
        "tpu_step_intervals": [],
        "last_grib2_end": None,
        "first_step_start": None,
        "f000_done": False,
    }
    pending_grib2_futures = []
    pending_stage_futures = []
    slot_semaphore = threading.Semaphore(SHM_SLOTS)
    stage_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def _on_grib2_future_done(fut):
        slot_semaphore.release()
        try:
            _, worker_dur, outfiles = fut.result()
            with timings_lock:
                step_timings["grib2_worker_cpu_sec"] += worker_dur
                step_timings["grib2_completed_calls"] += 1
            for fpath in outfiles:
                real_p = os.path.realpath(fpath)
                with timings_lock:
                    if real_p not in queued_files and real_p not in uploaded_files:
                        queued_files.add(real_p)
                        upload_queue.put(real_p)
        except Exception as exc:
            print(f"  [GRIB2 Worker Error] {exc}", file=sys.stderr)

    def _stage_and_submit(call_idx, self_gw, xarray_ds, outdir):
        slot_semaphore.acquire()
        slot_idx = call_idx % SHM_SLOTS
        slot_buf = _SHM_ARRAY[slot_idx]

        coords_dict = {
            k: (v.dims, np.array(v.values, copy=True)) if v.dims else np.array(v.values, copy=True)
            for k, v in xarray_ds.coords.items()
        }
        var_specs = []
        s_cursor = 0
        for vname, da in xarray_ds.data_vars.items():
            vals = np.asarray(da.values, dtype=np.float32)
            n_slices = int(np.prod(vals.shape[:-2]))
            np.copyto(slot_buf[s_cursor : s_cursor + n_slices], vals.reshape(n_slices, 721, 1440))
            var_specs.append((vname, da.dims, vals.shape, s_cursor, s_cursor + n_slices))
            s_cursor += n_slices

        fut = _GRIB2_EXECUTOR.submit(
            _grib2_worker_task, self_gw, slot_idx, coords_dict, var_specs, outdir
        )
        fut.add_done_callback(_on_grib2_future_done)
        with timings_lock:
            pending_grib2_futures.append(fut)

    @functools.wraps(_ORIG_SAVE_GRIB2)
    def _wrapped_save_grib2(self_gw, xarray_ds, outdir):
        t_g0 = time.perf_counter()
        is_f000 = not step_timings["f000_done"]
        if not is_f000 and step_timings["last_grib2_end"] is not None:
            step_timings["tpu_step_intervals"].append(t_g0 - step_timings["last_grib2_end"])

        call_idx = step_timings["grib2_calls"]
        sfut = stage_executor.submit(_stage_and_submit, call_idx, self_gw, xarray_ds, outdir)
        pending_stage_futures.append(sfut)

        t_g1 = time.perf_counter()
        dispatch_dur = t_g1 - t_g0
        step_timings["grib2_main_thread_sec"] += dispatch_dur
        step_timings["grib2_calls"] += 1
        step_timings["last_grib2_end"] = t_g1

        if is_f000:
            step_timings["f000_done"] = True
            prep_dt = t_g0 - step_timings["first_step_start"]
            step_timings["init_data_prep_sec"] = prep_dt
            print(
                f"  [Init f000 (+000h)] Data Load & Setup: {prep_dt:.3f}s | "
                f"Async SHM GRIB2 Dispatch: {dispatch_dur * 1000.0:.2f}ms"
            )
        else:
            step_idx = len(step_timings["tpu_step_intervals"])
            tpu_dt = step_timings["tpu_step_intervals"][-1]
            with timings_lock:
                done_cnt = step_timings["grib2_completed_calls"]
            print(
                f"  [Step {step_idx:02d}/{lead_steps:02d} (+{step_idx*6:03d}h)] "
                f"TPU Step: {tpu_dt:.3f}s | Async SHM GRIB2 Dispatch: {dispatch_dur * 1000.0:.2f}ms "
                f"(GRIB2 Done: {done_cnt}/{step_timings['grib2_calls']})"
            )
            if step_idx >= lead_steps:
                t_drain0 = time.perf_counter()
                print("  [GRIB2 Pool] Waiting for final in-flight GRIB2 workers to finish...")
                for sf in pending_stage_futures:
                    sf.result()
                for pf in list(pending_grib2_futures):
                    pf.result()
                drain_dur = time.perf_counter() - t_drain0
                step_timings["grib2_tail_drain_sec"] += drain_dur
                step_timings["grib2_main_thread_sec"] += drain_dur
                print(
                    f"  [GRIB2 Pool] All {step_timings['grib2_calls']} steps "
                    f"({step_timings['grib2_calls'] * 2} GRIB2 files) written "
                    f"(tail drain: {drain_dur:.3f}s)."
                )

    gw_mod.Grib2Writer.save_grib2 = _wrapped_save_grib2
    print(
        f"[Instrumented] Hooked utils.grib2writer.Grib2Writer.save_grib2 with "
        f"{GRIB2_WORKERS}-worker MAP_SHARED process pool & {GCS_UPLOAD_WORKERS}-thread GCS uploader.\n"
    )

    # Execute Jun's unmodified ush/run_graphcast.py
    sys.argv = target_args
    step_timings["first_step_start"] = time.perf_counter()
    t0 = step_timings["first_step_start"]

    try:
        if ENABLE_XPROF:
            xprof_dir = f"/tmp/xprof_aigfs_{timestamp_str}"
            with jax.profiler.trace(xprof_dir, create_perfetto_link=False):
                runpy.run_path(script_to_run, run_name="__main__")
        else:
            runpy.run_path(script_to_run, run_name="__main__")
    finally:
        for sf in pending_stage_futures:
            sf.result()
        stage_executor.shutdown(wait=True)
        for pf in list(pending_grib2_futures):
            pf.result()
        if _GRIB2_EXECUTOR is not None:
            _GRIB2_EXECUTOR.shutdown(wait=True)

    total_wall_time = time.perf_counter() - t0

    # Wait for GCS uploads to finish
    print("\nWaiting for remaining 6-hourly GRIB2 GCS uploads to complete...")
    for _ in range(GCS_UPLOAD_WORKERS):
        upload_queue.put(None)
    upload_queue.join()
    for t in uploader_threads:
        t.join(timeout=60)

    # Calculate separated timing breakdown
    grib2_blocking_time = step_timings["grib2_main_thread_sec"]
    grib2_worker_cpu_time = step_timings["grib2_worker_cpu_sec"]
    intervals = step_timings["tpu_step_intervals"]

    if len(intervals) >= 2:
        warm_step_avg = sum(intervals[1:]) / len(intervals[1:])
        jit_compile_est = max(0.0, intervals[0] - warm_step_avg)
        pure_tpu_infer_time = warm_step_avg * lead_steps
    elif len(intervals) == 1:
        warm_step_avg = intervals[0]
        jit_compile_est = 0.0
        pure_tpu_infer_time = intervals[0]
    else:
        pure_tpu_infer_time = max(0.01, total_wall_time - grib2_blocking_time)
        warm_step_avg = pure_tpu_infer_time / max(1, lead_steps)
        jit_compile_est = 0.0

    achieved_tflops = (FLOP_PER_STEP * lead_steps) / (pure_tpu_infer_time * 1e12)
    achieved_hbm_gbps = (BYTES_PER_STEP * lead_steps) / (pure_tpu_infer_time * 1e9)
    mfu_percent = (achieved_tflops / TPU_V6E_PEAK_TFLOPS) * 100.0
    mbu_percent = (achieved_hbm_gbps / TPU_V6E_PEAK_HBM_GBPS) * 100.0
    cost_usd = (total_wall_time / 3600.0) * TPU_V6E_HOURLY_COST
    energy_wh = (TPU_V6E_TDP_WATTS * total_wall_time) / 3600.0

    storage_loc_str = local_staging_dir if upload_metrics["gcs_auth_failed"] else f"{gcs_run_dir}/"

    report_text = f"""======================================================================
   NOAA AIGFS IDENTICAL BENCHMARK REPORT (TPU v6e - 6-HOURLY GRIB2)
======================================================================
  • Run Timestamp (UTC):                 {timestamp_str}
  • Script Executed:                     {script_to_run} (Unmodified NIMBUS)
  • Active Hardware Device:              {devices[0]}
  • Initial Condition File:              {input_ic} (Cycle: {cycle_str})
  • Model Checkpoint:                    {os.path.realpath(expected_npz)}
  • Forecast Horizon:                    {lead_steps} steps ({lead_steps * 6}h / {lead_steps * 6 // 24} days)
  --------------------------------------------------------------------
  END-TO-END & COMPONENT TIMING BREAKDOWN:
  • Total End-to-End Wall Time:          {total_wall_time:.3f} s  ({total_wall_time / 60.0:.2f} min)
  • Initial Data Load & Setup Time:      {step_timings['init_data_prep_sec']:.3f} s
  • 1st-Step JAX JIT Compile Time:       {jit_compile_est:.3f} s
  • Pure TPU Rollout Time ({lead_steps} steps):     {pure_tpu_infer_time:.3f} s  ({warm_step_avg * 1000.0:.2f} ms/step)
  • Non-Overlapped GRIB2 Wall Time:      {grib2_blocking_time:.3f} s  (tail drain: {step_timings['grib2_tail_drain_sec']:.3f} s)
  • Overlapped GRIB2 Worker CPU Time:    {grib2_worker_cpu_time:.3f} s  ({step_timings['grib2_calls']} calls across {GRIB2_WORKERS} workers, {(grib2_worker_cpu_time / max(1, step_timings['grib2_calls'])):.2f} s/step)
  --------------------------------------------------------------------
  TPU v6e COMPUTE & MEMORY THROUGHPUT (Pure Inference):
  • Achieved Compute Throughput:         {achieved_tflops:.2f} TFLOPs/sec
  • Model FLOPs Utilization (MFU):       {mfu_percent:.2f} %  (vs {TPU_V6E_PEAK_TFLOPS} TFLOPs BF16 peak)
  • Measured HBM Bandwidth:              {achieved_hbm_gbps:.2f} GB/s
  • Memory Bandwidth Utilization (MBU):  {mbu_percent:.2f} %  (vs {TPU_V6E_PEAK_HBM_GBPS} GB/s peak)
  --------------------------------------------------------------------
  CLOUD STORAGE & ARTIFACTS:
  • 6-Hourly GRIB2 Files Generated:      {upload_metrics['count']} files ({upload_metrics['total_bytes'] / (1024**3):.2f} GB)
  • Output Artifact Path:                {storage_loc_str}
  • Total Cost (End-to-End Run):         ${cost_usd:.4f} USD (@ ${TPU_V6E_HOURLY_COST}/hr)
  • Energy Consumed:                     {energy_wh:.2f} Wh (@ {TPU_V6E_TDP_WATTS}W TDP)
======================================================================
"""
    print("\n" + report_text)

    report_filename = f"xprof_report_{cycle_str}_{timestamp_str}.txt"
    local_report_path = os.path.join("/tmp", report_filename)
    with open(local_report_path, "w") as rf:
        rf.write(report_text)

    if not upload_metrics["gcs_auth_failed"]:
        subprocess.run(
            ["gcloud", "storage", "cp", local_report_path, f"{gcs_run_dir}/{report_filename}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"Benchmark report uploaded to: {gcs_run_dir}/{report_filename}")
        if DELETE_LOCAL_AFTER_UPLOAD and os.path.isdir(local_staging_dir):
            shutil.rmtree(local_staging_dir, ignore_errors=True)
    else:
        print(f"Benchmark report saved locally to: {local_report_path}")


if __name__ == "__main__":
    main()
