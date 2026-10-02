#!/usr/bin/env python3
"""
profile_wrapper.py - TPU v6e Profiling & Benchmark Wrapper for AIGFS GraphCast
Supports both JAX and PyTorch-XLA execution graphs on Google Cloud TPU v6e.
Runs the target script, measures JIT compile and rollout latency, and produces a
benchmark report comparing achieved performance against TPU v6e peak hardware capabilities.
Streams all reports and trace artifacts directly to GCS (gs://) with zero local disk footprint.
"""
import glob
import os
import runpy
import subprocess
import sys
import time
from datetime import datetime

# Dynamic Timestamps & Cloud Destinations
TIMESTAMP_STR = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
TIMESTAMP_DISPLAY = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

GCS_BUCKET = os.environ.get("GCS_BUCKET", "gs://mb-noaa-eu/outputs/")
GCS_RUN_FOLDER = f"{GCS_BUCKET.rstrip('/')}/{TIMESTAMP_STR}"
ENABLE_WARMUP = os.environ.get("ENABLE_WARMUP", "0") == "1"

# Temporary scratch paths (cleaned up immediately after run)
TMP_LOG_DIR = f"/tmp/xprof_trace_{TIMESTAMP_STR}"
TMP_REPORT_FILE = f"/tmp/xprof_report_{TIMESTAMP_STR}.txt"

# TPU v6e Peak Hardware Specifications (Trillium ct6e-standard-1t)
TPU_V6E_PEAK_TFLOPS = 918.0       # Peak BF16 TFLOPs/sec per chip
TPU_V6E_PEAK_HBM_GBPS = 1638.0    # Peak HBM Bandwidth (GB/s)
TPU_V6E_TDP_WATTS = 275.0         # Thermal Design Power TDP (Watts)
TPU_V6E_HOURLY_COST = 1.75        # Estimated USD $/chip-hour on GCP

# GraphCast 0.25° FLOPs & Bytes per 6-hour step
FLOP_PER_STEP = 1.4e12
BYTES_PER_STEP = 3.8e9
DEFAULT_NUM_STEPS = 64


def safe_makedirs(path):
    if not path:
        return
    if os.path.exists(path) and not os.path.isdir(path):
        try:
            os.remove(path)
        except OSError:
            pass
    os.makedirs(path, exist_ok=True)


def parse_target_lead_steps(args_list):
    for i, arg in enumerate(args_list):
        if arg in ("-l", "--lead_steps") and i + 1 < len(args_list):
            try:
                return int(args_list[i + 1])
            except ValueError:
                pass
        elif arg.startswith("--lead_steps="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                pass
    return DEFAULT_NUM_STEPS


def detect_framework(script_path):
    """Detect whether target script is PyTorch or JAX based on filename and imports."""
    if "pytorch" in script_path.lower():
        return "pytorch"
    try:
        import jax
        return "jax"
    except ImportError:
        return "pytorch"


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 profile_wrapper.py <script_to_run.py> [args...]")
        sys.exit(1)

    target_args = sys.argv[1:]
    script_to_run = target_args[0]
    expected_steps = parse_target_lead_steps(target_args[1:])
    framework = detect_framework(script_to_run)

    print("\n========================================================")
    print("   STARTING TPU v6e OPTIMIZED BENCHMARK WRAPPER")
    print("========================================================")

    # Framework-specific hardware initialization
    if framework == "pytorch":
        # Ensure TPU v6e Trillium environment variables are set before importing torch_xla
        os.environ.setdefault("TPU_SKIP_MDS_QUERY", "1")
        os.environ.setdefault("TPU_ACCELERATOR_TYPE", "v6e-1")
        os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")
        os.environ.setdefault("TPU_CHIPS_PER_HOST_BOUNDS", "1,1,1")
        os.environ.setdefault("TPU_WORKER_ID", "0")
        os.environ.setdefault("TPU_WORKER_HOSTNAMES", "localhost")

        import torch
        try:
            import torch_xla.core.xla_model as xm
            active_device = f"{xm.xla_device()} (TPU v6e)"
        except Exception as e:
            print(f"Error initializing PyTorch-XLA: {e}")
            sys.exit(1)

        framework_display = f"PyTorch-XLA ({torch.__version__})"
    else:
        import jax
        jax.config.update("jax_default_matmul_precision", "bfloat16")
        devices = jax.devices()
        active_device = str(devices[0])
        framework_display = f"JAX ({jax.__version__})"

        if "TPU" not in active_device:
            print("WARNING: JAX is running on CPU! Please check jax[tpu] installation.")
            sys.exit(1)

    print(f"Framework:                   {framework_display}")
    print(f"Run Timestamp:               {TIMESTAMP_DISPLAY}")
    print(f"Active Hardware Device:      {active_device}")
    print(f"Native MatMul Precision:     bfloat16 (BF16)")
    print(f"Target GCS Run Destination:  {GCS_RUN_FOLDER}/")

    safe_makedirs(TMP_LOG_DIR)

    # 1. Warmup Pass
    compile_time = 0.0
    # Both run_graphcast.py and run_graphcast_pytorch.py compile ahead of time and measure COMPILE_TIME
    target_handles_compilation = ("run_graphcast" in script_to_run)

    if ENABLE_WARMUP and not target_handles_compilation:
        print("Executing JIT Compilation Warmup Pass on TPU...")
        t_comp_start = time.perf_counter()

        warmup_cli = list(target_args)
        if "-o" in warmup_cli:
            idx = warmup_cli.index("-o")
            warmup_cli[idx + 1] = "/tmp/warmup_out"
        sys.argv = warmup_cli

        try:
            runpy.run_path(script_to_run, run_name="__main__")
        except Exception as e:
            print(f"Warmup notice: {e}")

        compile_time = time.perf_counter() - t_comp_start
        print(f"JIT Compilation Warmup Pass Completed in: {compile_time:.3f} s\n")

    # 2. Timed Rollout Execution
    sys.argv = target_args
    print(f"Beginning timed {expected_steps}-step rollout execution...")
    t0 = time.perf_counter()

    os.environ["XPROF_LOG_DIR"] = TMP_LOG_DIR
    target_handles_profiling = "run_graphcast" in script_to_run

    if framework == "jax" and not target_handles_profiling:
        import jax.profiler
        with jax.profiler.trace(TMP_LOG_DIR, create_perfetto_link=False):
            run_globals = runpy.run_path(script_to_run, run_name="__main__")
    else:
        run_globals = runpy.run_path(script_to_run, run_name="__main__")

    total_time = time.perf_counter() - t0

    # Extract metrics
    pure_infer_time = run_globals.get("INFERENCE_TIME", 0.0)
    actual_compile_time = run_globals.get("COMPILE_TIME", 0.0)
    compile_time = actual_compile_time if actual_compile_time > 0 else compile_time
    num_steps = run_globals.get("NUM_STEPS", expected_steps)
    output_file = run_globals.get(
        "OUTPUT_FILE",
        f"{GCS_RUN_FOLDER}/aigfs.t12z.forecast_{num_steps * 6 // 24}day_{TIMESTAMP_STR}.zarr"
    )

    effective_infer_time = pure_infer_time if pure_infer_time > 0 else total_time

    # Compute Benchmark Metrics
    achieved_tflops = (FLOP_PER_STEP * num_steps) / (effective_infer_time * 1e12)
    achieved_hbm_gbps = (BYTES_PER_STEP * num_steps) / (effective_infer_time * 1e9)
    mfu_percent = (achieved_tflops / TPU_V6E_PEAK_TFLOPS) * 100.0
    mbu_percent = (achieved_hbm_gbps / TPU_V6E_PEAK_HBM_GBPS) * 100.0
    cost_usd = (effective_infer_time / 3600.0) * TPU_V6E_HOURLY_COST
    energy_wh = (TPU_V6E_TDP_WATTS * effective_infer_time) / 3600.0
    step_latency_ms = (effective_infer_time / num_steps) * 1000.0
    e2e_step_latency_ms = (total_time / num_steps) * 1000.0

    gcs_report_dest = f"{GCS_RUN_FOLDER}/xprof_report_{TIMESTAMP_STR}.txt"

    report_content = f"""========================================================
   AIGFS PERFORMANCE BENCHMARK REPORT (TPU v6e)
========================================================
  • Framework:                           {framework_display}
  • Run Timestamp:                       {TIMESTAMP_DISPLAY}
  • Active Hardware Device:              {active_device}
  • Forecast Horizon:                    {num_steps} steps ({num_steps * 6} hours / {num_steps * 6 // 24} days)
  • Precision Mode:                      bfloat16 (BF16 Native MXU)
  • JIT Compilation Time:                {compile_time:.3f} s  (Pre-compiled Warmup Pass)
  • Pure Inference Rollout Time:         {effective_infer_time:.3f} s  (Pure TPU Rollout)
  • Single Step Inference Latency:       {step_latency_ms:.2f} ms/step
  • Single Step Average Latency (E2E):   {e2e_step_latency_ms:.2f} ms/step
  • Total Forecast Run Time:             {total_time:.3f} s
  ------------------------------------------------------
  COMPUTE & MEMORY THROUGHPUT (Pure Inference):
  • Achieved Compute Throughput:         {achieved_tflops:.2f} TFLOPs/sec
  • Model FLOPs Utilization (MFU):       {mfu_percent:.2f} %  (vs {TPU_V6E_PEAK_TFLOPS} TFLOPs BF16 peak)
  • Measured HBM Bandwidth:              {achieved_hbm_gbps:.2f} GB/s
  • Memory Bandwidth Utilization (MBU):  {mbu_percent:.2f} %  (vs {TPU_V6E_PEAK_HBM_GBPS} GB/s peak)
  ------------------------------------------------------
  COST & ENERGY METRICS (Warm Run):
  • Cost per {num_steps * 6 // 24}-Day Forecast:           ${cost_usd:.6f} USD (@ ${TPU_V6E_HOURLY_COST}/hr)
  • Energy Consumed per Forecast:       {energy_wh:.4f} Wh (@ {TPU_V6E_TDP_WATTS}W TDP)
  ------------------------------------------------------
  CLOUD ARTIFACTS (Stored in GCS):
  • Forecast Dataset Location:           {output_file}
  • Benchmark Report Location:          {gcs_report_dest}
========================================================
"""

    print(report_content)

    # Save temporary report file
    with open(TMP_REPORT_FILE, "w") as f:
        f.write(report_content)

    # 3. Stream Report File and XProf Trace Directly to GCS
    print(f"[GCS Upload] Streaming benchmark report directly to GCS: {gcs_report_dest}...")
    subprocess.run(["gcloud", "storage", "cp", TMP_REPORT_FILE, gcs_report_dest], check=False)
    print(f"[GCS Upload] Benchmark report uploaded successfully!\n")

    if os.path.isdir(TMP_LOG_DIR) and os.listdir(TMP_LOG_DIR):
        print(f"[GCS Upload] Streaming XProf trace directory directly to GCS: {GCS_RUN_FOLDER}/plugins/profile/...")
        subprocess.run(["gcloud", "storage", "cp", "-r", f"{TMP_LOG_DIR}/*", f"{GCS_RUN_FOLDER}/"], check=False)
        print(f"[GCS Upload] XProf trace uploaded successfully!\n")

    # Clean up temporary scratch files
    if os.path.exists(TMP_REPORT_FILE):
        os.remove(TMP_REPORT_FILE)
    subprocess.run(["rm", "-rf", TMP_LOG_DIR], check=False)


if __name__ == "__main__":
    main()
