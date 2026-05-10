from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_aesthetics.pipelines.preprocessing import run_preprocessing
from src.face_aesthetics.pipelines.preprocessing import run_preprocessing_gpu_runtime_check


def _str_to_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run morphometric preprocessing pipeline.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", type=_str_to_bool, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--overwrite", type=_str_to_bool, default=None)
    parser.add_argument("--preferred-execution-provider", type=str, default=None)
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--strict-gpu", type=_str_to_bool, default=None)
    parser.add_argument("--gpu-readiness-check", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    overrides = {
        "limit": args.limit,
        "offset": args.offset,
        "batch_size": args.batch_size,
        "resume": args.resume,
        "num_workers": args.num_workers,
        "overwrite": args.overwrite,
        "preferred_execution_provider": args.preferred_execution_provider,
        "device_id": args.device_id,
        "strict_gpu_mode": args.strict_gpu,
        "smoke_mode": args.smoke,
    }
    if (
        args.strict_gpu is None
        and args.preferred_execution_provider is not None
        and args.preferred_execution_provider.strip() == "CUDAExecutionProvider"
    ):
        overrides["strict_gpu_mode"] = True
    if args.gpu_readiness_check:
        report = run_preprocessing_gpu_runtime_check(project_root=PROJECT_ROOT, overrides=overrides)
        print("gpu_runtime_check=reports/metrics/gpu_runtime_check.json")
        print(f"onnxruntime_version={report.get('onnxruntime_version')}")
        print(f"available_providers={report.get('available_providers')}")
        print(f"requested_providers={report.get('requested_providers')}")
        print(f"session_providers={report.get('session_providers')}")
        print(f"selected_execution_provider={report.get('selected_execution_provider')}")
        print(f"gpu_provider_active={report.get('gpu_provider_active')}")
        print(f"missing_cuda_runtime_libraries={report.get('missing_cuda_runtime_libraries')}")
        print(f"preload_dlls_supported={report.get('preload_dlls_supported')}")
        print(f"preload_dlls_success={report.get('preload_dlls_success')}")
        print(f"preload_dlls_error={report.get('preload_dlls_error')}")
        print(f"error={report.get('error')}")
        return

    artifacts = run_preprocessing(project_root=PROJECT_ROOT, overrides=overrides)
    summary = artifacts.summary
    provider = summary.get("provider_diagnostics", {})

    print(f"preprocessed_manifest={artifacts.output_manifest_path}")
    print(f"summary_json={artifacts.summary_path}")
    print(f"input_total_rows={summary['input_total_rows']}")
    print(f"processed_rows={summary['processed_rows']}")
    print(f"successful_rows={summary['successful_rows']}")
    print(f"rejected_rows={summary['rejected_rows']}")
    print(f"error_rows={summary['error_rows']}")
    print(f"quality_pass_true_count={summary['quality_pass_true_count']}")
    print(f"processing_status_counts={summary['processing_status_counts']}")
    print(f"rejection_counts={summary['rejection_counts']}")
    print(f"aligned_nonempty_count={summary['aligned_nonempty_count']}")
    print(f"landmarks_nonempty_count={summary['landmarks_nonempty_count']}")
    print(f"alignment_attempted_count={summary.get('alignment_attempted_count')}")
    print(f"alignment_applied_count={summary.get('alignment_applied_count')}")
    print(f"alignment_improved_quality_count={summary.get('alignment_improved_quality_count')}")
    print(f"shards_written={summary['shards_written']}")
    print(f"aligned_examples={summary['aligned_examples']}")
    print(f"available_providers={provider.get('available_providers')}")
    print(f"selected_execution_provider={provider.get('selected_execution_provider')}")
    print(f"fallback_to_cpu={provider.get('fallback_to_cpu')}")
    print(f"device_id={provider.get('device_id')}")
    print(f"actual_model_providers={provider.get('actual_model_providers')}")
    print(f"three_d_summary=reports/metrics/preprocessing_3d_summary.json")


if __name__ == "__main__":
    main()
