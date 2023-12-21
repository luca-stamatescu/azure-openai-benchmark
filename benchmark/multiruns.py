"""
This module can be used to run multiple runs of the benchmarking script with different permutations of parameters. 
Since this can be run at the command line, it also allows the running of testing across multiple deployments at the same time.

To use:
# Set the api key for the environment, e.g.
> export OPENAI_API_KEY=<your key>

# Run the tool for a single batch of runs (e.g. a cold-start warmup, followed by a combination of 2x workload-token-profiles and 2x concurrency values = 5x total runs)
> python -m benchmark.multiruns --api-base-endpoint https://<YOUR_ENDPOINT>.openai.azure.com/ --deployment <MODEL_DEPLOYMENT> --log-save-dir logs --warmup-per-run 15 --cold-start-warmup 300 --aggregation-window 180 --concurrency-values 1,4 --workload-token-profiles 100-100,3000-500

# Run the tool for multiple batches of runs (e.g. 3x batches, with their start times 1 hour apart)
> python -m benchmark.multiruns --api-base-endpoint https://<YOUR_ENDPOINT>.openai.azure.com/ --deployment <MODEL_DEPLOYMENT> --log-save-dir logs --warmup-per-run 15 --cold-start-warmup 300 --aggregation-window 180 --concurrency-values 1,4 --workload-token-profiles 100-100,3000-500 --num-batches 3 --batch-repeat-delay 3600

# Combine the logs with the combine_logs tool
> python -m benchmark.combine_logs logs logs/combined_runs.csv
"""


import argparse
import copy
import itertools
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Type


# Create deployment config dataclasses that align to the different python entrypoints
@dataclass
class OpenAIDeploymentConfig:
    # Dynamic config
    api_key_env: str
    deployment: str
    api_base_endpoint: str
    log_save_dir: Optional[str]
    # Static config
    python_entrypoint: str = "benchmark.bench load"
    api_version: str = "2023-05-15"
    completions: int = 1

DeploymentConfigType = Type[OpenAIDeploymentConfig]

@dataclass
class BenchmarkConfig:
    # Dynamic config
    context_tokens: int
    max_tokens: int
    duration: int
    # Static config
    shape_profile: str = "custom"
    aggregation_window: int = 60

def generate_config_set(
    deployment_config: DeploymentConfigType, 
    benchmark_configs: Iterable[BenchmarkConfig],
    concurrency_values: Iterable[int],
    cold_start_warmup: int,
) -> Iterable[tuple[DeploymentConfigType, BenchmarkConfig, int]]:
    """
    Combines the deployment & benchmark configs with a list of various 
    concurrency values to create all possible permutations, 
    plus an initial warmup config to be run prior to all benchmarks.

    Args:
        deployment_config: Config of deployment details
        benchmark_configs: Configs of different workload params to run
        concurrency_values: Iterable of different concurrency values.
        cold_start_warmup: Number of seconds to run a full-load warm-up.

    Returns:
        List of execution strings to be run (in order) by the benchmarking script.
    """
    # Create warmup run with 10 clients and 60 RPM @ 200 total tokens - equates to max of 1.2M TPM
    warmup_bench_config = BenchmarkConfig(
        context_tokens=1000,
        max_tokens=1000,
        aggregation_window=cold_start_warmup,
        duration=cold_start_warmup
    )
    # Do not save logs for this run
    warmup_deployment_config = copy.copy(deployment_config)
    warmup_deployment_config.log_save_dir = None
    warmup_config = [warmup_deployment_config, warmup_bench_config, 10]

    # Combine all permutations of deployment, benchmark and concurrency values
    permutations = itertools.product(*[benchmark_configs, concurrency_values])
    permutations = [[deployment_config, *config] for config in list(permutations)]

    return [warmup_config, *permutations]

def config_to_execution_str(
    deployment_config: DeploymentConfigType, 
    benchmark_config: BenchmarkConfig,
    clients: int,
) -> str:
    """
    Combines configs into a single execution string, ready for execution by the
    benchmarking CLI.
    """
    cmd = f"python3 -m {deployment_config.python_entrypoint}"
    # Deployment config
    for param, value in deployment_config.__dict__.items():
        if param in ["python_entrypoint", "log_save_dir"]:
            continue
        elif param == "api_base_endpoint":
            cmd += f" {value}"
        else:
            cmd += f" --{param.replace('_', '-')} {value}"
    # Benchmark config
    for param, value in benchmark_config.__dict__.items():
        cmd += f" --{param.replace('_', '-')} {value}"
    # Logs save dir
    cmd += " --output-format jsonl"
    if deployment_config.log_save_dir:
        cmd += f" --log-save-dir {deployment_config.log_save_dir}/{deployment_config.deployment}"
    # Add clients 
    assert clients > 0
    cmd += f" --clients {clients}"
    return cmd


def generate_configs_from_simple_input(
    api_base_endpoint: str,
    deployment: str,
    log_save_dir: str,
    workload_token_profiles: Iterable[list[int, int]],
    concurrency_values: Iterable[int],
    aggregation_window: tuple[int, int] = 180,
    warmup_per_run: int = 15,
    cold_start_warmup: int = 300,
    api_key_env: str = "OPENAI_API_KEY",
) -> Iterable[tuple[DeploymentConfigType, BenchmarkConfig, int]]:
    """
    Generates a set of configs for the given workload profiles and concurrency 
    values.

    Args:
        api_key_env: Environment variable that contains the API KEY.
        api_base_endpoint: Azure OpenAI deployment base endpoint.
        deployment: Azure OpenAI deployment name.
        log_save_dir: Will save all logs to this directory.
        workload_token_profiles: List of [context, max_token] pairs.
        concurrency_values: List of concurrency values to test for each workload.
        aggregation_window: Length of time to collect and aggregate statistcs. Defaults to 180.
        warmup_per_run: Seconds spent warming up the endpoint per run. Defaults to 15.
        pre_test_warmup: Seconds to run load through the endpoint prior to any actual testing. Defaults to 300.
    """

    deployment_config = OpenAIDeploymentConfig(
        api_key_env=api_key_env,
        deployment=deployment,
        api_base_endpoint=api_base_endpoint,
        log_save_dir=log_save_dir
    )
    benchmark_configs = [
        BenchmarkConfig(
            context_tokens=context_tokens,
            max_tokens=max_tokens,
            aggregation_window=aggregation_window,
            duration=warmup_per_run+aggregation_window
        )
        for context_tokens, max_tokens in workload_token_profiles
    ]
    all_configs = generate_config_set(
        deployment_config=deployment_config,
        benchmark_configs=benchmark_configs,
        concurrency_values=concurrency_values,
        cold_start_warmup=cold_start_warmup
    )
    return all_configs

def run_configs(configs: Iterable[tuple[DeploymentConfigType, BenchmarkConfig, int]]):
    """
    Runs each config in the given list of configs.
    """
    try:
        for trial, (deployment_config, benchmark_config, clients) in enumerate(configs):
            exec_str = config_to_execution_str(deployment_config, benchmark_config, clients)
            print(f"Starting trial {trial+1} of {len(configs)}")
            process = subprocess.Popen(shlex.split(exec_str), shell=False)
            _return_code = process.wait()
    except KeyboardInterrupt as _kbi:
        print("Keyboard interrupt detected. Exiting...")
        process.kill()
        raise _kbi
    except Exception as exc:
        process.kill()
        raise exc
    return

# Create argparse parser for run_configs
def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-workload benchmarking.")
    parser.add_argument("--api-base-endpoint", type=str, help="Azure OpenAI deployment base endpoint.", required=True)
    parser.add_argument("--deployment", type=str, help="Azure OpenAI deployment name.", required=True)
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY", help="Environment variable that contains the API KEY.")
    parser.add_argument("--log-save-dir", type=str, help="If provided, will save stddout to this directory. Filename will include important run parameters.")
    parser.add_argument("--warmup-per-run", type=int, default=15, help="Seconds spent warming up the endpoint per run. Defaults to 15.")
    parser.add_argument("--cold-start-warmup", type=int, default=300, help="Seconds to run load through the endpoint prior to any actual testing. Defaults to 300.")
    parser.add_argument("--aggregation-window", type=int, default=180, help="Length of time to collect and aggregate statistcs. Defaults to 180.")
    parser.add_argument('--concurrency-values', type=str, default="1,2,4", help="List of concurrency values to test for each workload. Defaults to '1,2,4'.")
    parser.add_argument("--workload-token-profiles", type=str, default="100-200,200-500", help="List of comma-delimited `context-max_token` pairs to test. Defaults to '100-200,200-500'.")
    parser.add_argument("--num-batches", type=int, default=1, help="Number of times to repeat the full batch of benchmarks (including cold-start-warmup). Defaults to 1.")
    parser.add_argument("--batch-start-interval", type=int, default=3600, help="Seconds to wait between the start of each batch of runs (NOT from the end of one to the start of the next). Defaults to 3600 seconds (1 hour).")
    return parser.parse_args()

def main():
    args = parse_args()
    concurrency_values = [int(item) for item in args.concurrency_values.split(',')]
    workload_token_profiles = [[int(val) for val in item.split('-')] for item in args.workload_token_profiles.split(',')]
    configs = generate_configs_from_simple_input(
        api_key_env=args.api_key_env,
        api_base_endpoint=args.api_base_endpoint,
        deployment=args.deployment,
        log_save_dir=args.log_save_dir,
        workload_token_profiles=workload_token_profiles,
        concurrency_values=concurrency_values,
        aggregation_window=args.aggregation_window,
        warmup_per_run=args.warmup_per_run,
        cold_start_warmup=args.cold_start_warmup,
    )
    try:
        if args.num_batches == 1:
            # Single-batch runs
            run_configs(configs)
        else:
            # Multi-batch runs
            # Sanity check batch repeat amount
            expected_time_per_batch = sum([bench_cfg.duration + 15 for _, bench_cfg, _ in configs])
            if expected_time_per_batch < args.batch_start_interval:
                print(f"WARNING: Batch repeat delay ({args.batch_start_interval}s) is less than the expected time per batch ({expected_time_per_batch}s). This may result in overlapping runs.")
            start_time = time.time()
            runs_completed = 0
            while runs_completed < args.num_batches:
                print(f"Starting batch {runs_completed+1} of {args.num_batches}")
                run_configs(configs)
                runs_completed += 1
                if runs_completed < args.num_batches:
                    secs_to_wait = int((start_time + args.batch_start_interval * runs_completed) - time.time())
                    if secs_to_wait > 0:
                        print(f"Batch complete. Waiting {secs_to_wait} seconds before starting next batch...")
                        time.sleep(secs_to_wait)
                    else:
                        print(f"WARNING: Batch {runs_completed+1} took longer than {args.batch_start_interval} seconds. Starting next batch immediately.")
            print("All batches complete.")
        return
    except KeyboardInterrupt as _kbi:
        return

main()