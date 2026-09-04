#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <vllm|sglang> <endpoint> <tokenizer-model> <workload.jsonl>" >&2
  exit 2
fi

runtime="$1"
endpoint="$2"
tokenizer_model="$3"
workload="$4"

if [[ "$runtime" != "vllm" && "$runtime" != "sglang" ]]; then
  echo "runtime must be vllm or sglang" >&2
  exit 2
fi
if [[ ! -f "$workload" ]]; then
  echo "workload not found: $workload" >&2
  exit 2
fi

result_root="results/$runtime"
mkdir -p "$result_root"

{
  date -u
  uname -a
  nvidia-smi
  guidellm --version
  python -m pip show "$runtime" torch
  sha256sum "$workload"
} > "$result_root/environment.txt" 2>&1

for repetition in 1 2 3; do
  case "$repetition" in
    1) concurrencies=(4 8 16) ;;
    2) concurrencies=(16 8 4) ;;
    3) concurrencies=(8 4 16) ;;
  esac
  for concurrency in "${concurrencies[@]}"; do
    output_dir="$result_root/c$concurrency"
    mkdir -p "$output_dir"
    guidellm run \
      --backend "kind=openai_http,target=$endpoint" \
      --data "{\"kind\":\"json_file\",\"path\":\"$workload\",\"load_kwargs\":{\"split\":\"train\"}}" \
      --tokenizer "{\"kind\":\"huggingface_auto\",\"model\":\"$tokenizer_model\"}" \
      --profile "kind=concurrent,streams=$concurrency" \
      --constraint kind=max_requests,count=1000 \
      --metrics kind=generative,sample_size=100 \
      --output "kind=json,path=$output_dir/run-$repetition.json" \
      --output "kind=csv,path=$output_dir/run-$repetition.csv" \
      --disable-console-interactive
  done
done
