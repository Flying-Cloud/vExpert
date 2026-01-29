      
#!/bin/bash
set -x

pkill -9 python
pkill -9 python3
pkill -9 sglang

DEEP_EP_DIR=/icml_submission/DeepEP
cd $DEEP_EP_DIR
NVSHMEM_DIR=/nvshmem/install python setup.py install

DEEP_GEMM_DIR=/icml_submission/DeepGEMM
cd $DEEP_GEMM_DIR
python setup.py install

cd /icml_submission/sglang
cd ./semi-pd-ipc
python setup.py install
cd ..

ulimit -n 524288
ulimit -v unlimited

num_decode=4
max_bs=128
ib_device="mlx5_0,mlx5_1,mlx5_2,mlx5_3"
local_ip="0.0.0.0"
MML=131072
model_path=/public/DeepSeek-V3.1
MODEL_NAME=deepseek

RANK=$1
node_rank=$RANK
echo "RANK: $RANK"

NODE_IP=xx.xx.xx.xx


echo "decode-$node_rank"
command_parts=(
    "GLOO_SOCKET_IFNAME=eth0 SGLANG_SET_CPU_AFFINITY=1 SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=$max_bs MC_TE_METRIC=true PYTHONUNBUFFERED=1 SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL=10000000 SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 "
    "python -m sglang.launch_server --model-path $model_path --trust-remote-code --served-model-name $MODEL_NAME --host $local_ip --dist-init-addr ${master_ip_d0}:5959 --nnodes $num_decode --node-rank $((node_rank-$num_prefill_0))  "
    "--watchdog-timeout 1000000 --context-length $MML --mem-fraction-static 0.8 --max-running-requests $((num_decode * 8 * max_bs)) --decode-log-interval 100 "
    "--disaggregation-mode decode --disaggregation-ib-device $ib_device --deepep-mode low_latency --tp-size $((num_decode * 8)) --dp-size $((num_decode * 8)) --enable-dp-attention --moe-a2a-backend deepep --moe-dense-tp-size 1 --enable-dp-lm-head --enable-two-batch-overlap"
    "--enable-cpu-pinned --pinned-expert-num 256 --num-vexpert-buffer 2 --ep-dispatch-algorithm virtual"
    "--init-expert-location /init_expert_location.pt "
    "--expert-distribution-recorder-mode stat --enable-expert-distribution-metrics --expert-distribution-recorder-buffer-size -1"
    "--disable-shared-experts-fusion --cuda-graph-bs $max_bs --enable-metrics --tokenizer-worker-num 4   2>&1 | tee worker_d0.log"
)

command="${command_parts[*]}"

# To execute the command:
eval "$command"
