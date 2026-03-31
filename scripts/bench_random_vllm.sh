# Run a small, repeatable vLLM serving benchmark sweep using the built-in random dataset, varying
# input and output token lengths, and saving one result file per configuration.

MODEL="mistralai/Ministral-3-3B-Instruct-2512"
NUM_PROMPTS=100
REQUEST_RATE=10
MAX_CONC=32
PREFIX_LEN=100
RESULTS_DIR="./results/vllm_random_results"

mkdir -p $RESULTS_DIR

INPUT_LENS=(2500)
OUTPUT_LENS=(50 250 500)

echo "Start benchmark sweep..."

for in_len in "${INPUT_LENS[@]}"; do
    for out_len in "${OUTPUT_LENS[@]}"; do

        FILE_NAME="res_in${in_len}_out${out_len}.json"

        echo "-------------------------------------------------------"
        echo "Executing: Input $in_len | Output $out_len"

        vllm bench serve \
            --model $MODEL \
            --num-prompts $NUM_PROMPTS \
            --dataset-name random \
            --random-input-len $in_len \
            --random-output-len $out_len \
            --request-rate $REQUEST_RATE \
            --max-concurrency $MAX_CONC \
            --random-prefix-len $PREFIX_LEN \
            --result-dir $RESULTS_DIR \
            --result-filename $FILE_NAME \
            --save-result

    done
done

echo "Benchmark completed."