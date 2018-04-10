#!/usr/bin/env bash

set -e

#
# Boilerplate.
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/scripts/functions.sh"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

NUM_EPOCHS="100"

#
# Test collection.
#

COLLECTION_PATH="${SCRIPT_DIR}/test_data/cranfield_collection"
check_directory "${COLLECTION_PATH}"

QUERIES_PATH="${COLLECTION_PATH}/cranfield.topics"
QREL_PATH="${COLLECTION_PATH}/cranfield.qrel"

TOPICS_NAME="$(basename ${QUERIES_PATH})"

#
# Scripting arguments.
#

SCRATCH_DIR="${1:-}"
check_not_empty "${SCRATCH_DIR}" "SCRATCH_DIR"

if [[ -d "${SCRATCH_DIR}" ]]; then
    1>&2 echo "Scratch directory ${SCRATCH_DIR} already exists; running incremental steps only."
fi

mkdir -p "${SCRATCH_DIR}"

# ADD README here.

#
# Indexing.
#

print_section_title "Indexing collection"

INDEX_PARAMS="${SCRATCH_DIR}/indri.param"
STOPWORD_LIST="${SCRATCH_DIR}/stopwords.dst"
INDEX_DIR="${SCRATCH_DIR}/index"

if [[ ! -d "${INDEX_DIR}" ]]; then
    print_subsection_title "Building index"

    build_index \
        "${INDEX_PARAMS}" \
        "${STOPWORD_LIST}" \
        "${INDEX_DIR}" \
        "${COLLECTION_PATH}/cranfield.trectext"

    1>&2 echo
fi

print_subsection_title "Index statistics"

PyndriStatistics --index "${INDEX_DIR}"

#
# Query-likelihood model (exact matching).
#

print_section_title "Ranking using Query-likelihood Model (exact matching)."

INDRI_RUNS_DIR="${SCRATCH_DIR}/indri/runs"

mkdir -p "${INDRI_RUNS_DIR}"

print_subsection_title "Ranking using QLM."

declare -A QLM_ARGS

for LAMBDA in auto; do
    QLM_ARGS[jm_${LAMBDA}]="--smoothing_method jm --smoothing_param ${LAMBDA}"
done

for MU in auto; do
    QLM_ARGS[dirichlet_${MU}]="--smoothing_method dirichlet --smoothing_param ${MU}"
done

for MU in auto; do
    QLM_ARGS[dirichlet_${MU}_prf]="--smoothing_method dirichlet --smoothing_param ${MU} --prf"
done

for LAMBDA in auto; do
    QLM_ARGS[jm_${LAMBDA}_prf]="--smoothing_method jm --smoothing_param ${LAMBDA} --prf"
done

for RETRIEVAL_CONFIG_NAME in "${!QLM_ARGS[@]}"; do
    OUTPUT_PREFIX="${INDRI_RUNS_DIR}/indri-${RETRIEVAL_CONFIG_NAME}"

    if [[ -f "${OUTPUT_PREFIX}-${TOPICS_NAME}" ]]; then
        continue;
    fi

    PyndriQuery \
        --loglevel warning \
        --queries "${QUERIES_PATH}" \
        --index "${INDEX_DIR}" \
        ${QLM_ARGS[${RETRIEVAL_CONFIG_NAME}]} \
        --top_k 1000 \
        "${OUTPUT_PREFIX}"
done

function generate_qlm_results() {
    print_columns "QLM with Jelinek-Mercer smoothing (lambda = 0.5):" "$(compute_retrieval_effectiveness ${INDRI_RUNS_DIR}/indri-jm_auto-${TOPICS_NAME}) MAP"
    print_columns "QLM with Jelinek-Mercer smoothing (lambda = 0.5) w/ PRF:" "$(compute_retrieval_effectiveness ${INDRI_RUNS_DIR}/indri-jm_auto_prf-${TOPICS_NAME}) MAP"
    print_columns "QLM with Dirichlet smoothing (mu = avg_doc_length):" "$(compute_retrieval_effectiveness ${INDRI_RUNS_DIR}/indri-dirichlet_auto-${TOPICS_NAME}) MAP"
    print_columns "QLM with Dirichlet smoothing (mu = avg_doc_length) w/ PRF:" "$(compute_retrieval_effectiveness ${INDRI_RUNS_DIR}/indri-dirichlet_auto_prf-${TOPICS_NAME}) MAP"
}

generate_qlm_results

#
# Neural Vector Space Models (semantic matching).
#

print_section_title "Ranking using Neural Vector Space Models (semantic matching)."

NVSM_DIR="${SCRATCH_DIR}/nvsm"
NVSM_MODELS_DIR="${NVSM_DIR}/models"
NVSM_RUNS_DIR="${NVSM_DIR}/runs"

mkdir -p "${NVSM_MODELS_DIR}"

for MODEL_NAME in "${NVSM_MODELS[@]}"; do
    if [[ ! -f "${NVSM_MODELS_DIR}/${MODEL_NAME}_meta" ]]; then
        print_subsection_title "Training ${MODEL_NAME}."

        CUNVSM_TRAIN_LOG_FILE="${NVSM_MODELS_DIR}/${MODEL_NAME}.log"

        1>&2 echo "Writing ${MODEL_NAME} training logs to ${CUNVSM_TRAIN_LOG_FILE}."
        1>&2 echo

        train_nvsm \
            "${MODEL_NAME}" \
            "${NVSM_MODELS_DIR}/${MODEL_NAME}" \
            "${INDEX_DIR}" &> "${CUNVSM_TRAIN_LOG_FILE}"
    fi
done

mkdir -p "${NVSM_RUNS_DIR}"

NUM_LAST_EPOCHS="1"  # The number of last training epochs to consider during querying.

for MODEL_NAME in "${NVSM_MODELS[@]}"; do
    print_subsection_title "Ranking using ${MODEL_NAME}."

    for EPOCH in $(seq $(( ${NUM_EPOCHS} - ${NUM_LAST_EPOCHS} + 1 )) ${NUM_EPOCHS}); do
        MODEL_BIN="${NVSM_MODELS_DIR}/${MODEL_NAME}_${EPOCH}.hdf5"
        check_file "${MODEL_BIN}"

        OUTPUT_PREFIX="${NVSM_RUNS_DIR}/${MODEL_NAME}_${EPOCH}"

        if [[ -f "${OUTPUT_PREFIX}-${TOPICS_NAME}" ]]; then
            continue;
        fi

        CUNVSM_QUERY_LOG_FILE="${OUTPUT_PREFIX}.log"

        1>&2 echo "Writing ${MODEL_NAME} (epoch ${EPOCH}) query logs to ${CUNVSM_QUERY_LOG_FILE}."
        1>&2 echo

        build/py/cuNVSMQuery \
            --loglevel warning \
            --num_workers 8 \
            --topics "${COLLECTION_PATH}/cranfield.topics" \
            --index "${INDEX_DIR}" \
            ${NVSM_QUERY_ARGS[${MODEL_NAME}]} \
            --top_k 1000 \
            "${MODEL_BIN}" \
            "${OUTPUT_PREFIX}" &> "${CUNVSM_QUERY_LOG_FILE}"
    done
done

function generate_nvsm_results() {
    for MODEL_NAME in "${NVSM_MODELS[@]}"; do
        print_columns "${MODEL_NAME} (window_size = 10, word_dim = 300, doc_dim = 256):" "$(compute_retrieval_effectiveness ${NVSM_RUNS_DIR}/${MODEL_NAME}_100-${TOPICS_NAME}) MAP"
    done
}

generate_nvsm_results

#
# Combinations of matching features.
#

print_section_title "Ranking using combinations of QLM and NVSM."

COMBINATIONS_DIR="${SCRATCH_DIR}/combinations"
COMBINATIONS_RUNS_DIR="${COMBINATIONS_DIR}/runs"

mkdir -p "${COMBINATIONS_RUNS_DIR}"

function build_combination() {
    FIRST_RUN="${1:-}"
    check_not_empty "${FIRST_RUN}"
    check_file "${FIRST_RUN}"

    SECOND_RUN="${2:-}"
    check_not_empty "${SECOND_RUN}"
    check_file "${SECOND_RUN}"

    MODE="${3:-}"
    check_not_empty "${MODE}"
    check_valid_option "supervised" "unsupervised" "${MODE}"

    if [[ "${MODE}" == "supervised" ]]; then
        ARGS="--qrel ${QREL_PATH} --num_folds 20 --alpha_stepsize 0.01"
    elif [[ "${MODE}" == "unsupervised" ]]; then
        ARGS="--alpha 0.5"
    fi

    OUTPUT_RUN_NAME="$(basename ${FIRST_RUN})-$(basename ${SECOND_RUN})-${MODE}"
    OUTPUT_RUN="${COMBINATIONS_RUNS_DIR}/${OUTPUT_RUN_NAME}"

    if [[ -f "${OUTPUT_RUN}" ]]; then
        return
    fi

    python ${SCRIPT_DIR}/py/combine_runs.py \
        --loglevel error \
        --runs "${FIRST_RUN}" "${SECOND_RUN}" \
        --score_normalizer standardize \
        "${OUTPUT_RUN}" \
        ${ARGS}
}

BASE_RUNS=(
    "${INDRI_RUNS_DIR}/indri-jm_auto-${TOPICS_NAME}"
    "${INDRI_RUNS_DIR}/indri-jm_auto_prf-${TOPICS_NAME}"
    "${INDRI_RUNS_DIR}/indri-dirichlet_auto-${TOPICS_NAME}"
    "${INDRI_RUNS_DIR}/indri-dirichlet_auto_prf-${TOPICS_NAME}"
)

BASE_RUN_NAMES=(
    "QLM (Jelinek-Mercer)"
    "QLM (Jelinek-Mercer) w/ PRF"
    "QLM (Dirichlet)"
    "QLM (Dirichlet) w/ PRF"
)

function generate_combinations_results() {
    MODE="${1:-}"

    for MODEL_NAME in "${NVSM_MODELS[@]}"; do
        for IDX in $(seq 0 $(( ${#BASE_RUN_NAMES[@]} - 1 )) ); do
            BASE_RUN="${BASE_RUNS[${IDX}]}"
            BASE_RUN_NAME="${BASE_RUN_NAMES[${IDX}]}"

            print_columns "${MODEL_NAME} + ${BASE_RUN_NAME}:" "$(compute_retrieval_effectiveness ${COMBINATIONS_RUNS_DIR}/$(basename ${BASE_RUN})-${MODEL_NAME}_100-${TOPICS_NAME}-${MODE}) MAP"
        done
    done
}

function build_combinations() {
    MODE="${1:-}"

    for MODEL_NAME in "${NVSM_MODELS[@]}"; do
        for BASE_RUN in "${BASE_RUNS[@]}"; do
            build_combination "${BASE_RUN}" "${NVSM_RUNS_DIR}/${MODEL_NAME}_100-${TOPICS_NAME}" "${MODE}"
        done
    done
}

# The "build_combinations" function also supports supervised combinations. However, this is a bit excessive for this example.
for MODE in "unsupervised"; do
    print_subsection_title "Computing ${MODE} combinations."

    build_combinations "${MODE}"
    generate_combinations_results "${MODE}"
done

print_section_title "Results overview"
print_subsection_title "Query-likelihood Model (exact matching)."
generate_qlm_results
1>&2 echo
print_subsection_title "Neural Vector Space Models (semantic matching)."
generate_nvsm_results
1>&2 echo
print_subsection_title "QLM + NVSM (lexical + semantic matching)."
generate_combinations_results "unsupervised"
