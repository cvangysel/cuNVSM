#!/usr/bin/env bash

set -e

#
# Boilerplate.
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/scripts/functions.sh"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

NUM_EPOCHS="15"

#
# Test collection.
#

REUTERS_URL="http://www.daviddlewis.com/resources/testcollections/reuters21578/reuters21578.tar.gz"

#
# Scripting arguments.
#

SCRATCH_DIR="${1:-}"
check_not_empty "${SCRATCH_DIR}" "SCRATCH_DIR"

SCRATCH_DIR=$(realpath ${SCRATCH_DIR})

mkdir -p "${SCRATCH_DIR}"

#
# Fetch collection, process and index.
#

print_section_title "Indexing collection"

REUTERS_TARBALL="${SCRATCH_DIR}/reuters21578.tar.gz"

if [[ ! -f "${REUTERS_TARBALL}" ]]; then
    print_subsection_title "Fetching reuters21578."

    curl "${REUTERS_URL}" > "${REUTERS_TARBALL}"
fi

if [[ ! -d "${SCRATCH_DIR}/raw_data" ]]; then
    print_subsection_title "Unpacking reuters21578."

    mkdir -p "${SCRATCH_DIR}/raw_data"
    cd "${SCRATCH_DIR}/raw_data" && tar xzvf ${REUTERS_TARBALL}
fi

TOPICS_NAME="reuters.topics"
TOPICS_PATH="${SCRATCH_DIR}/processed_data/${TOPICS_NAME}"

if [[ ! -d "${SCRATCH_DIR}/processed_data" ]]; then
    print_subsection_title "Processing reuters21578."

    mkdir -p "${SCRATCH_DIR}/processed_data"

    python ${SCRIPT_DIR}/py/extract_reuters.py \
        ${SCRATCH_DIR}/raw_data/reut2-*.sgm \
        --trectext_out_prefix "${SCRATCH_DIR}/processed_data/reuters" \
        --document_classification_out "${TOPICS_PATH}"
fi

INDEX_PARAMS="${SCRATCH_DIR}/indri.param"
STOPWORD_LIST="${SCRATCH_DIR}/stopwords.dst"
INDEX_DIR="${SCRATCH_DIR}/index"

if [[ ! -d "${INDEX_DIR}" ]]; then
    print_subsection_title "Building index."

    build_index \
        "${INDEX_PARAMS}" \
        "${STOPWORD_LIST}" \
        "${INDEX_DIR}" \
        "${SCRATCH_DIR}/processed_data/reuters_1.trectext"
fi

#
# Neural Vector Space Models (semantic matching).
#

print_section_title "Training Neural Vector Space Models."

NVSM_DIR="${SCRATCH_DIR}/nvsm"
NVSM_MODELS_DIR="${NVSM_DIR}/models"

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

print_section_title "Plotting document embeddings."

PLOTS_DIR="${SCRATCH_DIR}/plots"

mkdir -p "${PLOTS_DIR}"

for MODEL_NAME in "${NVSM_MODELS[@]}"; do
    print_subsection_title "${MODEL_NAME}"

    for EPOCH in $(seq 0 ${NUM_EPOCHS}); do
        MODEL_BIN="${NVSM_MODELS_DIR}/${MODEL_NAME}_${EPOCH}.hdf5"
        check_file "${MODEL_BIN}"

        if [[ -f "${PLOTS_DIR}/${MODEL_NAME}_${EPOCH}-${TOPICS_NAME}.pdf" ]]; then
            continue
        fi

        ${SCRIPT_DIR}/build/py/cuNVSMVisualize \
            --object_classification "${TOPICS_PATH}" \
            --mode tsne --edges --border --l2_normalize \
            --filter_unclassified \
            --plot_out "${PLOTS_DIR}/${MODEL_NAME}_${EPOCH}.pdf" \
            "${MODEL_BIN}" "${INDEX_DIR}"
    done
done

print_section_title "Generating animations."

if command -v convert >/dev/null 2>&1; then
    for MODEL_NAME in "${NVSM_MODELS[@]}"; do
        GIF_OUT="${SCRATCH_DIR}/${MODEL_NAME}.gif"

        if [[ -f "${GIF_OUT}" ]]; then
            continue;
        fi

        print_subsection_title "${MODEL_NAME}"

        PDFS=()

        for EPOCH in $(seq 0 ${NUM_EPOCHS}); do
            PDFS+=( "${PLOTS_DIR}/${MODEL_NAME}_${EPOCH}-${TOPICS_NAME}.pdf" )
        done

        convert -delay 150 -loop 0 ${PDFS[@]} "${GIF_OUT}"
    done
else
    1>&2 echo "Please install ImageMagick to generate animations."
fi
