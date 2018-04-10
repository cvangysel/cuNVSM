#!/usr/bin/env bash

set -e

export MPLBACKEND="Agg"

#
# Boilerplate.
#
# Most of the functions below were copied from https://github.com/cvangysel/cvangysel-common.
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#
# Software management functions.
#

check_installed() {
    command -v $1 >/dev/null 2>&1 || { echo >&2 "Required tool '$1' is not installed. Please make sure that the executable '$1' is in your PATH. Aborting."; exit 1; }
}

package_root() {
    git rev-parse --show-toplevel
}

#
# Array-related functions.
#

# From http://stackoverflow.com/questions/3685970/check-if-an-array-contains-a-value.
#
# Usage: if [[ $(contains ${HAYSTACK[@]} ${NEEDLE}) == "y" ]]; then ...
function contains() {
    local n=$#
    local value=${!n}
    for ((i=1;i < $#;i++)) {
        if [ "${!i}" == "${value}" ]; then
            echo "y"
            return 0
        fi
    }
    echo "n"
    return 1
}

# Applies predictate $1 to array $2..
function apply_array() {
    PREDICATE=${1}
    shift

    for ARG in $@; do
        ${PREDICATE} $ARG
    done
}

#
# Value comparison.
#

check_eq() {
    check_not_empty "${1:-}" "first argument"
    check_not_empty "${2:-}" "second argument"

    if [[ "${1}" != "${2}" ]]; then
        1>&2 echo "Value '${1}' is not equal to '${2}'."

        exit -1
    fi
}

check_ne() {
    check_not_empty "${1:-}" "first argument"
    check_not_empty "${2:-}" "second argument"

    if [[ "${1}" == "${2}" ]]; then
        1>&2 echo "Value '${1}' is equal to '${2}'."

        exit -1
    fi
}

check_not_empty() {
    check_not_empty "${1:-}" "first argument"
    check_not_empty "${2:-}" "second argument"

    if [[ "${1}" == "${2}" ]]; then
        1>&2 echo "Value '${1}' is equal to '${2}'."

        exit -1
    fi
}

#
# Integral.
#

check_int() {
    if [[ ! "$1" =~ ^-?[0-9]+?$ ]] ; then
        1>&2 echo "String $1 is not a float."
        exit -1
    fi
}

int_compare() {
    if [[ "$1" < "$2" ]]; then
        1>&2 echo "$1 is smaller than $2."
        exit -1
    fi
}

check_pos_int() {
    check_int "$1"
    int_compare "$1" "0"
}

#
# Floating point.
#

check_float() {
    if [[ ! "$1" =~ ^-?[0-9]+([.][0-9]+)?(e-?[0-9]+)?$ ]] ; then
        1>&2 echo "String $1 is not a float."
        exit -1
    fi
}

float_compare() {
    awk -v n1=$1 -v n2=$2 'BEGIN{ if (n1<n2) print 0; print 1}'
}

max() {
    check_float "${1:-}"
    check_float "${2:-}"
    echo "${1:-}" | awk "{if (\$0 > ${2:-}) {print} else {print ${2:-}}}"
}

#
# String validation.
#

check_not_empty() {
    if [[ -z "$1" ]]; then
        1>&2 echo "Received empty string instead of $2."
        exit -1
    fi
}

check_valid_option() {
    if [[ $(contains "$@") != "y" ]]; then
        local n=$#
        local value=${!n}

        1>&2 printf "Option '%s' is not valid (allowed:" "${value}"
        for ((i=1;i < $#;i++)) {
            1>&2 printf " %s" "${!i}"
        }
        1>&2 printf ").\n"

        exit -1
    fi
}

#
# String manipulation
#

join_strings() {
    SEPARATOR="${1:-}"
    check_not_empty "${SEPARATOR}" "separator"

    shift

    if [[ "$#" -eq "0" ]]; then
        return
    fi

    echo -n "$1"
    shift

    for ELEMENT in $@; do
        echo -n "${SEPARATOR}"
        echo -n "${ELEMENT}"
    done

    echo
}

#
# File system.
#

check_multiple() {
    INVARIANT="${1:-}"
    shift

    for ARG in $@; do
        ${INVARIANT} ${ARG}
    done
}

check_file() {
    if [[ ! -f "$1" ]]; then
        1>&2 echo "File $1 does not exist."
        exit -1
    fi
}

check_file_not_exists() {
    if [[ -f "$1" ]]; then
        1>&2 echo "File $1 already exists."
        exit -1
    fi
}

check_directory() {
    if [[ ! -d "$1" ]]; then
        1>&2 echo "Directory $1 does not exist."
        exit -1
    fi
}

check_directory_not_exists() {
    if [[ -d "$1" ]]; then
        1>&2 echo "Directory $1 already exists."
        exit -1
    fi
}

directory_md5sum() {
    DIRECTORY="${1:-}"
    check_not_empty "${DIRECTORY}" "m5sum directory"
    check_directory "${DIRECTORY}"

    CURRENT_DIR=$(pwd)

    cd "${DIRECTORY}" && find . -type f -exec md5sum {} \; \
        | awk '{print $2 " " $1}' \
        | sort -k 1

    cd "${CURRENT_DIR}"
}

#
# Time.
#

timestamp() {
    date +%s
}

check_installed "IndriBuildIndex"
check_installed "curl"
check_installed "tar"
check_installed "trec_eval"
check_installed "PyndriQuery"
check_installed "PyndriStatistics"

#
# Constants.
#

NVSM_MODELS=( "LSE" "NVSM" )

declare -A NVSM_TRAIN_ARGS
NVSM_TRAIN_ARGS[NVSM]="--batch_size 51200 --nonlinearity hard_tanh --batch_normalization"
NVSM_TRAIN_ARGS[LSE]="--batch_size 4096 --nonlinearity tanh --bias_negative_samples"

declare -A NVSM_QUERY_ARGS
NVSM_QUERY_ARGS[NVSM]="--linear"
NVSM_QUERY_ARGS[LSE]=""

#
# Helper functions.
#

function compute_retrieval_effectiveness() {
    RUN_PATH="${1:-}"
    check_not_empty "${RUN_PATH}"
    check_file "${RUN_PATH}"

    trec_eval -m all_trec "${QREL_PATH}" "${RUN_PATH}" | awk '/^map_cut_1000\s+all/{print $3}'
}

function print_section_title() {
    TITLE="${1:-}"
    check_not_empty "${TITLE}" "TITLE"

    1>&2 echo
    for IDX in $(seq 1 $(( ${#TITLE} + 4 )) ); do
        1>&2 echo -n "#"
    done
    1>&2 echo

    1>&2 echo "# ${TITLE} #"

    for IDX in $(seq 1 $(( ${#TITLE} + 4 )) ); do
        1>&2 echo -n "#"
    done
    1>&2 echo
    1>&2 echo
}

function print_subsection_title() {
    TITLE="${1:-}"
    check_not_empty "${TITLE}" "TITLE"

    1>&2 echo "${TITLE}"

    for IDX in $(seq 1 $(( ${#TITLE} )) ); do
        1>&2 echo -n "-"
    done
    1>&2 echo
}

function print_columns() {
    FIRST_COLUMN="${1:-}"
    SECOND_COLUMN="${2:-}"

    1>&2 echo -n "${FIRST_COLUMN}"

    for IDX in $(seq 1 $(( 80 - ${#FIRST_COLUMN} )) ); do
        1>&2 echo -n " "
    done

    1>&2 echo -n "${SECOND_COLUMN}"
    1>&2 echo
}

function build_index() {
    INDEX_PARAMS="${1:-}"
    check_not_empty "${INDEX_PARAMS}" "INDEX_PARAMS"

    STOPWORD_LIST="${2:-}"
    check_not_empty "${STOPWORD_LIST}"

    INDEX_DIR="${3:-}"
    check_not_empty "${INDEX_DIR}" "INDEX_DIR"
    check_directory_not_exists "${INDEX_DIR}" "INDEX_DIR"

    TRECTEXT_PATH="${4:-}"
    check_not_empty "${INDEX_DIR}" "INDEX_DIR"
    check_file "${TRECTEXT_PATH}" "TRECTEXT_PATH"

    if [[ ! -f "${STOPWORD_LIST}" ]]; then
        1>&2 echo "Downloading Indri stoplist.dft."

        curl -s http://www.lemurproject.org/stopwords/stoplist.dft \
            > "${STOPWORD_LIST}"
    fi

    if [[ ! -f "${INDEX_PARAMS}" ]]; then
        cat > "${INDEX_PARAMS}" <<EOF
<parameters>
<memory>1024MB</memory>
<storeDocs>false</storeDocs>
<index>${INDEX_DIR}</index>
<corpus>
    <path>${TRECTEXT_PATH}</path>
    <class>trectext</class>
</corpus>"
</parameters>
EOF
    fi

    IndriBuildIndex "${INDEX_PARAMS}" "${STOPWORD_LIST}"
}

function train_nvsm() {
    NVSM_MODEL_="${1:-}"
    check_valid_option ${NVSM_MODELS[@]} "${NVSM_MODEL_}"

    OUTPUT_PREFIX_="${2:-}"
    check_not_empty "${OUTPUT_PREFIX_}"

    INDEX_DIR_="${3:-}"
    check_not_empty "${INDEX_DIR_}" "INDEX_DIR"
    check_directory "${INDEX_DIR_}"

    GLOG_logtostderr=1 \
    ${SCRIPT_DIR}/build/cpp/cuNVSMTrainModel \
        --num_epochs "${NUM_EPOCHS}" \
        --max_vocabulary_size 65536 \
        --min_document_frequency 0 \
        --regularization_lambda 1e-2 \
        --learning_rate 1e-3 \
        --window_size 10 \
        --word_repr_size 300 \
        --entity_repr_size 256 \
        --num_random_entities 10 \
        ${NVSM_TRAIN_ARGS[${NVSM_MODEL_}]} \
        --weighting uniform \
        --seed 1 \
        --update_method full_adam \
        --compute_initial_cost \
        --dump_initial_model \
        --dump_every 1000000 \
        --output "${OUTPUT_PREFIX_}" \
        "${INDEX_DIR_}"
}
