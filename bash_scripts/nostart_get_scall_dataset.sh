#!/bin/bash

# Gets selected feature vector for all applications in the given file
# Run as:
#        ./get_scall_dataset.sh "SYSCALL_LIST_FILE" "FEATURE_VECTOR_PARAMS" OUTPUT_FOLDER
#        ./get_scall_dataset.sh all_syscalls.txt "--com 5" com5

APP_LIST_PATH="$HOME/MEGA/dataset/app_list.txt"
PYTHON_FEATURE_VECTOR_PATH="$HOME/PycharmProjects/ids_for_android/get_feature_vector.py"
SYSCALL_LOGS_PATH="$HOME/MEGA/dataset/benign_traces_filtered/nostart"
OUTPUT_PATH="$HOME/MEGA/dataset/benign_feature_vectors/nostart/$3"
#SYSCALL_LIST_PATH="$HOME/MEGA/dataset/all_syscalls.txt"
SYSCALL_LIST_PATH="$HOME/MEGA/dataset/$1"
FEATURE_VECTOR_PARAMS="$2" # e.g. "--com 5"

mkdir -p "$OUTPUT_PATH"

apps=($(grep . $APP_LIST_PATH))
for app in "${apps[@]}"
do
    log_files=($(find "$SYSCALL_LOGS_PATH" -maxdepth 1 -name "*$app*" | sort -t '\0' -n))
    if [ $? -ne 0 ]
    then
        printf "%s\n" "Error: Cannot find logs for $app"
        continue
    fi

    output_file_path="$OUTPUT_PATH/$app.log"
    rm -f "$output_file_path"

    for log_file in "${log_files[@]}"
    do

        python3.5 "$PYTHON_FEATURE_VECTOR_PATH" "$log_file" --syscalls "$SYSCALL_LIST_PATH" --csv $FEATURE_VECTOR_PARAMS --normalise>>"$output_file_path"
        if [ $? -ne 0 ]
        then
            printf "%s\n" "Error: Cannot get feature vector for log $log_file"
            continue
        fi
    done
done
