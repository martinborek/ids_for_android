#!/bin/bash

# parse zygote process(es) PID 
# start tracing zygote(s)
#   -> maybe option not to include calls connected to tracing?
# start application (+remember PID)
# sleep for x seconds
# wake up, stop tracing
# copy file with process PID to the PC
# remove all created logs
# ? stop process

runtime=100 # 100 seconds

notify_end=10 # Notify 10 seconds before the end of tracking

# Runtime cannot be less than notification as the notification time is subtracted from the runtime.
if [ $runtime -le $notify_end ]
then

    printf "%s\n" "Error: Runtime cannot be less than notification time."
    exit 1
fi

temp_phone_folder="/sdcard/traces"
temp_phone_file="trace.log"
strace_path="/data/local/tmp/strace_aarch"
#strace_path="/data/local/tmp/strace_aarch_static"

output_folder=$1
output_prefix=$2
app=$3
output_path="$output_folder/$output_prefix"_"$app.log"

mkdir -p "$output_folder"

adb shell mkdir -p "$temp_phone_folder"
if [ $? -ne 0 ]
then
    printf "%s\n" "Error: Cannot create temporary folder on the phone. Might be a problem with ADB connection." >&2 
    exit 1
fi
    
adb shell "rm -f $temp_phone_folder/*" # Empty the temporary folder

zygote_pids=($(adb shell "ps | grep zygote" | tr -s ' ' | cut -d ' ' -f 2))
if [ $? -ne 0 ] || [ ${#zygote_pids[@]} -le 0 ]
then
    printf "%s\n" "Error: Cannot obtain zygote PID." >&2 
    exit 1
fi

#adb shell "su -c '$strace_path -p 4485 -ff -o $temp_phone_folder/$temp_phone_file'" &

for zygote in "${zygote_pids[@]}"
do
    adb shell "su -c '$strace_path -p $zygote -q -ff -o $temp_phone_folder/$temp_phone_file'" &
#    adb shell "su -c '$strace_path -p $zygote -ff -o $temp_phone_folder/$temp_phone_file &'"
done

sleep 5 # To give time to all strace processes to start

strace_pids=($(adb shell "ps | grep $strace_path" | tr -s ' ' | cut -d ' ' -f 2))
echo "Running strace processes are ${strace_pids[@]}"
# Check how many strace processes are running
if [ $? -ne 0 ] || [ ${#strace_pids[@]} -ne ${#zygote_pids[@]} ]
then

    printf "%s\n" "Error: Cannot run strace for all zygote processes." >&2 

    # Close all started strace processes
    adb shell "su -c 'kill -9 ${strace_pids[@]}'"

    exit 1
fi


adb shell "monkey -p $app 1"

if [ $? -ne 0 ]
then
    printf "%s\n" "Error: Cannot start the application $app." >&2 

    # Stop tracing
    adb shell "su -c 'kill -9 ${strace_pids[@]}'"
    exit 1
fi

sleep $(($runtime - $notify_end))
echo "Tracing ends in $notify_end seconds."
sleep "$notify_end"

# Stop tracing
adb shell "su -c 'kill -9 ${strace_pids[@]}'"

# Get process ID to be able to identify the correct log
app_pid=($(adb shell "ps | grep $app" | tr -s ' ' | cut -d ' ' -f 2))
if [ ${#app_pid[@]} -ne 1 ]
then
    if [ ${#app_pid[@]} -eq 0 ]
    then
        printf "%s\n" "Error: Application $app has already stopped. Logs not pulled." >&2 
    else
        printf "%s\n" "Error: Application $app has more running processes: ${app_pid[@]}." >&2 

    fi

    # Stop all processes of the application
    adb shell "su -c 'kill -9 ${background_app_pid[@]}'"
    
    exit 1
fi

adb shell "am force-stop $app"
if [ $? -ne 0 ]
then
    printf "%s\n" "Error: Cannot stop the application $app." >&2 
    exit 1
fi

log_path="$temp_phone_folder/$temp_phone_file.${app_pid[0]}"

adb pull "$log_path" "$output_path"
if [ $? -ne 0 ]
then
    printf "%s\n" "Error: Cannot pull the log $log_path." >&2 
    exit 1
fi

adb shell "rm -r $temp_phone_folder"


sleep 5 # To give time to all processes to finish

# Get app process ID again in case there was a background process to stop. If
# not stopped before the next tracing of the same app, logs would not be recorded.
background_app_pid=($(adb shell "ps | grep $app" | tr -s ' ' | cut -d ' ' -f 2))
if [ ${#background_app_pid[@]} -ge 1 ]
then
    echo "Killing process ${background_app_pid[@]}"

    adb shell "su -c 'kill -9 ${background_app_pid[@]}'"
fi

echo "Tracking successful"
exit 0
