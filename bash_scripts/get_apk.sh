#!/bin/bash

apps=($(grep . $1))

for app in "${apps[@]}"
do
    package="$(adb shell pm path $app)"
    app_path="$(cut -d ':' -f 2 <<< $package)"
    echo "Copying $app_path"
    adb shell "su -c 'cp $app_path /sdcard/$app.apk'"

    echo "Pulling $app_path"
    adb pull "/sdcard/$app.apk" "$2"

    echo "Deleting temp"
    adb shell "rm /sdcard/$app.apk"

    #"$(adb pull "$app_path" $2)"
done
