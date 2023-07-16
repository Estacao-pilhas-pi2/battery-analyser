


FOLDER="$1";

for img in $(ls $folder);
    do python crop.py $folder/$img; done
