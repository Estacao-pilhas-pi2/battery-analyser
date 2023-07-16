folder=$1;


for img in $(ls $folder); do
    python3 crop.py $folder/$img;
done
