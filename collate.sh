rm -rf submit/
mkdir -p submit

prepare () {
    if [[ $(git diff origin -- $1 | wc -c) -eq 0 ]]; then 
        echo "WARNING: $1 is unchanged according to git."
    fi
    cp $1 submit/
}

echo "Creating tarball..."
prepare src/convolutional_layer.c
prepare src/maxpool_layer.c
prepare trycifar.py
prepare homework1_colab.ipynb
prepare homework1_python.pdf

tar cvzf submit.tar.gz submit
rm -rf submit/
echo "Done. Please upload submit.tar.gz to Canvas."

