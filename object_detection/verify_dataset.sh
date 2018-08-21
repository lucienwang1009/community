if md5sum -c dataset_weights.md5
then
  echo "Dataset and weights are correct"
else
  echo "Dataset and weights are wrong"
fi
