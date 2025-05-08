conda env export --no-builds > environment.yml
conda env update --file environment.yml --prune