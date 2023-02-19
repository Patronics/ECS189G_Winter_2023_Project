# ECS189G_Winter_2023_Project
For ECS189G at UC Davis, A Deep Learning team project


### setup 
- Either use pycharm to create the python venv or manually create it.
- add a line of the form `export PYTHONPATH='/absolute/path/to/ECS189G_Winter_2023_Project'` to the `./venv/bin/activate` script
- optionally also add lines to reset it to default upon exiting the venv as described [here](https://stackoverflow.com/a/4758351/4268196)

### usage
- run `source venv/bin/activate`
- from the `script` directory of the applicable stage, run `../../venv/bin/python3 scriptname.py`
- alternatively use the big friendly play button in pycharm.

for ease of use while unattended (on MacOS at least), run the script with a few `say` commands, such as
```bash
python3 ./script/stage_3_script/script_CNN.py && say "processing complete" || say "processing failed"
```
