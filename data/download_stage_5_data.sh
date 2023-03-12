#downloads stage 5 data
# all datasets and instructions from https://docs.google.com/document/d/1iOFZW1V-IWAu_EOi9-2sC90FM3IdVVQrrRDra_-gKCM/edit
#stage 5 wget derived from the link https://drive.google.com/file/d/1L_MolRhZczh8eECQuTzABMXlVkyxl-yJ/view?usp=sharing
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1L_MolRhZczh8eECQuTzABMXlVkyxl-yJ&confirm=t' -O 'stage_5_data.zip'


unzip stage_5_data.zip
rm stage_5_data.zip 
rm -r __MACOSX
