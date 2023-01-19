#stage 2 data
# all datasets and instructions from https://docs.google.com/document/d/1iOFZW1V-IWAu_EOi9-2sC90FM3IdVVQrrRDra_-gKCM/edit
#stage 2 wget derived from the link https://drive.google.com/file/d/1WG0NV7JTcDmpcEB-GRf-qecyWTisFYM-/view
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1WG0NV7JTcDmpcEB-GRf-qecyWTisFYM-' -O 'stage_2_data.zip'


unzip stage_2_data.zip
rm stage_2_data.zip 
rm -r __MACOSX
