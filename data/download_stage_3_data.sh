#stage 3 data
# all datasets and instructions from https://docs.google.com/document/d/1iOFZW1V-IWAu_EOi9-2sC90FM3IdVVQrrRDra_-gKCM/edit
#stage 3 wget derived from the link https://drive.google.com/file/d/15yb5-X1ck2JJ269fVIB3CVUozGa2nTik/view
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=15yb5-X1ck2JJ269fVIB3CVUozGa2nTik&confirm=t' -O 'stage_3_data.zip'


unzip stage_3_data.zip
rm stage_3_data.zip 
rm -r __MACOSX
