#downloads stage 4 data
# all datasets and instructions from https://docs.google.com/document/d/1iOFZW1V-IWAu_EOi9-2sC90FM3IdVVQrrRDra_-gKCM/edit
#stage 4 wget derived from the link https://drive.google.com/file/d/1yNr8deq1JO-HNhlAa1yV4Zycjqmr1Hfz/view
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1yNr8deq1JO-HNhlAa1yV4Zycjqmr1Hfz&confirm=t' -O 'stage_4_data.zip'


unzip stage_4_data.zip
rm stage_4_data.zip 
rm -r __MACOSX
