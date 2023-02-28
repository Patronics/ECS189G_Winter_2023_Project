import os

os.chdir('../../data')
if not os.path.exists("./generated_stage_4_data"):
	os.makedirs("./generated_stage_4_data")

def scanDirectory(dirName, fileLimit = 0):
	print("---- begin scan of "+dirName+" ----")
	fileCount = 0
	wordDict = {}
	for file in os.listdir(os.getcwd()+"/stage_4_data/"+dirName):
		filename = os.fsdecode(file)
		#print(filename)
		fileCount+=1
		for line in open(os.getcwd()+"/stage_4_data/"+dirName+"/"+file):
			#print(line)
			for word in line.split():
				#print(word)
				if(word in wordDict):
					wordDict[word]=wordDict[word]+1
				else:
					wordDict[word]=1
		if (fileLimit and fileCount >= fileLimit):
			break
	#print(wordDict)
	print({k: v for k, v in sorted(wordDict.items(), key=lambda item: item[1])})
	print("---- end scan of "+dirName+", "+str(fileCount)+" files scanned ----")

scanDirectory("text_classification/train/pos")
#scanDirectory("text_classification/train/neg")
#scanDirectory("text_classification/test/pos")
#scanDirectory("text_classification/test/neg")
#scanDirectory("text_generation")