import os

os.chdir('../../data')


inFileName = "thresh-10_lim-0.txt"


def wordIndexes(inDataFile):
	wordList = {}
	with open(os.getcwd()+"/generated_stage_4_data/"+inDataFile) as sourceDataFile:
		for line in sourceDataFile:
			[index, word, _] = line.split(",",2)
			#for index, word in line.split(","):
			#print (str(index), word)
			wordList[word]=index
	#print (wordList)
	return wordList


#Convert all words in directory to array of indexes
def processDirectory(dirName, fileLimit = 0, wordList={}, splitByFile=False, outFileSentiment=None):
	print("---- begin scan of "+dirName+" ----")
	fileCount = 0
	outStr = ""
	outFile = None
	if (outFileSentiment):
		outFile=open(os.getcwd()+"/generated_stage_4_data/indexed_"+outFileSentiment+"_"+inFileName, "w")
	
	for file in os.listdir(os.getcwd()+"/stage_4_data/"+dirName):
		filename = os.fsdecode(file)
		#print(filename)
		fileCount+=1
		for line in open(os.getcwd()+"/stage_4_data/"+dirName+"/"+file):
			#print(line)
			for word in line.split():
				#print(word)
				if(word in wordList):
					outStr+= "," + wordList[word]
				else:
					#remove to entirely exclude unrecognized words
					#outStr+= ",-1"
					pass
		if (fileLimit and fileCount >= fileLimit):
			break
		if splitByFile:
			outStr+="\n"
	print(outStr, file=outFile)
	#print({k: v for k, v in sorted(wordDict.items(), key=lambda item: item[1])})
	print("---- end scan of "+dirName+", "+str(fileCount)+" files scanned ----")
	if outFileSentiment:
		outFile.close()
	return outStr

#adjust input file name based on threshold value to match desired output file from analyzeData.py
wordList = wordIndexes(inFileName)


processDirectory("text_classification/train/pos", 0, wordList, True, "pos")
processDirectory("text_classification/train/neg", 0, wordList, True, "neg")

