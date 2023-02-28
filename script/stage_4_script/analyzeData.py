import os

os.chdir('../../data')
if not os.path.exists("./generated_stage_4_data"):
	os.makedirs("./generated_stage_4_data")

#scans through a directory's files counting each word's occurrence and associating it with a sentiment.
def scanDirectory(dirName, fileLimit = 0, sentiment = "unset", wordDict={}):
	print("---- begin scan of "+dirName+" with sentiment '"+sentiment+"'----")
	fileCount = 0
	for file in os.listdir(os.getcwd()+"/stage_4_data/"+dirName):
		filename = os.fsdecode(file)
		#print(filename)
		fileCount+=1
		for line in open(os.getcwd()+"/stage_4_data/"+dirName+"/"+file):
			#print(line)
			for word in line.split():
				#print(word)
				if(wordDict.get(word,{}).get(sentiment,{})):
					wordDict[word][sentiment] += 1
					wordDict[word]["count"] += 1
				else:
					if not word in wordDict:
						wordDict[word]={}
						wordDict[word]["count"] = 0
					wordDict[word][sentiment] = 1
					wordDict[word]["count"] += 1
		if (fileLimit and fileCount >= fileLimit):
			break
	#print(wordDict)
	#print({k: v for k, v in sorted(wordDict.items(), key=lambda item: item[1])})
	print("---- end scan of "+dirName+", "+str(fileCount)+" files scanned ----")
	return wordDict
	
#for the jokes file, slightly different format than the movie reviews
def scanFile(fileName, lineLimit = 0, wordDict={}):
	print("---- begin scan of "+fileName+" ----")
	lineCount = 0
	for line in open(os.getcwd()+"/stage_4_data/"+fileName):
		lineCount+=1
		#print(line)
		for word in line.split():
			#print(word)
			if(word in wordDict):
				#wordDict[word][sentiment] += 1
				wordDict[word]["count"] += 1
			else:
				if not word in wordDict:
					wordDict[word]={}
					wordDict[word]["count"] = 0
				#wordDict[word][sentiment] = 1
				wordDict[word]["count"] += 1
		if (lineLimit and lineCount >= lineLimit):
			break
	print("---- end scan of "+fileName+", "+str(lineCount)+" lines scanned ----")
	return wordDict

#print the sorted results in the form:
#indexNum, word, {count: totalCount, pos: posCount, neg: negCount}
#stops after words with fewer than threshold occurrences, or after wordLimit (if nonzero) total words have been printed
def printResults(wordDict, threshold=0, wordLimit=0, toFile=False, outFilePrefix=""):
	outFile=None
	if toFile:
		outFile=open(os.getcwd()+"/generated_stage_4_data/"+outFilePrefix+"thresh-"+str(threshold)+"_lim-"+str(wordLimit)+".txt", "w")
	index = 0
	for k, v in sorted(wordDict.items(), key=lambda item: item[1]["count"], reverse=True):
		if v["count"] < threshold:
			break
		if wordLimit and wordLimit<index:
			break
		print(str(index)+","+str(k)+","+str(v), file=outFile)
		index += 1
	if toFile:
		outFile.close()

wordDict = scanDirectory("text_classification/train/pos", 0, "pos", {})
wordDict = scanDirectory("text_classification/train/neg", 0, "neg", wordDict)

#just for getting a visual idea of how close the datasets are, don't actually use as ML input: 
#wordDict = scanDirectory("text_classification/test/pos", 0, "testPos", wordDict)
#wordDict = scanDirectory("text_classification/test/neg", 0, "testNeg", wordDict)
#scanDirectory("text_generation")


#for example, print all words occurring more than 100 times in the corpus:
	#printResults(wordDict, 100, 0)
#for example, print the 5,000 most common words:
	#printResults(wordDict, 0, 5000)
#for example, output all words found into a file:
	#printResults(wordDict, 0, 0, True)


printResults(wordDict, 10, 0, True)

jokeDict = scanFile("text_generation/data", 0, {})

printResults(jokeDict, 0, 0, True, "jokes_")




