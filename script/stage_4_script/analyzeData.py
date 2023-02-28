import os

os.chdir('../../data')
if not os.path.exists("./generated_stage_4_data"):
	os.makedirs("./generated_stage_4_data")

#scans through a directory's files counting each word's occurrence and associating it with a sentiment.
def scanDirectory(dirName, fileLimit = 0, sentiment = "unset", wordDict={}):
	print("---- begin scan of "+dirName+" ----")
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

#print the sorted results in the form:
#indexNum, word, {count: totalCount, pos: posCount, neg: negCount}
#stops after words with fewer than threshold occurrences, or after wordLimit (if nonzero) total words have been printed
def printResults(wordDict, threshold=0, wordLimit=0):
	index = 0
	for k, v in sorted(wordDict.items(), key=lambda item: item[1]["count"], reverse=True):
		if v["count"] < threshold:
			break
		if wordLimit and wordLimit<index:
			break
		print(str(index)+","+str(k)+","+str(v))
		index += 1

wordDict = scanDirectory("text_classification/train/pos", 0, "pos", {})
scanDirectory("text_classification/train/neg", 0, "neg", wordDict)
#scanDirectory("text_classification/test/pos")
#scanDirectory("text_classification/test/neg")
#scanDirectory("text_generation")

printResults(wordDict, 10, 0)




