import pickle

theFile = open("parsedSGML.txt", "r")
content = theFile.readlines()
theFile.close()

theData = list()
article = None
tagdata = None
for line in content:

	if line.startswith("start tag: "):
		line = line.strip("start tag: ")
		line = line.strip()
		if line.startswith("<reuters"):
			article = dict()
			theData.append(article)
		elif line.startswith("<") and article != None:
			tagdata = list()
			article[line.strip("<>")] = tagdata

	if line.startswith("end tag: "):
		line = line.strip("end tag: ")
		line = line.strip()
		if line.startswith("</reuters"):
			for tag in article:
				listdata = article[tag]
				strdata = ""
				for element in listdata:
					strdata += element
				article[tag] = strdata.replace("\\n", " ")
			article = None
		elif line.startswith("</") and article != None and tagdata != None:
			tagdata = None

	if line.startswith("data: "):
		line = line.strip("data: ")
		line = line.strip()
		if article != None and tagdata != None:
			tagdata.append(line.strip("'"))

pickle.dump(theData, open("reuters.pickle", "w"))
