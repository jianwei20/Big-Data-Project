import csv
import StringIO
//loading
input = sc.textFile("file://home/bigdatas16/workspace/Recommend/data/train100k")def loadRecord(line):
    """Parse a CSV line"""
    input = StringIO.StringIO(line)
    reader = csv.DictReader(input,fieldname=["name","favouriteAnimal"])
    return reader.next()
    input = sc.textFile(inputFile).map(loadRecord)
//saving

def writeRecords(record):
    """Write out CSV line"""
    output = StringIO.StringIO()
    write = csv.DictWriter(output,fieldname=["name","favouriteAnimal"])
    for record in records:
               writer.writerow(record)
     return [output.getvalue()]
     pandaLovers.mapPartitions(writeRecords).saveAsTextFile(outputFile)
~
