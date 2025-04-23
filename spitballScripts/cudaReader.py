import pandas as pd
import matplotlib.pyplot as plt
import time


def file_to_dict_array(file_path, delimiter=':'):
    """
    Reads a file and returns a list of dictionaries. Each line in the file is split
    into key-value pairs based on the delimiter, and each line becomes a dictionary
    in the list.

    Args:
        file_path (str): The path to the file.
        delimiter (str, optional): The delimiter separating keys and values. Defaults to ':'.

    Returns:
        list: A list of dictionaries, or an empty list if the file is empty or an error occurs.
    """
    dict = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line:  # Avoid processing empty lines
                    notUsed, extra = line.split(delimiter, 1)
                    key, extra = extra.split("\t", 1)
                    key = key.strip()
                    key2, extra = extra.split(delimiter, 1)
                    key2 = key2.strip()
                    value2, extra = extra.split("\t", 1)
                    value2 = int(value2.strip())
                    key3, extra = extra.split(delimiter, 1)
                    key3 = key3.strip()
                    value3, extra = extra.split("\t", 1)
                    value3 = int(value3.strip())
                    key4, value4 = extra.split(delimiter, 1)
                    key4 = key4.strip()
                    value4 = int(value4.strip())
                    if key in dict :
                        dict[key][key2] += value2
                        dict[key][key3] += value3
                        dict[key][key4] += value4
                    else :
                        dict[key] = {key2: value2, key3 : value3, key4: value4}
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except ValueError:
         print(f"Error: Incorrect format in line: {line}. Ensure lines contain a '{delimiter}'.")
         
    return dict

# Example usage
mainDict = []
for i in range(1, 1002, 100):
    dataName = 'tmp/outData_'+ str(i)
    file_path =  dataName + '.txt'
    mainDict.append({ "n" : i, 'data' : file_to_dict_array(file_path)})

mainDictG = []
for i in range(1, 1002, 100):
    dataName = 'tmp/outDataG_'+ str(i)
    file_path =  dataName + '.txt'
    mainDictG.append({ "n" : i, 'data' : file_to_dict_array(file_path)})

# dataName = 'outData'
# file_path =  dataName + '.txt'
# mainDict.append({ "n" : 100, 'data' : file_to_dict_array(file_path)})
# dataName = 'outData'
# file_path =  dataName + '.txt'
# mainDict.append({ "n" : 10, 'data' : file_to_dict_array(file_path)})

testName = 'NumberIterationsTestCupti'
        
runTimestamp = str(int(time.time()))
df = pd.DataFrame(mainDict)
dfG = pd.DataFrame(mainDictG)
plotData = []

for data in df['data']:
    tmpDict = {}
    for element in data:
        for metric in data[element]:
            # print(data)
            # print(element)
            # print(metric)
            tmpDict[element+metric] = data[element][metric]
    # print(tmpDict)
    plotData.append(tmpDict)
dataDf = pd.DataFrame(plotData)

plotData = []
for data in dfG['data']:
    tmpDict = {}
    for element in data:
        for metric in data[element]:
            # print(data)
            # print(element)
            # print(metric)
            tmpDict[element+metric] = data[element][metric]
    # print(tmpDict)
    plotData.append(tmpDict)
dataDfG = pd.DataFrame(plotData)


fig, axes = plt.subplots(nrows=4, ncols=2, figsize = (15,25))

axes[3,0].plot(df['n'], dataDf['MEMSETTotalDuration'], label="MEMSET TotalDuration")
axes[3,0].set_xlabel('Number Iterations')
axes[3,0].set_ylabel('Execution Time (Nanoseconds)')

axes[1,0].plot(df['n'], dataDf['MEMCPYTotalDuration'], label="MEMCPY TotalDuration")
axes[1,0].set_xlabel('Number Iterations')
axes[1,0].set_ylabel('Execution Time (Nanoseconds)')

axes[2,0].plot(df['n'], dataDf["MEMCPYBytes"], label="MEMCPY Bytes")
axes[2,0].set_xlabel('Number Iterations')
axes[2,0].set_ylabel('Bytes Copied (Bytes)')

axes[0,0].plot(df['n'], dataDf["RUNTIMETotalDuration"], label="RUNTIME TotalDuration")
axes[0,0].set_xlabel('Number Iterations')
axes[0,0].set_ylabel('Execution Time (Nanoseconds)')


axes[3,1].set_visible(False) 
axes[1,1].plot(dfG['n'], dataDfG['MEMCPYTotalDuration'], label="MEMCPY TotalDuration")
axes[1,1].set_xlabel('Number Iterations')
axes[1,1].set_ylabel('Execution Time (Nanoseconds)')

axes[2,1].plot(dfG['n'], dataDfG["MEMCPYBytes"], label="MEMCPY Bytes")
axes[2,1].set_xlabel('Number Iterations')
axes[2,1].set_ylabel('Bytes Copied (Bytes)')

axes[0,1].plot(dfG['n'], dataDfG["RUNTIMETotalDuration"], label="RUNTIME TotalDuration")
axes[0,1].set_xlabel('Number Iterations')
axes[0,1].set_ylabel('Execution Time (Nanoseconds)')

axes[3,0].set_title("MEMSET TotalDuration (Nanoseconds) vs Number Iterations")
axes[1,0].set_title("MEMCPY TotalDuration (Nanoseconds) vs Number Iterations")
axes[2,0].set_title("MEMCPY Bytes Copied (Bytes) vs Number Iterations")
axes[0,0].set_title("RUNTIME TotalDuration (Nanoseconds) vs Number Iterations")

# axes[0,1].set_title("TotalDuration (Nanoseconds) vs Number Iterations Gauss")
axes[1,1].set_title("MEMCPY TotalDuration (Nanoseconds) vs Number Iterations Gauss")
axes[2,1].set_title("MEMCPY Bytes Copied (Bytes) vs Number Iterations Gauss")
axes[0,1].set_title("RUNTIME TotalDuration (Nanoseconds) vs Number Iterations Gauss")

fig.suptitle('Canny vs Gauss', fontsize=20)
# plt.legend()
# plt.xlabel('Number Iterations')
# plt.ylabel('Execution Time')
plotName = "plots/" + str(runTimestamp) + "_" + testName +".jpg"
fig.savefig(plotName)

# plt.yscale('log')
# plt.savefig("plots/" +str(runTimestamp) + "_" + testName + "_logScale.jpg")