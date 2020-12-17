import re

f = open("test.log", "r")
loads = {}
stores = {}
loops = []
for line in f:
    if line[0] == '@':
        loops = []
    elif line[0:4] == "Loop":
        loops.append(re.search("Loop2 \((\d+), (\d+)\)", line).groups()[0])
    elif line[0:4] == "Load":
        var, index = re.search("Load \(([A-Za-z]+), (\d+)\)", line).groups()
        for loop in loops:
            if loop in loads:
                if var in loads[loop]:
                    loads[loop][var].append(index)
                else:
                    loads[loop] = {var: [index]}
            else:
                loads[loop] = {var: [index]}
    elif line[0:5] == "Store":
        var, index = re.search("Store \(([A-Za-z]+), (\d+)\)", line).groups()
        for loop in loops:
            if loop in stores:
                if var in stores[loop]:
                    stores[loop][var].append(index)
                else:
                    stores[loop] = {var: [index]}
            else:
                stores[loop] = {var: [index]}

f.close()
f = open("test2.log", "w")

f.write("@0\n")

for loop, load in loads.items():
    for var, indices in load.items():
        out = "LoadPattern (" + loop + ", " + var + ", "
        for index in indices:
            out += "_" + index
        out += ")\n"
        f.write(out)

for loop, store in stores.items():
    for var, indices in store.items():
        out = "StorePattern (" + loop + ", " + var + ", "
        for index in indices:
            out += "_" + index
        out += ")\n"
        f.write(out)

f.close()
