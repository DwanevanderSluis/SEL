import operator

t_dict = {'a':{'b':3},'c':{'b':2}}


#x = sorted(t_dict.items(), key=operator.itemgetter(1), reverse=True)

res = []
for docid in t_dict.keys():
    res.append(t_dict[docid]['b'])


print(res)
