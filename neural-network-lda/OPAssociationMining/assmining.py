import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import op.CorpusReader as opd

op = opd.CorpusReader("corpus");

# Build the ngrams x Organizational Patterns matrix, constrained under tf-idf
data = op.getUniBigramDataUsingTFIDF(tfidfThreshold=0.3)
columns = data.pop(0)
df = pd.DataFrame(data=data, columns=columns)

#  Find the frequent patterns sets constrained by the min support paramenter
freq_patterns = apriori(df, min_support=0.02, use_colnames=True, verbose=1)

# Mine the association rules constrained by confidence metric
rules = association_rules(freq_patterns, metric="confidence", min_threshold=0.3)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',900)
pd.set_option('display.max_rows',rules.shape[0]+1)

# print the mined association rules
i=0

#  Statistics about the patterns combination sets
twopatternsset = 0
threepatternset=0
fourpatternset=0
for row in rules.itertuples():
    i=i+1
    anticidents = list(row.antecedents)
    consequents = list(row.consequents)
    size = (len(anticidents)+len(consequents))
    twopatternsset = twopatternsset +({True:1, False:0} [size==2])
    threepatternset = threepatternset +({True:1, False:0} [size==3])
    fourpatternset = fourpatternset +({True:1, False:0} [size==4])

    print(i,'\t(',anticidents,',',consequents,')\t',row.support,'\t',row.confidence)

print('Size of two patterns sets: ',twopatternsset,'\t','Size of three patterns sets: ',threepatternset,'\t','Size of four patterns sets: ','\t',fourpatternset)



# filtering the considered and discarded rules and plotting the results

# rules= rules.loc[((rules['confidence']>=0.4) & (rules['support']>=0.015))]
# rules= rules.loc[((rules['confidence']>=0.2)& (rules['support']>=0.015))]
#
# considered= rules.loc[((rules['confidence']>=0.3) & (rules['support']>=0.02))]
# discarded= rules.loc[((rules['confidence']<0.3) | (rules['support']<0.02))]
# print(len(rules),len(considered),len(discarded))


# classes =['considered','discarded']
# colors =['tab:blue','tab:orange']
#
# pltconsidered = plt.scatter(considered['support'],considered['confidence'],marker='o',alpha=0.5,edgecolors='none', color=colors[0])
# pltdiscarded = plt.scatter(discarded['support'],discarded['confidence'],marker='o',alpha=0.5,edgecolors='none', color=colors[1])
#
# plt.legend((pltconsidered,pltdiscarded),('Considered','Discarded'),scatterpoints=1, loc='upper right', ncol=1,fontsize=8)
#
# plt.xlabel('support')
# plt.ylabel('confidence')
# plt.show()

# # plt.scatter(df['support'], df['confidence'], alpha=0.5)
# plt.xlabel('support')
# plt.ylabel('confidence')
# plt.title('Scattered Plot for '+str(len(df))+' Rules')
# plt.show()





#
# commonwords,statistics\
#     = op.getWordsStatisticsFor('ArchitectControlsProduct','ArchitectAlsoImplements','DeveloperControlsProcess')
# print(statistics)
# for f,v in commonwords.items():
#     if type(f) is tuple:
#          print( f[0]+' '+f[1]  ,',',v[0],',',v[1],',',v[2])
#     else:
#         print( f ,',',v[0],',',v[1],',',v[2])


# result = op.getStateistics(['ArchitectControlsProduct','ArchitectAlsoImplements','DeveloperControlsProcess'],0.2)
#
# for f,v in result.items():
#      if type(f) is tuple:
#          print( f[0]+' '+f[1]  ,',',v[0],',',v[1],',',v[2])
#      else:
#         print( f ,',',v[0],',',v[1],',',v[2])

# result = op.getNgramsExistence(['ArchitectControlsProduct','ArchitectAlsoImplements','DeveloperControlsProcess'],0.2)
#
# for f,v in result.items():
#      if type(f) is tuple:
#          print( f[0]+' '+f[1]  ,',',v[0],',',v[1],',',v[2])
#      else:
#         print( f ,',',v[0],',',v[1],',',v[2])

# result =  op.getNgramsExistenceStatistics(['ArchitectControlsProduct','ArchitectAlsoImplements','DeveloperControlsProcess'],0.2)
# print(result)
