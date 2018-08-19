
#
# Milne & witten similarity : http://gsi-upm.github.io/sematch/similarity/

from sematch.semantic.similarity import EntitySimilarity
entity_sim = EntitySimilarity()

x = entity_sim.relatedness('http://dbpedia.org/resource/Madrid','http://dbpedia.org/resource/Barcelona')
print (x)