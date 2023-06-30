# BioOntology Recommender API
BioPortal provides a [Recommender API](http://data.bioontology.org/documentation#nav_recommender) with a keyword search feature for finding relevant ontology codes from text input.

An API key can be generated by registering at [BioPortal](https://bioportal.bioontology.org/login).

Example execution of [recommender.py](recommender.py) using the prompt `A migraine is usually a moderate or severe headache felt as a throbbing pain on one side of the head. Many people also have symptoms like nausea, vomiting and increased sensitivity to light or sound.`.
``` shell
$ python parse.py 
hits: 8

raw data

[('HEADACHE', 'http://purl.obolibrary.org/obo/HP_0002315'),
 ('MIGRAINE', 'http://purl.obolibrary.org/obo/HP_0002076'),
 ('MODERATE', 'http://purl.obolibrary.org/obo/HP_0012826'),
 ('NAUSEA', 'http://purl.obolibrary.org/obo/HP_0002018'),
 ('PAIN', 'http://purl.obolibrary.org/obo/HP_0012531'),
 ('SENSITIVITY', 'http://purl.obolibrary.org/obo/MONDO_0000605'),
 ('SEVERE', 'http://purl.obolibrary.org/obo/HP_0012828'),
 ('VOMITING', 'http://purl.obolibrary.org/obo/HP_0002013')]

table

Term ID		Term Label
HP:0002315	HEADACHE
HP:0002076	MIGRAINE
HP:0012826	MODERATE
HP:0002018	NAUSEA
HP:0012531	PAIN
MONDO:0000605	SENSITIVITY
HP:0012828	SEVERE
HP:0002013	VOMITING

request runtime: 10.23201 s
parse runtime: 0.00039 s
total runtime: 10.23245 s
```