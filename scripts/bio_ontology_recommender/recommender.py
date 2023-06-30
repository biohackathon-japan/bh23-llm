from os import getenv
from sys import exit
from json import dumps, loads
from time import time
from pprint import PrettyPrinter
from hashlib import sha256
from pathlib import Path
from requests import get

pp = PrettyPrinter()

start = time()

text_to_search = "A migraine is usually a moderate or severe headache felt as a throbbing pain on one side of the head. Many people also have symptoms like nausea, vomiting and increased sensitivity to light or sound."
api_url = f"https://data.bioontology.org/recommender?input={text_to_search}"
api_key = getenv("BIO_ONTOLOGY_API_KEY", "secret")
filepath = f"{sha256(text_to_search.encode()).hexdigest()}.json"

request_start = time()
data = None
if Path(filepath).is_file():
    # if we already have a response with an identical search text
    # don't request again, but read the response from file
    with open(filepath, "rb") as f:
        data = loads(f.read())
else:
    # we have never made this request before,so we need to request it,
    # the response will be saved to file for re-runs
    response = get(api_url, headers={"Authorization": f"apikey token={api_key}"})
    if response.status_code == 200:
        data = response.json()
        with open(filepath, "wb") as f:
            f.write(dumps(data, indent=4).encode())
    else:
        exit(f"request to {api_url} failed with {response.status_code}, {response.text}")
request_end = time()

# list on ontologies with hits
# ontologies = []
# for ontology in data:
#     ontologies.append(ontology["ontologies"][0]["acronym"])
# print(f"ontologies: {ontologies}")

# filter for narrowing down results
# only listed ontologies are taken
desired_ontologies = (
    "http://www.orpha.net/ORDO",
    "http://purl.obolibrary.org/obo/MONDO",
    # "http://id.nlm.nih.gov/mesh",
    # "http://sbmi.uth.tmc.edu/ontology/ochv",
    "http://purl.obolibrary.org/obo/HP",
    "http://purl.obolibrary.org/obo/OGG",
)

parse_start = time()
# parse the JSON response for ontology codes and term names
# hits are a list of tuples [(name, code)]
hits = []
for result in data:
    for annotation in result["coverageResult"]["annotations"]:
        if annotation["annotatedClass"]["@id"].startswith(desired_ontologies):
            hits.append((annotation["text"], annotation["annotatedClass"]["@id"]))

# remove duplicates, and sort them by term name
# for displaying them in table format
hits = set(hits)
hits = sorted(hits)

print(f"hits: {len(hits)}\n")

print("raw data\n")
pp.pprint(hits)

def tabs(n: int) -> str:
    """Helper function to print table with correct amount of tabs."""
    return (3-round(n/6)) * "\t"

print("\ntable\n")
print("Term ID\t\tTerm Label")
for hit in hits:
    # do a little parsing to transform the URI ontology codes to ONTO:CODE format
    # e.g. https://ontology.org/HP_123 -> HP:123
    ontology_code = hit[1].split('/')[-1].replace('_', ':').replace('Orphanet', 'ORPHA')
    print(f"{ontology_code}{tabs(len(ontology_code))}{hit[0]}")
parse_end = time()

print(f"\nrequest runtime: {round(request_end - request_start, 5)} s")
print(f"parse runtime: {round(parse_end - parse_start, 5)} s")
print(f"total runtime: {round(time() - start, 5)} s")
