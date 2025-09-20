---
title: 'BioHackJP 2023 Report R1: linked data applications of large language models (LLMs)'
title_short: 'BioHackJP 2023 LLM'
tags:
  - Linked Data
  - Large Language Models
  - Human Phenotype Ontology
  - Concept Recognition
authors:
  - name: Orion Buske
    orcid: 0000-0002-9064-092X
    affiliation: 1
  - name: Jerven Bolleman
    orcid: 0000-0002-7449-1266
    affiliation: 2
  - name: Teemu Kataja
    orcid: 0009-0001-1434-0415
    affiliation: 3
  - name: Takayuki Suzuki
    orcid: 0000-0003-4481-9109
    affiliation: 4
  - name: Tazro Ohta
    orcid: 0000-0003-3777-5945
    affiliation: 5
  - name: Hiroshi Mori
    orcid: 0000-0003-0806-7704
    affiliation: 6
  - name: Hirotaka Suetake
    orcid: 0000-0003-2765-0049
    affiliation: 7
  - name: Masaki Fukui
    orcid:
    affiliation: 7
  - name: Toyofumi Fujiwara
    orcid: 0000-0002-0170-9172
    affiliation: 8
  - name: Soichi Ogishima
    orcid: 0000-0001-8613-2562
    affiliation: 9
affiliations:
  - name: PhenoTips, Canada
    index: 1
  - name: SIB Swiss Institute of Bioinformatics
    index: 2
  - name: CSC - IT Center for Science Ltd.
    index: 3
  - name: Hiroshima University, Japan
    index: 4
  - name: Institute for Advanced Academic Research, Chiba University
    index: 5
  - name: National Institute of Genetics
    index: 6
  - name: Sator, Inc.
    index: 7
  - name: Database Center for Life Science, Japan
    index: 8
  - name: ToMMo, Tohoku University, Japan
    index: 9
date: 30 June 2023
cito-bibliography: paper.bib
event: BH23JP
biohackathon_name: "BioHackathon Japan 2023"
biohackathon_url:   "https://2023.biohackathon.org/"
biohackathon_location: "Kagawa, Japan, 2023"
group: R1
git_url: https://github.com/biohackathon-japan/bh23-llm
authors_short: Buske \emph{et al.}
---

![BioHackrXiv logo](./biohackrxiv.png)


# Background

The field of bioinformatics plays a crucial role in enabling researchers to extract meaningful insights from the vast amount of biological data generated today. With advancements in technology and the availability of large-scale datasets, it has become increasingly important to develop standardized approaches for representing and integrating biological information. Linked data, a method for publishing structured data on the web, has emerged as a promising solution for facilitating the integration and interoperability of diverse biological data sources.

The BioHackathon 2023, held in Japan, provided an ideal platform for researchers and bioinformatics enthusiasts to collaborate and explore innovative solutions to address the challenges in the field. Our project focused on the application of Linked Data and Large Language Models (LLMs) to standardize biological data and enhance its accessibility and usability.

LLMs, such as OpenAI's GPT-3.5 architecture, have demonstrated remarkable capabilities in understanding and generating human-like text. Leveraging the power of LLMs, we aimed to automate the process of extracting relevant biological terms from unstructured text and mapping them to existing ontology terms. Ontologies, which are hierarchical vocabularies of terms and their semantic relationships, provide a standardized framework for organizing and categorizing biological concepts.

# Outcomes

## Selecting models and establishing a working environment

To achieve our objectives, we conducted a comprehensive survey of available language models and evaluated their suitability for our task. We explored different models, taking into consideration factors such as performance, computational requirements, and ease of deployment. Subsequently, we sought to run the selected models on a local computer, ensuring that the infrastructure requirements were met.

Having established a working environment for LLMs, we developed a set of pipelines that incorporated various natural language processing techniques to extract relevant biological terms from textual data. These terms were then matched and mapped to the corresponding ontology terms, thereby providing a standardized representation of the extracted information. By utilizing Linked Data principles, we aimed to create an interconnected network of biological knowledge that would facilitate data integration and enable advanced analysis.


## Medical transcription from physician dictation <!-- (Teemu) -->

### Rationale
Writing patient records is a time consuming task, and many physicians resort to dictation, with transcription and data entry being done by other personnel at a later date. Speech recognition is a powerful tool that could be used to reduce this intermediate task of transcription, and the output could be fed to an LLM for inference, summary, and even data entry. We experimented with OpenAI's Whisper speech recognition model for writing automated patient records.

### Implementation
OpenAI provides Whisper in both a local model run on your CPU, and a remote API.[10] Both versions of Whisper were operated on with great success.

Audio was recorded using the free open source tool Audacity.[11] The speech sample was manually prepared, by cleaning background noise, and cutting the sample to size.

Scripts were prepared for sending audio files to the Whisper API, as well as downloaded the Whisper model for local execution. Both versions of Whisper performed with great performance and accuracy. The local version was tested using the `tiny.en`, `base.en`, and `small.en` models, with 39, 74, and 244 million parameters respectively.[14]

The remote API has more options for tweaking its use, but the local model also performed sufficiently with the options available. The most important options tweaked with were `temperature` to set the level of focus, and `initial_prompt` for giving the model a setting to reduce confusion between similar sounding words.

The exact settings used were:
- `temperature=0.2`
- `initial_prompt="medical terminology"`

One unexpected behaviour was observed with the remote API. The remote version of Whisper has the ability to detect the speaker's nationality from their accent, and it automatically translates the transcription to the predicted language, and not to the language the sample was given in. Fortunately there was an option to force the transcription language to overcome this issue.

### Future Work
For the future we plan to develop a script with a command line interface (CLI) to automatically record and transcribe speech, and sending it to an LLM inference. During the biohackathon, these steps were carried out manually.


## Phenotype extraction and summarization <!-- (Orion, Takayuki, Atsuko) -->

The current state-of-the-art algorithms for rare disease diagnosis and variant prioritization take as input the patient's clinical presentation coded as a list of Human Phenotype Ontology (HPO) terms. High quality terms are important for accurate diagnosis of rare diseases. HPO terms are curated as part of standard practice by either the clinician or the laboratory as part of the genomic sequencing process. This process can be time-consuming and labor-intensive, so there have been many efforts to help automate this process.

Deep-learning methods have become the state-of-the-art and outperform earlier dictionary and rule-based methods, such as Clinphen [Clinphen], though their performance still falls short of human curators. NeuralCR [NeuralCR] employs a convolutional neural network (CNN). PhenoTagger is a hybrid method using a fine-tuned BioBERT [BioBERT] model along with a dictionary-based method. PhenoBERT is also a hybrid method using a CNN along with a fine-tuned BioBERT model.

In the last several years, the capabilities and availability of generative pre-trained LLMs, such as LLaMA and GPT, have grown extremely rapidly. While BioBERT (2019) has just 110 million parameters, GPT 3 (2020) has 175 billion parameters and GPT 4 (2023) has 1.7 trillion parameters. Generative LLMs have demonstrated high profiency at a wide variety of tasks without any fine-tuning, so we set out to explore the use of generative LLMs at the task of HPO extraction.

### Exploration of existing models for HPO extraction

An immense variety of LLMs are now available, both closed and open-source. PhenoBERT and similar BERT-based models are currently the state-of-the-art for HPO extraction. ChatGPT and similar generative models are especially well-suited to abstractive summarization.

We started by exploring a variety of generative LLMs, including the following:

| Model        | Open vs Closed Source |
| ------------ | --------------------- |
| Bard         | Closed                |
| Bing         | Closed                |
| BioGPT       | Open                  |
| GPT-3.5      | Closed                |
| GPT-4        | Closed                |
| gpt-neox-20b | Open                  |
| MPT-7B-chat  | Open                  |
| PaLM         | Closed                |

We tested various prompts ranging from simple ("extract human phenotype ontology from the given text:") to more complex (Appendix A1).

### HPO term extraction

The first task we set out to explore was the direct extraction of HPO terms (labels, identifiers, and present/absent status) from text.

Examples of the output from PaLM and GPT-4 are included in Appendix A1.

We also explored the use of the BioOntology Recommender as a potential non-LLM baseline, and PhenoBERT as the current state-of-the-art model.

#### PhenoBERT re-evaluation on HPO extraction task <!-- (Takayuki) -->

PhenoBERT's performance in HPO (Human Phenotype Ontology) extraction has been previously evaluated, demonstrating F1 scores of 73% and 75% for micro and macro averages, respectively [7]. In our study, We experimented to re-evaluate the HPO extraction task in order to assess if the model reproduces similar outcomes. The installation and execution of PhenoBERT were successfully carried out on a GPU server equipped with CUDA version 12.1 and an NVIDIA GeForce RFX 3060 with 12GB of FB memory.

To assess the accuracy of HPO extraction, we extracted a list of HPO IDs from 28 clinical notes and systematically compared each list with the corresponding HPO IDs annotated. For each of the 28 cases, we calculated the percentage of matching HPO IDs between the two lists. The average of these percentages across all 28 cases was found to be 50.1%.

#### BioOntology Recommender <!-- (Teemu) -->

National Center for Biomedical Ontology (NCBO) provides a BioOntology Recommender API, which can be used to find ontology codes from free text search. The system is keyword based and doesn't use inference, so perfect spelling is required, as it can't differentiate between typos.[12]

The BioOntology API is not an LLM, but serves as a "ground truth" for sanity checking our data, as it doesn't have the ability to invent nonexistant terms, but queries a curated database of ontologies.

The Recommender API takes free text as input and tries to suggest relevant ontology codes as a response. The response comes in JSON format, and in order to conform to the same table format as for the LLM prompts, a small script was devised for parsing the data from JSON into the tabular table format. The suggested ontologies were extremely verbose, so a filter had to be implemented for discarding irrelevant results.[13]

### Evaluation of generative models at HPO extraction

We found that most of the generative models we tested were able to successfully return labels of plausible clinical features, such as Brachydactyly in the example.

However, we discovered that the generative models would hallucinate many or most of the identifiers, though larger models often captured identifier patterns, such has giving inheritance terms very low HPO term identifiers (e.g., HP:0000007).

We attempted to address these hallucinations by modifying the prompt template, eventually arriving at the prompt in Appendix A1. While this resulted in some improvement, we were unable to consistently avoid hallucination. We noticed that larger models performed better, with GPT-4 more often returning the correct identifier than GPT-3.5. Several recent preprints have similarly highlighted the challenge of term identifier hallucination. In [ChatGPT for phenotypes extraction], Labbé et al evaluate using the labels to resolve correct identifiers, while using the LLM only for phenotype summarization. In [Enhancing Phenotype Recognition], Yang et al found that fine-tuning GPT-based models was able to resolve this issue.

We therefore sought to explore these two approaches in more detail. First, we investigated the related but simpler task of phenotype summarization. Second, we explored fine-tuning a generative LLM for the HPO term extraction tast.

### Phenotype summarization and inference

We designed a prompt for a phenotype summarization task, in which the LLM summarizes the phenotypic features mentioned in the given text and categorizes them as present or absent. After several iterations, we arrived at the single-shot prompt in Appendix A2.

We evaluated the performance on several example referral notes and clinical summaries and noted extremely strong performance.

Excitingly, in this exploration we noticed that GPT-3.5 exhibited an unexpected capability. It was able to infer and summarize abnormal development features based on a description of the ages at developmental milestones (Appendix A3). Inference of this kind is impossible with dictionary-based methods, and uniquely well-suited to generative LLMs.

### Fine-tuning of open-source generative models for HPO extraction

Because of the sensitivity of health data, we attempted to fine-tune the largest open-source generative model we could manage, on the hardware we could procure within the time constraints of the hackathon.

#### Lambda Labs

We decided to rent an on-demand NVIDIA H100 PCIe GPU server with 80 GB VRAM on Lambda Labs. We encountered several compatibilities issues with library versions and the pre-installed Lambda Stack environment on the GPU server and needed to perform the following additional steps:

1. Symlink the cudo libraries to the main library path:
```
for cudnn_so in /usr/lib/python3/dist-packages/tensorflow/libcudnn*; do
  sudo ln -s "$cudnn_so" /usr/lib/x86_64-linux-gnu/
done
```
2. Make sure the code looks there for the cuda libs, which means for scripts:
```
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
```
and within Python scripts:
```
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'
```
3. When running on an H100, we found the CUDA compatibility (sm_90) to be too recent for the stable branch of pytorch, so we had to upgrade pytorch to the nightly build:
```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```
4. If Python 3.10 is required, set up a miniconda environment:
```
conda create -n my-new-environment python=3.10
```

We were then able to peform proof-of-concept fine-tuning of BioGPT and gpt-neox-20b models. However, we did not have time to explore the performance of the resulting models or prepare a high-quality fine-tuning dataset.

### Future work

Our initial findings suggest that generative LLMs offer unique advantages and present unique challenges for phenotype summarization and HPO extraction from clinical text. However, these challenges may be addressable with fine-tuning, and QLoRA enables efficient fine-tuning on modest hardware.

The next steps would be to establish a quantitative evaluation framework for HPO extraction. The fine-tuning and test dataset should include cases that explicitly test phenotype inference. Fine-tuning of GPT-3.5 and open generative LLMs should then be compared to PhenoBERT.


## Extracting metadata using LLMs

### Sample metadata extraction <!-- (Tazro, Mori, Hiro, Masaki) -->

The reuse of public data in bioinformatics research relies on the accuracy and comprehensiveness of sample information. However, the description of sample information is often written in natural language by data registrants, lacking standardized vocabulary. This poses a challenge in constructing databases and knowledge bases based on public data, requiring the use of natural language processing to extract terms and map them to standardized vocabulary. In particular, when dealing with large-scale public databases containing millions of samples, such as DNA sequence data, automating the mapping process becomes essential.

The introduction of large-scale language models, like GPT3.5 provided by ChatGPT, holds promise for streamlining and improving the standardization of sample information. By utilizing GPT3.5, we have confirmed its ability to extract sample attribute information from naturally written sample descriptions and generate structured JSON data. While the API provided by OpenAI enables the processing of a substantial amount of sample information, the preference for local deployment of language models arises due to the ongoing costs associated with the API. Moreover, the standardization of sample descriptions is not only valuable in bioinformatics but also in clinical research, emphasizing the need for methods that can handle sensitive data in offline environments.

In our investigation, we explored the feasibility of using open-source language models shared by HuggingFace to extract sample attribute information. These language models, readily operationalized through the Huggingface Transformers library, offer interfaces for various natural language processing tasks such as summarization and token classification. However, when attempting to replicate the processing performed by GPT3.5, we were unable to achieve outputs with practical accuracy using pipelines like text generation and token classification.

Consequently, our task of obtaining standardized and structured data from naturally written sample information necessitates dividing the challenge into distinct natural language processing tasks. This entails simplifying the descriptions through summarization, extracting words via token classification, and ultimately mapping them to a standardized vocabulary. As there are multiple models available in the published language models, such as stanford-crfm/BioMedLM and microsoft/biogpt, selecting the appropriate model for each specific task becomes imperative.

### Timecourse <!-- (Eisuke) -->
One clinical information that clinicians value but that is not yet fully addressable by machines is the time course. In particular, the time course of clinical symptoms directly related to the patient is one of the easiest to understand not only for the clinician but also for the patient, and it is an important issue to be able to handle this by machine.　Thus, we attempted to extract patients symptoms with ChatGPT4 and found that when we extract symptoms and clinical findings without tagging with HPO, we successfully extracted these data based on time course. We also asked ChatGPT4 for a python code that would visualize the time-course of multiple symptoms in the three dimentions, and were able to produce a graph in less than three minutes.
Very interestingly, the ChatGPT was able to determine these symptoms and clinical findings as positive or negative, but not as normal or abnormal. For example, head circumference is one of the most important features in pediatrics, but its normal range varies by month and age. Even when such clinical findings were extracted along with the time series, ChatGPT4 could not be determined as normal or abnormal at this time. This is indeed an interesting and worthwhile finding to know when applying LLM to clinical practice.


## Using LLMs in graph database queries <!-- (Jerven) -->

Integrating the output of LLMs and Knowledgegraphs is possible on the fly. We wrote a demonstration of a [custom function](https://github.com/JervenBolleman/libretranslate-rdf4j), that can be used during [SPARQL](https://www.w3.org/TR/sparql11-query/) query evaluation, in this case using an open source language translation model that allows "casting" a string with a language tag to a different language. e.g. Dutch to Japanese.

```sparql
PREFIX translate : <http://example.org/libretranslate>
SELECT ?litJa ?lit
WHERE {
    VALUES (?lit) {( "goedemorgen"@nl ) ( "goodmorning"@en )}
    BIND(translate:translate(?lit, "ja") AS ?litJa)
}
```

This function build on the opensource translation model and API of [libretranslate](https://libretranslate.com/). Similar functions can be used for LLMs that predict domain specific knowledge e.g. genes or chemical interactions.

## Multimodal rare disease diagnosis based on facial images and phenotypic features within PubCaseFinder <!-- (Takayuki, Toyofumi) -->

Numerous clinical decision support systems (CDSSs) have been developed for rare disease diagnosis suggestion, but most of them are based on phenotypic similarities using HPO-based annotation. Recently, some databases share medical images such as facial images or MRI images with the FAIR principles and dynamic consent. By fine-tuning the Multi-modal LLMs using these medical images and HPO-based annotation, the Multi-modal LLMs have the potential to provide more accurate rare disease diagnostic suggestion compared to conventional CDSSs by inputting medical images as well. With the goal of incorporating the Multi-modal LLMs into PubCaseFinder, a CDSS using only HPO-based annotation, this study aims to investigate how to set up and fine tune the Multi-modal LLMs.

### Attempts to install LLaMa-adapter
We made attempts to set up the environment on four different types of computers/clouds in order to install and run LLaMA-adapter, a Multi-modal LLM. Unfortunately, we encountered issues that prevented us from successfully running the model during BH23. We will document these attempts as the outcomes.

#### Setup
```
conda create -n llama_adapter -y python=3.8
conda activate llama_adapter

# install pytorch
conda install pytorch cudatoolkit -c pytorch -y

# install dependency and llama-adapter
pip install -r requirements.txt
pip install -e .
```

#### CPU (Rocky Linux release 9.2)
The setup was not completed because we could not install the NVIDIA Driver.

#### GPU (CUDA ver 12.1,NVIDIA GeForce RFX 3060,FB memory: 12Gb)
The model installation was completed. however, it encountered a memory-related issue when attempting to run an example generation task, specifically an error message indicating insufficient CUDA memory (torch.cuda.OutOfMemoryError: CUDA out of memory).

#### NIG supercomputer (CUDA ver 11.6)
The task is still ongoing.

### Future work
Since the medical images registered in GestaltMatcher and other databases are mainly Western cases, it is necessary to investigate databases that share medical images of Asian and Japanese cases. If such databases do not exist, it is necessary to consider how to collect local medical images and promote their registration in FIAR databases such as GestaltMatcher.

## Phenotypic feature capure and ontology description using multi-modal data <!-- (Soichi) -->

We worked on an attempt to extract phenotypic terms from multimodal data such as clinical text and medical images. On the other hand, in the short term, there is the idea that AI should be able to select phenotypic features without extracting phenotypic terms and concepts. In fact, more and more studies are using image data as phenotypes to directly explore the relationship between genetic factors and not using terms such as HPO. In particular, if the phenotype ontologies and concepts is not sufficient, the extraction of phenotypic terms may reduce the accuracy. It is costly for a physician to describe a clinical phenotype using a phenotype ontology and for the physician to improve the ontology when there is no appropriate term or concept. However, the extraction of phenotypic terms is important when considering understandability to physicians and patients, and when considering the use of so-called semantic knowledge for clinical decision support.


## Acknowledgements

We would like to thank the fellow participants at BioHackathon 2023 for their collaboration and constructive advice, which greatly influenced our project. We are grateful to the organizers for providing this platform and the developers of open source language models. Special thanks to our mentors, advisors, and colleagues for their guidance and support. Without their contributions, our project in linked data standardization with LLMs in bioinformatics would not have been possible.

## References

1. LoRA: Low-Rank Adaptation of Large Language Models
https://doi.org/10.48550/arXiv.2106.09685

1. QLoRA: Efficient Finetuning of Quantized LLMs
https://doi.org/10.48550/arXiv.2305.14314

1. Structured information extraction from complex scientific text with fine-tuned large language models
https://doi.org/10.48550/arXiv.2212.05238

1. PhenoRerank
https://www.sciencedirect.com/science/article/pii/S1532046422000752

1. PhenoBERT
https://ieeexplore.ieee.org/abstract/document/9763337

1. BioGPT
https://arxiv.org/pdf/2210.10341.pdf

1. ChatGPT for phenotypes extraction: one model to rule them all?
https://hal.science/hal-04098499/document

1. LLaMa-Adapter
https://github.com/OpenGVLab/LLaMA-Adapter

1. PubCaseFinder
https://pubcasefinder.dbcls.jp/

1. OpenAI Whisper
https://github.com/openai/whisper

1. Audacity
https://www.audacityteam.org/

1. NCBO BioPortal BioOntology API Documentation
http://data.bioontology.org/documentation

1. BioOntology Recommender script
https://github.com/biohackathon-japan/bh23-llm/tree/main/scripts/bio_ontology_recommender

1. Whisper Transcription script
https://github.com/biohackathon-japan/bh23-llm/tree/main/scripts/transcription

## Appendix

### A1: HPO term extraction

#### Example clinical text

> Branchio-oculo-facial syndrome (BOFS) is a rare autosomal-dominant cleft palate-craniofacial disorder with variable expressivity. The major features include cutaneous anomalies (cervical, infra- and/or supra-auricular defects, often with dermal thymus), ocular anomalies, characteristic facial appearance (malformed pinnae, oral clefts), and, less commonly, renal and ectodermal (dental and hair) anomalies. The molecular basis for this disorder is heretofore unknown. We detected a 3.2 Mb deletion by 500K SNP microarray in an affected mother and son with BOFS at chromosome 6p24.3. Candidate genes in this region were selected for sequencing on the basis of their expression patterns and involvement in developmental pathways associated with the clinical findings of BOFS. Four additional BOFS patients were found to have de novo missense mutations in the highly conserved exons 4 and 5 (basic region of the DNA binding domain) of the TFAP2A gene in the candidate deleted region. We conclude BOFS is caused by mutations involving TFAP2A. More patients need to be studied to determine possible genetic heterogeneity and to establish whether there are genotype-phenotype correlations.

#### Prompt

> Given the following clinical text, please analyze it and identify all of the Human Phenotype Ontology (HPO) terms occurring in the text.
>
> Provide your output in the form of a table. The table should have four columns. The first column should be the exact word or words from the original text that reference an abnormal phenotypic feature. The second column should be the identifier of the referenced HPO term, and the third column should be the canonical label or name of this term. The fourth column should specify whether the referenced HPO term is present or absent in the context of the text.
>
> Here is the clinical text: <CLINICAL_TEXT>

#### Output from GPT 4 (version: May 24 2023)

> Here is the analysis of the clinical text, identifying the Human Phenotype Ontology (HPO) terms referenced in the text:
>
> | Phenotypic Feature | HPO Identifier | HPO Label | Present? |
> | --- | --- | --- | --- |
> | cleft palate | HP:0000175 | Cleft palate | Present |
> | craniofacial disorder | HP:0001363 | Craniofacial malformation | Present |
> | cutaneous anomalies | HP:0000976 | Skin rash | Present |
> | cervical defects | HP:0000791 | Cervical skin redundancy | Present |
> | infra-auricular defects | HP:0000378 | Posteriorly rotated ears | Present |
> | supra-auricular defects | HP:0000410 | Preauricular skin tag | Present |
> | dermal thymus | HP:0200058 | Dermal thymic tissue | Present |
> | ocular anomalies | HP:0008057 | Abnormality of the eye | Present |
> | characteristic facial appearance | HP:0001999 | Abnormal facial shape | Present |
> | malformed pinnae | HP:0008551 | Malformed ears | Present |
> | oral clefts | HP:0000202 | Cleft lip | Present |
> | renal anomalies | HP:0000077 | Renal hypoplasia | Present |
> | ectodermal anomalies | HP:0000967 | Abnormality of the ectoderm | Present |
> | dental anomalies | HP:0000164 | Abnormality of the dentition | Present |
> | hair anomalies | HP:0001597 | Abnormality of hair texture | Present |
> | autosomal-dominant inheritance | HP:0000006 | Autosomal dominant inheritance | Present |
> | genetic heterogeneity | HP:0001425 | Genetic heterogeneity | Present |
> | genotype-phenotype correlations | HP:0003812 | Genotype-phenotype correlation | Present |
> | TFAP2A gene mutations | HP:0005460 | Mutation of the TFAP2A gene | Present |
>
> Please note that not all terms may have an exact one-to-one mapping to HPO terms, and some terms might be synonymous or describe similar phenotypic features.

#### Output from PaLM

> | Word or phrase | HPO ID | HPO term | Present |
> |---|---|---|---|
> | cleft palate | HP:0000175 | Cleft lip and/or palate | Present |
> | cutaneous anomalies | HP:0001574 | Cutaneous anomaly | Present |
> | dermal thymus | HP:0002372 | Dermal thymus | Present |
> | ocular anomalies | HP:0000508 | Ocular anomaly | Present |
> | malformed pinnae | HP:0000366 | Malformed pinna | Present |
> | oral clefts | HP:0000176 | Oral cleft | Present |
> | renal anomalies | HP:0000113 | Renal anomaly | Absent |
> | dental anomalies | HP:0000164 | Dental anomaly | Absent |
> | hair anomalies | HP:0000598 | Hair anomaly | Absent |

#### Output from Bard

> | Term ID | Term Label | Present/Absent/Uncertain |
> | --- | --- | --- |
> | HP:0000006 | Anomaly | Present |
> | HP:0000007 | Mendelian inheritance | Present |
> HP:0000077 | Autosomal dominant inheritance | Present |
> | HP:0002526 | Brachydactyly | Present |
> | HP:0002548 | Brachydactyly type A-1 | Present |
> HP:0002558 | Shortening of middle phalanges | Present |
> | HP:0009794 | Radiography | Present |
> | HP:0002295 | Chromosome 2q | Present |
> | HP:0001927 | Recombination fraction | Uncertain |
> | HP:0002893 | Haplotype | Present |
> | HP:0002063 | Chromosome 2q35-q36 | Present |
> | HP:0002064 | Chromosome 2q35-q37 | Present |
> | HP:0000302 | Relatedness | Absent |

#### Output from BioOntology Recommender

| Term ID | Term Label |
| --- | --- |
| HP:0032320 | AFFECTED |
| HP:0000175 | CLEFT PALATE |
| HP:0001425 | GENETIC HETEROGENEITY |
| HP:0000356 | MALFORMED PINNAE |
| HP:0000001 | PHENOTYPE |
| HP:0003828 | VARIABLE EXPRESSIVITY |

Note: The output from BioOntology Recommender was filtered to only terms from the HPO ontology using [this script](https://gist.github.com/teemukataja/3752dc56665b6aba93decf9937fe40a0).

### A2: Phenotype summarization prompt for GPT

> Given a clinical text, please analyze it and summarize the patient's phenotypic profile using terms from the Human Phenotype Ontology (HPO). A patient profile should list the noteworthy clinical signs and physical findings that are 1) observed in the patient and 2) investigated and not observed in the patient. For example:
>
> Input: Grandparents noticed abnormality in his gait in 2015. Grandparents noted: Toe walking, frequent falling, difficulty climbing stairs. Complains of having shortness of breath at time. No dysmorphic features.
>
> Output:
> - Observed: gait abnormalities, toe walking, frequent falling, shortness of breath
> - Absent: dysmorphic features.
>
> Input: <CLINICAL_SUMMARY>
>
> Output:


### A3: Phenotype inference

#### Clinical summary

> Development: rolled at 6, crawled at 7, walked at 13, single words before 2.

#### Output from ChatGPT 3.5 (May 24)

> Observed: delayed motor milestones (rolled at 6, crawled at 7, walked at 13)
> Absent: delayed language development (single words before 2)