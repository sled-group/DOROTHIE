# SDN dataset file explanation
The extracted SDN folder consists of train, validate and test folders. For each session inside these splits, the structure is 
```bash
$ tree log_1651158519
log_1651158519
├── ad_wizardlog.csv
├── annotated_log.json
├── carlalog.log
├── config.json
├── event_bert_emb.pkl
├── event_emb.pkl
├── plan.json
├── rgb
│   ├── 0.png
│   ...
│   └── 9.png
├── rgb_emb.pkl
├── speech.pkl
├── trajectory.csv
└── utterance
    ├── utterance_0.flac
    ...
    └── utterance_9.flac
```
**config.json**:This file contains the session configuration, including information about the map, street names, and other settings.

**annotate_log.json**: This file contains a log of the entire session, including all speech inputs, dialogues, and physical movements of the agent.

**plan.json**: This file contains the agent's whole plan for the current session.

**utterance**: This folder contains all the speech input files.

**ad_wizardlog.csv** and **carlalog.log**: These files can be used together with the Carla replay function to generate the rgb folder and the trajectory.csv.

**rgb**: This folder contains all the RGB images captured at a rate of 10Hz using the carla replay function.

**trajectory.csv**: This file contains the trajectory of the vehicle during the session.

**rgb_emb.pkl** and **speech.pkl**: These files contain embeddings of the RGB images and speech, respectively, using ResNet50 and HuBert models.

**rgb_def_detr.pkl**: This file contains the object-detection proposal for each RGB image encoded by a pre-trained Deformable DETR and Segformer model.

**event_emb.pkl**: This file contains an embedding for each event in the session log. It was constructed using the prep_dataset.py and can be used for training, validation, and testing.

**event_bert_emb.pkl**: This file is similar to event_emb.pkl but is used for fine-tuning BERT as an ablation study.