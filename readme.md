# PEPMHC
A Siamese BERT model to predict peptide-HLA immunogenic binding.

### Variable Params
- peptide_max_length [48]
- maximum MHC length [350]
- MLM masking ratio [0.25]
- last saved checkpoint step [44000]

## About
1. Peptide pretraining run files format: 'pretraining_LLM_{}_mlm_{}.out'.format(peptide_max_length, mask rate)
2. Siamese BERT run files format: 'prot_bert_{}_{}_mlm_{}_{}.out'.format(peptide_max_length, mhc_max_length, mask_rate, previous training steps for peptide bert)
if followed by 'new_split', then it represents new mhc split in train and test sets. 
3. bash commands: 
    (1) LLM pretraining: 'train_llm.sh' for single gpu, and 'train_llm_ddp.sh' for multiple gpus (max 4 in ccdb cedar clusters). 
    (2) Siamese BERT training: 'train_protbert.sh'. I tested using multi-gpus, however it is even slower than using single gpus. So I recommend using single gpu for now.
4. virtual machine package requirements: 'requirements_pep.txt' for LLM pretraining  and 'requirements_prot_bert' for Siamese BERT training
5. Sample interactive run with 4 gpus: 'interactive'
6. Result analysis: sample code 'plot.py' 
7. My working log on this dataset: 'MHCAttNet_log'
8. Model architecture details: 'Our_model_details'

