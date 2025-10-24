# AI-for-Health-Revolutionizing-Clinical-Decision-Support

## Fine-tuning Datasets

This repository contains the fine-tuning data for the DDI chatbot.

### TWOSIDES Dataset
- Validation and test files are included in the repo:
  - `data/finetune/twosides/valtwosides.jsonl`
  - `data/finetune/twosides/testtwosides.jsonl`
- The **training file** (too large for Git) can be downloaded from:
  [Download train_twosides.jsonl (127 MB)](https://github.com/Muhammed-Farrag/AI-for-Health-Revolutionizing-Clinical-Decision-Support/releases/download/v1.0/traintwosides.jsonl)

### OFFSIDES Dataset
All files are available directly in:
- `data/finetune/offsides/{trainoffside,valoffside,testoffside}.jsonl`

File                                   Description     

`TwosidesCleaning&TuningCode.ipynb`  :  Full cleaning and fine-tuning preparation workflow for TWOSIDES.

`offside_cleaning_code.ipynb`        : Data-cleaning notebook for OFFSIDES.

`offsideFineTuningCodeipynb.ipynb`   : Fine-tuning code and analysis for OFFSIDES.
  
`cleaned_offside_data.csv`           : The cleaned OFFSIDES dataset used to generate fine-tuning splits.
  
`twosides_ddi_rag_ready`             : *(Not uploaded — too large)*. This file is automatically created by the TWOSIDES cleaning notebook during execution, so the full data can be regenerated locally. 

