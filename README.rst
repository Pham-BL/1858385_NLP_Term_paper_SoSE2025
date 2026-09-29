ViT5 as 1-best Error Correction model for Vietnamese spoken question answering

Requirements
----
requirements_vit5.txt - to run fine-tuning on the vit5 model
requirements_whisper.txt - to run phoasr - afaik only transformers==4.48.0 is hard requirement


Source code
----
RQ1 - src/rq1/
--
phoasr_inference.ipynb - runs asr model on ViSQA audio context files to generate transcriptions, runs in configurable batches
vit5_train.ipynb - train file for vit5
vit5_inference.ipynb - inference file for vit5

RQ2 - src/rq2/
--
bartpho.ipynb
phobert.ipynb
Vit5-qa.ipynb
xlmr.ipynb

Utility files - src/util/
--
transcription_matching.ipynb - matches audio context files with corresponding id field in the transcription json. use after phoasr_inference.ipynb
wer_calc.ipynb - calculates wer
uit_viquad.ipynb - extracts data from the UiT-ViQUAD 2.0 dataset to a json file as ground truth in the study
vit5_ft_set_creating - creates the sets for fine-tuning vit5 in rq1

Test results - data/rq1/; data/rq2/output/
----
rq1/ec/output/base_vit5 - results for mean wer of base vit5 ("mean_vit5_wer") compared to mean wer of no ec model ("mean_asr_wer")
rq1/ec/output/ft_vit5 - results for mean wer of fine-tuned vit5 ("mean_vit5_wer") compared to mean wer of no ec model ("mean_asr_wer")
output/slide - results for mean em and f1 of different models. data from this folder was used in the final submission
output/noslide - results for mean em and f1 of different models. the predict function for bartpho, xlmr and vit5 used differently here, thus they were separated; the result trends were mostly the same

Input files - data/rq1/; data/rq2/input/
----
rq1/ec/input/output_phoasr_train.csv - used in fine-tuning vit5 ec model
rq1/ec/input/output_phoasr_test.csv - used in inferencing vit5 ec model
rq1/ec/input/output_visqa_test.csv - used in inferencing vit5 ec model
rq2/input/context_mapping_phoasr_VIQUAD_val.json - matched baseline phoasr transcripts to viquad ground truth, used in rq2 as baseline context input for running the models. also includes the questions and ground truth answers for comparison in rq2. *Note: ignore "summary" section, it was used to match audio files and transcripts together early on and was not removed after
rq2/input/context_mapping_visqa_VIQUAD_val.json - matched baseline visqa transcripts to viquad ground truth, used in rq2 as baseline context input for running the models. also includes the questions and ground truth answers for comparison in rq2. *Note: ignore "summary" section, it was used to match audio files and transcripts together early on and was not removed after
rq2/input/ft_PhoASR_VIQUAD_val.json - matched corrected phoasr transcripts to viquad ground truth, used in rq2 as corrected context input for running the models. also includes the questions and ground truth answers for comparison in rq2. *Note: ignore "summary section, it was used to match audio files and transcripts together early on and was not removed after
rq2/input/ft_VISQA_VIQUAD_val.json - matched corrected visqa transcripts to viquad ground truth, used in rq2 as baseline context input for running the models. also includes the questions and ground truth answers for comparison in rq2. *Note: ignore "summary" section, it was used to match audio files and transcripts together early on and was not removed after
rq2/uit_viquad_train.json - ground truth file, extracted from uit-viquad 2.0 train set
uit_viquad_val.json - ground truth file, extracted from uit-viquad 2.0 validation set


Demo dataset - data/demo - intended to make a demo featuring a sample of the ViSQA dataset for every source code in the repository but not enough time as of now
 