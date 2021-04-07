# Adolescent resting state MEG

[ILABS](www.ilabs.uw.edu) CNS resting-state magnetoencephalography component of multimodal neuroimaging, personality, and cognition measurements across adolescence.

## Run

Point [mnefun-pipline](genz/processing/mnefun-pipeline.py) to appropriate data server(s) & provide user credentials.
Script should fetch data from storage and create legacy [MNEFUN](https://github/labsn/mnefun.git) directory for each subject in {config:subjects}.
