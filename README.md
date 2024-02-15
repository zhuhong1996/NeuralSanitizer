# NeuralSanitizer

### ABOUT

This repository contains code implementation of NeuralSanitizer. The datasets and backdoored models can be downloaded [here](https://mailsucasaccn-my.sharepoint.com/:f:/g/personal/zhuhong18_mails_ucas_ac_cn/Ei16AlmTNJVMm9fV0TfYV2wBHAb0mbOVYRQs-RyNPgw2gg?e=Lx82cd).

### DEPENDENCIES

Our code is implemented and tested on TensorFlow. Following packages are used by our code.
- `python==3.6.13`
- `numpy==1.17.0`
- `tensorflow-gpu==1.15.4`
- `opencv==3.4.2`

### HOW TO DETECT PATCH-BASED BACKDOORS

#### Partial Neural Network Initialization and Retraining (PNNIR)

Please run the following command.

```bash
python pnnir.py
```

This script will load the to-be-examined model and generate seven tuned models. 

#### Potential Triggers Reconstruction

Please run the following command.

```bash
python potential_triggers_reconstruction.py
```

This script will load the to-be-examined model and one tuned model generated in the previous step, and reconstruct a potential trigger for each label. 

#### Critical Features Preservation

Please run the following command.

```bash
python critical_features_preservation.py
```

This script will load the to-be-examined model and the potential triggers generated in the previous step, and preserve the critical features (remove unrelated features). 

#### Backdoor Detection

Please run the following command.

```bash
python backdoor_detection.py
```

This script will load the to-be-examined model and the potential triggers, and generate the results of backdoor detection. 


### HOW TO DETECT FEATURE SPACE BACKDOORS

#### Partial Neural Network Initialization and Retraining (PNNIR)

Please run the following command.

```bash
python pnnir.py
```

This script will load the to-be-examined model and generate seven tuned models, which is the same as detecting patch-based backdoors. 

#### Potential Triggers Reconstruction

Please run the following command.

```bash
python potential_triggers_reconstruction_feature_space.py
```

This script will load the to-be-examined model and one tuned model generated in the previous step, and reconstruct a potential trigger for each label. 

#### Backdoor Detection

Please run the following command.

```bash
python backdoor_detection_feature_space.py
```

This script will load the to-be-examined model and the potential triggers, and generate the results of backdoor detection. 
