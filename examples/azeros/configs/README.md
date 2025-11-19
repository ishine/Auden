# Data & Lhotse Manifests (AZeroS)

This guide explains how to prepare datasets and Lhotse CutSet manifests for the **Auden-AZeroS** examples.  
We provide manifests for training the Auden-AZeroS, but **you have to download the datasets firstly** and update absolute paths as described below.

---

## Manifest Format

The model expects **Lhotse CutSet manifests** (`.jsonl.gz`), where each cut contains:

- Speech recording information  
- Supervision metadata with attributes: `text` and optional `emotion`, `gender`, `age_group`
- Supervision with self-generated response: `input_text`, `instruction`, `response`, `from_model`

Example supervision structure (in `SIFT_ssp` mode, system-message ommited.):

```json
{
  "id": "1079_WSI_SAD_XX-0",
  "start": 0,
  "duration": 2.6693125,
  "channel": 0,
  "supervisions": [
    {
      "id": "1079_WSI_SAD_XX",
      "recording_id": "1079_WSI_SAD_XX",
      "start": 0.0,
      "duration": 2.6693125,
      "channel": 0,
      "text": "we'll stop in a couple of minutes",
      "language": "en",
      "speaker": "1079",
      "gender": "female",
      "custom": {
        "age": "21",
        "emotion": "sad",
        "age_group": "young adult",
        "response": "I'm sorry to hear that you're feeling sad. It's okay to take breaks when you need them. Is there anything you'd like to talk about or anything I can help with?",
        "instruction": "",
        "input_text": "<audio><meta>age: young adult, gender: female, emotion: sad</meta><text>we'll stop in a couple of minutes</text></audio>",
        "from_model": "Qwen/Qwen2.5-7B-Instruct"
      }
    }
  ],
  "recording": {
    "id": "1079_WSI_SAD_XX",
    "sources": [
      {
        "type": "file",
        "channels": [
          0
        ],
        "source": "myfolder/data/crema/AudioWAV/1079_WSI_SAD_XX.wav"
      }
    ],
    "sampling_rate": 16000,
    "num_samples": 42709,
    "duration": 2.6693125,
    "channel_ids": [
      0
    ]
  },
  "type": "MonoCut"
}
```

### Dataset Download

These datasets are characterized by different supervision fields on semantic and paralinguistic attributes.

⚠️ Please follow each dataset’s license terms and cite the corresponding papers properly.

### 1. WenetSpeech

Tags: `zh`, `transcript`.

Download instructions: [WenetSpeech GitHub](https://github.com/wenet-e2e/WenetSpeech). Follow the official guide to download the audios files. Only the `wavs/train_l` folder is used.

```bash
wenetspeech/
└── wavs/
    └── train_l/
        └── Y0000000000_--5llN02F84/
            ├── Y0000000000_--5llN02F84_S00000.wav
            ├── Y0000000000_--5llN02F84_S00002.wav
            └── ...
```

License: Distributed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

### 2. GigaSpeech

Tags: `en`, `transcript`.

Download instructions: [GigaSpeech Github](https://github.com/SpeechColab/GigaSpeech). Follow the official guide to download the audios files. Only the `wavs/train_l` folder is used.

```bash
gigaspeech/
└── train/
    └── l_files_additional/
        └── l_chunks_0000/
            └── Y0000000000_--5llN02F84
            └── ...
        └── ...
    └── m_files_additional/
    └── s_files_additional/
    └── xl_files_additional/
    └── xs_files/
```

License: Distributed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

### 3. CommonVoice

Tags: `multilingual`, `transcript`, `gender`, `age`.

Download instructions: [CommonVoice Website](https://datacollective.mozillafoundation.org/datasets/cmflnuzw52mzok78yz6woemc1). Download CommonVoice datasets of english and other optional languages (zh, ja, ko, fr, es, pt, ru, vi, id, etc.) from the website. Only the `xx/clips` folders are used. The provided manifests are based on the `20.0` version.

```bash
commonvoice/
└── en/
    └── clips/
        └── common_voice_en_130014.mp3
        └── ...
└── zh/
└── ...
```

License: Distributed under the [CC0-1.0 License](https://creativecommons.org/publicdomain/zero/1.0/legalcode).

#### 4. IEMOCAP

Tags: `en`, `pseudo-transcript`, `gender`, `emotion`.

Download instructions: Request access from [USC SAIL](https://sail.usc.edu/iemocap/iemocap_release.htm) by completing the release form.  
After extracting, the folder structure should look like:

```bash
IEMOCAP_full_release/
├── Session1/
│   └── sentences/wav/
│       ├── Ses01F_impro01/
│           ├── Ses01F_impro01_F000.wav
│           └── ...
│       ├── Ses01M_impro01/
│       └── ...
├── Session2/
│   └── sentences/wav/
│       └── ...
```

License: Refer to [IEMOCAP Data Release Form](https://sail.usc.edu/iemocap/Data_Release_Form_IEMOCAP.pdf)


### 5. CREMA-D

Tags: `en`, `pseudo-transcript`, `gender`, `age`, `emotion`.

Download instructions: [CREMA-D GitHub](https://github.com/CheyneyComputerScience/CREMA-D). Use `git lfs` to download large audio files.  Only the `AudioWAV` folder is used.

```bash
crema/
└── AudioWAV/
    ├── 1001_DFA_ANG_XX.wav
    ├── 1077_TAI_DIS_XX.wav
    └── ...
```

License: Distributed under the [Open Database License (ODbL)](http://opendatacommons.org/licenses/odbl/1.0/).

### 6. MELD

Tags: `en`, `pseudo-transcript`, `gender`, `emotion`.

Download instructions: [MELD GitHub](https://github.com/declare-lab/MELD). Follow instructions to download the data. Only the `MELD_train_wav` folder is used.

```bash
meld/
└── MELD_train_wav/
    ├── dia9_utt9.wav
    ├── dia9_utt8.wav
    └── ...
```

License: Distributed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

### 7. RAVDESS

Tags: `en`, `pseudo-transcript`, `gender`, `emotion`.

Download instructions: [Zenodo Record](https://zenodo.org/records/1188976). We only need `Audio_Speech_Actors_01-24.zip` (no video files):

```bash
wget -c https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip
unzip Audio_Speech_Actors_01-24.zip -d ravdess
```

Expected structure:

```bash
ravdess/
 ├── Actor_01/
 │   ├── 03-01-01-01-01-01-01.wav
 │   ├── 03-01-01-01-02-01-01.wav
 │   └── ...
 ├── Actor_02/
 └── Actor_24/
```

License: Distributed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

### 8. TESS

Tags: `en`, `pseudo-transcript`, `gender`, `age`, `emotion`.

Download via [Scholars Portal Dataverse](https://utoronto.scholaris.ca/collections/036db644-9790-4ed0-90cc-be1dfb8a4b66). Unzip the downloaded `dataverse_files.zip`.

Expected structure:

```bash
TESS/
 ├── MANIFEST.TXT
 ├── OAF_back_angry.wav
 ├── OAF_back_disgust.wav
 └── ...
```
License: Distributed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

### 9. DailyTalk

Tags: `en`, `transcript`, `emotion`.

Download Instructions: [DailyTalk Github](https://github.com/keonlee9420/DailyTalk). Follow instructions to download the data.

```bash
dailytalk/
└── data
    ├── 0
        └── 0_1_d0.wav
        └── 10_1_d0.wav
        └── ...
    ├── 1
    └── ...
```

License: Distributed under [MIT License](https://opensource.org/license/mit)

### 10. AISHELL-1

Tags: `zh`, `transcript`, `gender`.

Download Instructions: [AISHELL OpenSLR](https://www.openslr.org/33/). Only the `data_aishell/wav/train` folder is used.

```bash
aishell1/
└── data_aishell/
    └── wav/
        └── train/
            └── S0002
                └── BAC009S0002W0122.wav
                └── ...
            └── ...
```

License: Distributed under [Apache License v.2.0](https://www.apache.org/licenses/LICENSE-2.0)

### 11. EmotionTalk

Tags: `zh`, `transcript`, `gender`, `emotion`.

Download Instructions: [EmotionTalk Huggingface](https://huggingface.co/datasets/BAAI/Emotiontalk/tree/main). Download data from huggingface dataset and extract the audio files. Only the `Audio/wav` folder is used.

```bash
emotiontalk/
└── Audio/
    └── wav/
        └── G00001/
            └── G00001_02/
                └── G00001_02_01/
                    └── G00001_02_01_002.wav
                    └── ...
                └── ...
            └── ...
        └── ...
```

License: Distributed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

### 12. CS-Dialogue

Tags: `zh/en`, `transcript`, `gender`, `age`.

Download Instructions: [CS-Dialogue Huggingface](https://huggingface.co/datasets/BAAI/CS-Dialogue/tree/main). Download data from huggingface dataset. Only the `data/short_wav/WAVE/C0/` folder is used.

```bash
cs_dialogue/
└── data/
    └── short_wav/
        └── WAVE/
            └── C0/
                └── ZH-CN_U0001_S0/
                    └── ZH-CN_U0001_S0_101.wav
                    └── ...
                └── ...
```

License: Distributed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

### 13. VoxCeleb2

Tags: `mutlilingual`, `pseudo-transcript`, `gender`.

Download **VoxCeleb2 dev** set from [KAIST MM Lab](https://mm.kaist.ac.kr/datasets/voxceleb/).  
Conversion from `.m4a` → `.wav` requires `ffmpeg`:

```bash
python convert_voxceleb2_m4a_to_wav.py
```

Expected structure:

```bash
voxceleb2/
├── dev/
│   ├── aac/
│   │   └── id00012/abc123/00001.m4a
│   └── wav/   # converted output
│       └── id00012/abc123/00001.wav
```

License: Distributed under [Creative Commons BY 4.0](https://creativecommons.org/licenses/by/4.0/)


---

## Manifest Preparation

After all datasets are downloaded and organized, you may either link the data sources to the same relative path, or replace the relative paths inside the manifests with their actual paths.

For latter case, please edit the `USER_PATHS` section and run the script `python scripts/data_generation/replace_manifest_path.py`:


**Notes on Provided Manifests**

- Label normalization: Emotion labels are standardized across datasets (e.g., `"angry"`, `"happy"`, `"sad"`).
- Age normalization:  Ages are grouped into:
   - `teenager` (<18)  
   - `young adult` (18–39)  
   - `middle-age adult` (40–60)  
   - `senior` (>60)
- Pseudo-Transcript: We apply ASR models to add text transcriptions for all the datasets.

- Using custom data to train with your own dataset:
   - Refer to `scripts/data_generation/run_data_generation.sh` to generate self-generated data manifests.
   - Follow the same JSON manifest structure.  
   - Each supervision must include `text`, and optionally `gender`, `emotion`, `age`, and `age_group`.  
   - Missing values should not be presented or be `"null"`/`None`, and will be discarded.

---


### Data Configuration

Update the following files with your manifest paths as needed:
- `configs/data_configs/train_data_config.yaml`: Training datasets
- `configs/data_configs/valid_data_config.yaml`: Validation datasets
- `configs/data_configs/test_data_config.yaml`: Test datasets

