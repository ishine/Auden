# Auden Design Philosophy and Usage Guide

## Table of Contents
1. [Design Philosophy](#design-philosophy)
2. [Core Architecture](#core-architecture)
3. [Quick Start](#quick-start)
4. [How to Build Your Own Project](#how-to-build-your-own-project)
5. [Best Practices](#best-practices)

---

## Design Philosophy

Auden is a research toolbox for audio and multimodal understanding tasks. Inspired by HuggingFace Transformers, it aims to provide:

### 1. **Unified User Experience**
- HuggingFace-style `Auto*` APIs (`AutoConfig`, `AutoModel`)
- Separation of configuration and model for easy management and reuse
- Seamless switching between local paths and HuggingFace Hub

### 2. **Modularity and Extensibility**
- All components are independently designed but freely composable
- Quickly implement new tasks by inheriting base classes
- Plugin-style model registration system

### 3. **Complete Training Infrastructure**
- Out-of-the-box distributed training support (DDP/FSDP)
- Mixed precision training, model averaging, checkpoint management
- Unified data processing pipeline based on Lhotse

### 4. **Support for Multiple Audio Tasks**
- ASR (Automatic Speech Recognition)
- CLAP (Contrastive Learning)
- Audio Captioning
- Speech-LLM
- Audio Tagging
- And more...

---

## Core Architecture

### Architecture Diagram

```
auden/
├── auto/                    # Auto loading system
│   ├── auto_config.py      # Config auto-loading
│   └── auto_model.py       # Model auto-loading
├── models/                  # Model implementations
│   ├── base/               # Base classes
│   │   ├── model_config.py # BaseConfig (config base class)
│   │   └── model.py        # BaseModel (model base class)
│   ├── asr/                # ASR models
│   ├── clap/               # CLAP models
│   ├── audio_caption/      # Audio Caption models
│   └── ...                 # More tasks
├── trainer/                # Trainers
│   ├── ddp_trainer.py      # BaseTrainer (DDP training base)
│   └── fsdp_trainer.py     # FSDP training support
├── data/                   # Data processing
│   └── lhotse_datamodule.py # BaseLhotseDatamodule (data base class)
├── optim/                  # Optimizers and schedulers
│   ├── optimizer.py        # ScaledAdam, etc.
│   └── scheduler.py        # Eden series schedulers
└── utils/                  # Utility functions
    ├── checkpoint.py       # Checkpoint management
    ├── metric_tracker.py   # Metrics tracking
    └── ...

examples/                    # Example projects
├── asr/                    # ASR example
│   ├── train.py           # Training entry point
│   ├── trainer.py         # Task-specific trainer
│   ├── data_module.py     # Task-specific data module
│   └── configs/           # Hydra config files
├── clap/                   # CLAP example
├── audio_caption/          # Audio Caption example
└── ...                     # More examples
```

### Core Components Explained

#### 1. **Auto System** - Unified Model/Config Loading Interface

**Design Goal**: Mimic HuggingFace's user experience, so users don't need to worry about specific class names.

**AutoConfig** (`auden/auto/auto_config.py`):
- Automatically selects the correct config class based on `model_type`
- Supports loading from local paths and HuggingFace Hub
- Lazy loading mechanism to avoid importing all modules at startup

```python
# Load config from pretrained model
config = AutoConfig.from_pretrained("path/to/model")  # or HF repo ID

# Create config for specific model type
config = AutoConfig.for_model("asr", encoder_config=encoder_config)
```

**AutoModel** (`auden/auto/auto_model.py`):
- Automatically selects the correct model class based on config
- Supports loading weights from checkpoints
- Works with AutoConfig

```python
# Load model with weights from pretrained
model = AutoModel.from_pretrained("path/to/model")

# Create model from config (random initialization)
model = AutoModel.from_config(config)
```

**Registration Mechanism**: Support for dynamically registering custom models
```python
from auden.auto import register_model, register_config

# Register custom config
register_config("my-model", "examples.my_model.config", "MyConfig")

# Register custom model
register_model("my-model", "examples.my_model.model", "MyModel")

# Now you can use Auto* APIs
config = AutoConfig.from_pretrained("path/to/my-model")
model = AutoModel.from_pretrained("path/to/my-model")
```

#### 2. **Base Classes** - Inheritable Foundation Components

**BaseConfig** (`auden/models/base/model_config.py`):
- Base class for all model configurations
- Provides JSON serialization/deserialization
- Must set `model_type` field

```python
class MyModelConfig(BaseConfig):
    model_type = "my-model"  # Required, for Auto system identification
    
    # Define config parameters
    hidden_size: int = 512
    num_layers: int = 6
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
```

**BaseModel** (`auden/models/base/model.py`):
- Provides reference implementation for model save/load
- **Not meant to be inherited** - just an example and reference
- Your models should be simple `nn.Module`s

**BaseTrainer** (`auden/trainer/ddp_trainer.py`):
- Provides complete training loop and infrastructure
- Supports DDP distributed training, mixed precision, model averaging
- Abstract method: `_forward_one_batch()` - must be implemented

```python
class MyTrainer(BaseTrainer):
    def _forward_one_batch(self, batch, is_training=True):
        """
        Implement forward pass and loss computation for a single batch
        
        Args:
            batch: Data batch
            is_training: Whether in training mode
            
        Returns:
            loss: Loss value (Tensor)
            metrics: Metrics dict or MetricsTracker object
        """
        # Implement your forward pass logic
        outputs = self.model(batch["inputs"])
        loss = F.cross_entropy(outputs, batch["targets"])
        
        # Record metrics
        metrics = MetricsTracker()
        metrics.set_value("loss", loss.item())
        
        return loss, metrics
```

**BaseLhotseDatamodule** (`auden/data/lhotse_datamodule.py`):
- Base class for Lhotse-based data processing
- Provides feature extraction, data augmentation, sampler configuration
- Abstract methods: `setup_train()`, `setup_valid()` - must be implemented

```python
class MyDatamodule(BaseLhotseDatamodule):
    def setup_train(self):
        """Setup training data loader"""
        # 1. Load CutSet
        cutset = CutSet.from_file("train.jsonl.gz")
        
        # 2. Filter and process
        cutset = self._filter_cutset(cutset, split="train")
        
        # 3. Create sampler
        sampler = self._build_train_sampler(cutset)
        
        # 4. Create DataLoader
        self.train_dl = DataLoader(...)
    
    def setup_valid(self):
        """Setup validation data loader"""
        # Similar to training data setup
        self.valid_dls = [...]
        self.valid_names = [...]
```

#### 3. **Checkpoint Management** - Training Checkpoints vs Model Checkpoints

Auden distinguishes between two types of checkpoints:

**Training Checkpoints** (Trainer Checkpoints):
- For resuming training
- Contains: model, optimizer, scheduler, scaler, training progress
- Functions: `save_trainer_checkpoint()`, `load_trainer_checkpoint()`

**Model Checkpoints** (Model Checkpoints):
- For deployment/inference
- Contains: model weights + config
- Functions: `model.save_pretrained()`, `model.from_pretrained()`

```python
# Training checkpoint - automatically handled in trainer
save_trainer_checkpoint(
    filename="checkpoint-1000.pt",
    model=model,
    model_avg=model_avg,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    batch_idx_train=1000,
)

# Generate model checkpoint from training checkpoints (for deployment)
from auden.utils.checkpoint import generate_model_checkpoint_from_trainer_checkpoints

generate_model_checkpoint_from_trainer_checkpoints(
    model_dir="./exp_dir",
    iters=1000,
    avg=5,  # Average last 5 checkpoints
    model_name="final_model.pt"
)
```

---

## Quick Start

### 1. Load Pretrained Models

```python
from auden.auto import AutoModel, AutoConfig

# Method 1: Load model with weights directly
model = AutoModel.from_pretrained("path/to/model")

# Method 2: Load config first, then create model
config = AutoConfig.from_pretrained("path/to/model")
model = AutoModel.from_config(config)  # Note: random initialization, no weights

# Method 3: Create config for specific type
config = AutoConfig.for_model("zipformer", hidden_size=512)
```

### 2. View Supported Models

```python
from auden.auto import list_available_models, list_available_configs

print("Available models:", list_available_models())
print("Available configs:", list_available_configs())
```

### 3. Use Existing Examples

```bash
# Single GPU training
cd examples/asr
python train.py exp_dir=./exp/my_exp

# Multi-GPU DDP training
torchrun --nproc_per_node=8 train.py exp_dir=./exp/my_exp
```

---

## How to Build Your Own Project

### Complete Workflow

Suppose you want to implement a new task "MyTask". Here are the complete steps:

#### Step 1: Create Project Structure

```bash
examples/mytask/
├── train.py              # Training entry point
├── trainer.py            # Task-specific trainer
├── data_module.py        # Task-specific data module
├── configs/              # Hydra configs
│   ├── train.yaml
│   └── data_configs/
│       ├── train_data_config.yaml
│       └── valid_data_config.yaml
└── scripts/              # Training scripts
    └── train.sh
```

#### Step 2: Define Model Configuration

```python
# examples/mytask/model_config.py (if custom model needed)
from auden.models.base.model_config import BaseConfig

class MyTaskConfig(BaseConfig):
    model_type = "mytask"  # Required, unique identifier
    
    # Define model parameters
    encoder_config: dict = None  # Encoder config
    decoder_dim: int = 512
    num_classes: int = 10
    dropout: float = 0.1
    
    def __init__(self, encoder_config=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder_config = encoder_config
```

#### Step 3: Define Model

```python
# examples/mytask/model.py (if custom model needed)
import torch.nn as nn
from auden.auto import AutoModel

class MyTaskModel(nn.Module):
    # Required: specify config class
    config_class = MyTaskConfig
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Use Auto system to load encoder
        self.encoder = AutoModel.from_config(config.encoder_config)
        
        # Define task-specific layers
        self.decoder = nn.Linear(
            self.encoder.encoder_out_dim,
            config.num_classes
        )
        
    def forward(self, x, x_lens):
        # Encoder
        encoder_output = self.encoder(x, x_lens)
        encoder_out = encoder_output["encoder_out"]
        
        # Decoder
        logits = self.decoder(encoder_out.mean(dim=1))
        
        return {"logits": logits}
    
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        """Support loading from pretrained model"""
        # Can refer to implementation in auden/models/base/model.py
        pass
    
    def save_pretrained(self, save_dir):
        """Save model"""
        # Can refer to implementation in auden/models/base/model.py
        pass
```

#### Step 4: Register Model (Optional but Recommended)

If you want to use `AutoModel` to load your model:

```python
# At the beginning of examples/mytask/train.py
from auden.auto import register_model, register_config

# Register config and model
register_config("mytask", "examples.mytask.model_config", "MyTaskConfig")
register_model("mytask", "examples.mytask.model", "MyTaskModel")
```

#### Step 5: Implement Trainer

```python
# examples/mytask/trainer.py
from auden.trainer.ddp_trainer import BaseTrainer
from auden.utils.metric_tracker import MetricsTracker

class MyTaskTrainer(BaseTrainer):
    def _forward_one_batch(self, batch, is_training=True):
        """
        Must implement: forward pass for a single batch
        
        Returns:
            loss: Loss value (Tensor)
            metrics: MetricsTracker object
        """
        # 1. Prepare data
        device = self.device
        x = batch["inputs"].to(device)
        x_lens = batch["supervisions"]["num_frames"].to(device)
        labels = batch["labels"].to(device)
        
        # 2. Forward pass
        with torch.set_grad_enabled(is_training):
            outputs = self.model(x, x_lens)
            logits = outputs["logits"]
            
            # 3. Compute loss
            loss = F.cross_entropy(logits, labels)
        
        # 4. Record metrics
        metrics = MetricsTracker()
        metrics.set_value("loss", loss.item())
        
        # Compute accuracy
        pred = logits.argmax(dim=-1)
        acc = (pred == labels).float().mean()
        metrics.set_value("accuracy", acc.item())
        
        return loss, metrics
    
    # Optional: Override validation logic
    def validate(self, epoch: int):
        """Custom validation logic (if special evaluation needed)"""
        # Can call super().validate(epoch) to use default logic
        # Or implement your own validation logic
        super().validate(epoch)
```

#### Step 6: Implement DataModule

```python
# examples/mytask/data_module.py
import yaml
from lhotse import CutSet
from lhotse.dataset import DynamicBucketingSampler
from torch.utils.data import DataLoader

from auden.data.lhotse_datamodule import BaseLhotseDatamodule, _SeedWorkers

class MyTaskDatamodule(BaseLhotseDatamodule):
    def setup_train(self):
        """Setup training data"""
        # 1. Load data config
        with open(self.cfg.train_data_config, "r") as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)
        
        # 2. Build CutSet
        cutset = self._build_train_mux_cutset(train_config)
        
        # 3. Filter (optional)
        cutset = self._filter_cutset(cutset, split="train")
        
        # 4. Create sampler
        sampler = self._build_train_sampler(cutset)
        
        # 5. Create Dataset (customize based on task)
        from lhotse.dataset import K2SpeechRecognitionDataset
        dataset = K2SpeechRecognitionDataset(
            input_strategy=self.input_strategy,
            cut_transforms=self.transforms,
            input_transforms=self.input_transforms,
            return_cuts=True,
        )
        
        # 6. Create DataLoader
        self.train_dl = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=None,
            num_workers=self.cfg.get("num_workers", 8),
            persistent_workers=True,
        )
    
    def setup_valid(self):
        """Setup validation data"""
        with open(self.cfg.valid_data_config, "r") as f:
            valid_config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.valid_dls = []
        self.valid_names = []
        
        for valid_set in valid_config:
            cutset = CutSet.from_file(valid_set["manifest"])
            cutset = self._filter_cutset(cutset, split="valid")
            
            # ... create sampler and dataloader
            self.valid_dls.append(valid_dl)
            self.valid_names.append(valid_set["name"])
```

#### Step 7: Write Training Entry Point

```python
# examples/mytask/train.py
import logging
import os

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from data_module import MyTaskDatamodule
from trainer import MyTaskTrainer

from auden.auto import AutoConfig, AutoModel
from auden.auto import register_config, register_model

# Register custom model (if any)
# register_config("mytask", "examples.mytask.model_config", "MyTaskConfig")
# register_model("mytask", "examples.mytask.model", "MyTaskModel")


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    logging.info("\n" + OmegaConf.to_yaml(cfg))
    
    # 1. Setup DDP
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    
    # 2. Create experiment directory
    os.makedirs(cfg.exp_dir, exist_ok=True)
    
    # 3. Load/create model
    # Method A: If using existing encoder
    encoder_config = AutoConfig.from_pretrained(cfg.model.encoder.pretrained_model)
    config = AutoConfig.for_model(
        cfg.model.model_type,
        encoder_config=encoder_config
    )
    model = AutoModel.from_config(config)
    
    # Method B: If completely custom
    # from model import MyTaskModel
    # from model_config import MyTaskConfig
    # config = MyTaskConfig(...)
    # model = MyTaskModel(config)
    
    # 4. Save config (for later loading)
    if rank == 0:
        config.save_pretrained(cfg.exp_dir)
    
    # 5. Create data module
    data_module = MyTaskDatamodule(cfg.data)
    
    # 6. Create trainer and start training
    trainer = MyTaskTrainer(
        cfg,
        model,
        data_module,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size
    )
    trainer.run()
    
    # 7. Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

#### Step 8: Configuration File

```yaml
# examples/mytask/configs/train.yaml
exp_dir: ./exp/mytask_exp

hydra:
  run:
    dir: ${exp_dir}/logs/${now:%Y-%m-%d}/${now:%H-%M-%S}

model:
  model_type: mytask  # or existing type like "asr"
  encoder:
    model_type: zipformer
    pretrained_model: path/to/pretrained/encoder  # optional

trainer:
  optimizer:
    type: adamw
    lr: 1e-4
    weight_decay: 0.01
  
  scheduler:
    type: cosine
    t_max: 100
  
  num_epochs: 30
  start_epoch: 1
  num_steps: 100000
  start_batch: 0
  
  keep_last_k: 30
  use_averaged_model: True
  average_period: 200
  
  log_interval: 50
  valid_interval: 1000
  save_every_n: 4
  reset_interval: 200
  
  use_fp16: True
  tensorboard: True

data:
  train_data_config: configs/data_configs/train_data_config.yaml
  valid_data_config: configs/data_configs/valid_data_config.yaml
  
  on_the_fly_feats: True
  feature: fbank  # or whisper_fbank, wav
  sampling_rate: 16000
  num_workers: 8
  
  use_infinite_dataset: True
  
  data_augmentation:
    enable_spec_aug: True
    enable_musan: False
    enable_speed_perturb: False
  
  sampler:
    type: bucketing_sampler
    num_buckets: 30
    max_duration: 600
    shuffle: True
    drop_last: True
```

#### Step 9: Run Training

```bash
# Single GPU
python train.py

# Multi-GPU (8 cards)
torchrun --nproc_per_node=8 train.py

# Override config
python train.py exp_dir=./my_exp trainer.lr=1e-3
```

---

## Best Practices

### 1. **Configuration Management**

**Hydra Best Practices**:
```yaml
# Use variables to avoid repetition
exp_dir: ./exp/my_exp
output_dir: ${exp_dir}/outputs

# Use defaults to manage multiple configs
defaults:
  - data: librispeech
  - model: zipformer_base
  - optimizer: adamw

# Override in command line
# python train.py model=zipformer_large data.num_workers=16
```

### 2. **Model Weight Loading**

**Separate Pretrained Encoder and Task Head**:
```python
# Load pretrained encoder
encoder_config = AutoConfig.from_pretrained("path/to/encoder")
pretrained_encoder = AutoModel.from_pretrained("path/to/encoder")

# Create task model
config = AutoConfig.for_model("asr", encoder_config=encoder_config)
model = AutoModel.from_config(config)

# Load encoder weights
model.encoder.load_state_dict(pretrained_encoder.state_dict(), strict=True)

# Optional: Freeze encoder
for param in model.encoder.parameters():
    param.requires_grad = False
```

### 3. **Checkpoint Management**

**Training Checkpoint Strategy**:
```yaml
trainer:
  # Keep last 30 checkpoints
  keep_last_k: 30
  
  # Validate every 1000 steps
  valid_interval: 1000
  
  # Save checkpoint every 4 validations
  save_every_n: 4
  
  # Use model averaging
  use_averaged_model: True
  average_period: 200
```

**Resume Training from Checkpoint**:
```yaml
trainer:
  # Resume from specific step
  start_batch: 10000  # will load checkpoint-10000.pt
  
  # Or resume from specific epoch
  start_epoch: 5  # will load epoch-4.pt
```

**Export Model for Inference**:
```python
from auden.utils.checkpoint import generate_model_checkpoint_from_trainer_checkpoints

# Average last 5 checkpoints and export
generate_model_checkpoint_from_trainer_checkpoints(
    model_dir="./exp_dir",
    iters=10000,      # Start from iter=10000
    avg=5,            # Average last 5
    model_name="final_model.pt"
)

# Then load with AutoModel
model = AutoModel.from_pretrained("./exp_dir")
```

### 4. **Data Processing**

**Lhotse Data Pipeline**:
```python
# 1. Create manifest
from lhotse import CutSet, RecordingSet, SupervisionSet

recordings = RecordingSet.from_dir("audio_dir", pattern="*.wav")
supervisions = SupervisionSet.from_segments(segments)
cuts = CutSet.from_manifests(recordings, supervisions)

# 2. Save
cuts.to_file("train_cuts.jsonl.gz")

# 3. Use in data_module
cutset = CutSet.from_file("train_cuts.jsonl.gz")
```

**Data Configuration File**:
```yaml
# configs/data_configs/train_data_config.yaml
- manifest: /path/to/dataset1.jsonl.gz
  hours: 100
  weights: 1
  lang: en

- manifest: /path/to/dataset2.jsonl.gz
  hours: 500
  weights: 2
  lang: zh
```

### 5. **Debugging Tips**

**Quick Test Configuration**:
```yaml
trainer:
  num_epochs: 1
  num_steps: 100
  log_interval: 10
  valid_interval: 50

data:
  num_workers: 0  # Single process for debugging
  sampler:
    max_duration: 100  # Small batch
```

**Use Single GPU for Debugging**:
```bash
# Don't use torchrun, run directly
python train.py

# Set environment variable for verbose logging
LOGLEVEL=DEBUG python train.py
```

### 6. **Performance Optimization**

**Mixed Precision Training**:
```yaml
trainer:
  use_fp16: True  # Enable FP16
```

**Data Loading Optimization**:
```yaml
data:
  num_workers: 8          # Adjust based on CPU cores
  sampler:
    num_buckets: 30       # More buckets = more uniform batches
    max_duration: 600     # Adjust based on GPU memory
    shuffle: True
```

**Gradient Accumulation** (if larger effective batch size needed):
```python
# Modify _forward_backward_optimize in trainer
class MyTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accumulation_steps = 4
    
    def train_one_epoch(self, epoch):
        # Modify training loop to implement gradient accumulation
        pass
```

### 7. **Distributed Training**

**Launch Multi-GPU Training**:
```bash
# Single machine, multiple GPUs
torchrun --nproc_per_node=8 train.py

# Multiple machines, multiple GPUs
# Node 0
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=29500 \
    train.py

# Node 1
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.1" \
    --master_port=29500 \
    train.py
```

**DDP Considerations**:
- Only save checkpoints and logs on rank 0
- Use `dist.barrier()` to synchronize processes
- Need to reduce metrics during validation: `metrics.reduce(device=device)`

---

## FAQ

### Q1: How to add custom optimizer/scheduler?

**Method 1: Override in trainer**
```python
class MyTrainer(BaseTrainer):
    def build_optimizer(self, model):
        # Custom optimizer
        return torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    def build_scheduler(self, optimizer):
        # Custom scheduler
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

**Method 2: Extend BaseTrainer's config support**
```python
# Add new type in cfg.trainer.optimizer
# Then add corresponding branch in BaseTrainer.build_optimizer
```

### Q2: How to handle variable-length sequences?

Lhotse's `DynamicBucketingSampler` automatically groups samples of similar length into the same batch and automatically pads. If you need custom padding:

```python
# In data_module
def _filter_cutset(self, cutset, split="train"):
    if self.cfg.get("pad_to_30s", False):
        cutset = cutset.pad(duration=30.0)
    return cutset
```

### Q3: How to implement custom data augmentation?

```python
# In subclass of BaseLhotseDatamodule
def _build_data_augmentation(self):
    transforms, input_transforms = super()._build_data_augmentation()
    
    # Add custom transform
    from lhotse.dataset import SomeCustomTransform
    transforms.append(SomeCustomTransform(...))
    
    return transforms, input_transforms
```

### Q4: How to export model for inference?

```python
# Method 1: Average and export from training checkpoints
from auden.utils.checkpoint import generate_model_checkpoint_from_trainer_checkpoints

generate_model_checkpoint_from_trainer_checkpoints(
    model_dir="./exp_dir",
    iters=10000,
    avg=5,
    model_name="averaged_model.pt"
)

# Method 2: Save model directly
model = AutoModel.from_config(config)
# ... load weights ...
model.save_pretrained("./export_dir")

# Load for inference
model = AutoModel.from_pretrained("./export_dir")
model.eval()
```

### Q5: How to compute specific metrics (e.g., WER) during validation?

```python
class MyTrainer(BaseTrainer):
    def validate(self, epoch: int):
        """Override validation method to compute specific metrics"""
        self.model.eval()
        
        with torch.no_grad():
            for valid_name, valid_dl in zip(
                self.data_module.valid_names,
                self.data_module.valid_dls
            ):
                # Collect predictions and references
                all_preds = []
                all_refs = []
                
                for batch in valid_dl:
                    # Generate predictions
                    preds = self.model.generate(batch["inputs"])
                    refs = batch["texts"]
                    
                    all_preds.extend(preds)
                    all_refs.extend(refs)
                
                # Compute metrics (e.g., WER)
                from kaldialign import edit_distance
                total_errors = 0
                total_words = 0
                
                for pred, ref in zip(all_preds, all_refs):
                    err = edit_distance(ref.split(), pred.split())
                    total_errors += err
                    total_words += len(ref.split())
                
                wer = total_errors / total_words
                
                # Log
                if self.rank == 0:
                    logging.info(f"Validation {valid_name} WER: {wer:.2%}")
                    if self.tb_writer:
                        self.tb_writer.add_scalar(
                            f"valid/{valid_name}_wer",
                            wer,
                            self.global_step
                        )
        
        self.model.train()
```

---

## Summary

Auden's design philosophy is **simple, modular, and extensible**:

1. **Use Auto System** to simplify model loading
2. **Inherit Base Classes** to quickly implement new tasks
3. **Based on Lhotse** for unified data processing
4. **Use Hydra** to manage configurations
5. **Separate Encoder and Task Head** for easier transfer learning
6. **Distinguish Training and Model Checkpoints** for different needs

By examining the examples in the `examples/` directory (ASR, CLAP, Audio Caption, etc.), you can learn how to combine these components to implement your own audio understanding tasks.

**Reference Examples**:
- `examples/asr/` - Complete ASR training pipeline
- `examples/clap/` - Contrastive learning example
- `examples/audio_caption/` - Sequence generation example

**Further Reading**:
- Lhotse documentation: https://lhotse.readthedocs.io/
- Hydra documentation: https://hydra.cc/
- K2 documentation: https://k2-fsa.github.io/k2/

Happy coding! If you have any questions, please feel free to open an issue or check the source code comments for more details.
