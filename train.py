from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.utils.audio import AudioProcessor
from trainer import Trainer, TrainerArgs
dataset_config = BaseDatasetConfig(
    formatter="ljspeech",
    meta_file_train="train.csv",
    meta_file_val="val.csv",
    path="/workspace/data/processed/",
    language="ml"
)
audio_config = VitsAudioConfig(
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None
)
config = VitsConfig(
    audio=audio_config,
    run_name="malayalam_vits",
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="multilingual_cleaners",
    use_phonemes=False,
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=True,
    mixed_precision=True,
    output_path="/workspace/output/",
    datasets=[dataset_config]
)
ap = AudioProcessor.init_from_config(config)
model = Vits.init_from_config(config)
from TTS.tts.datasets import load_tts_samples
train_samples, eval_samples = load_tts_samples(config.datasets, eval_split=True)
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path="/workspace/output/",
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples
)
trainer.fit()
