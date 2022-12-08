import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from zerospot.asr.constants import ASR_FILE_SUFFIX
from zerospot.asr.interface import ASRModelInterface


def parse_args() -> argparse.Namespace:
    """
    Parse arguments for exporting.
    Returns: argparse.Namespace.
    """
    arguments_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arguments_parser.add_argument(
        "-src",
        "--src_dir",
        help="Source directory with .wav (or .mp3) files in it",
        type=Path,
        required=True,
    )
    arguments_parser.add_argument(
        "-d",
        "--device",
        help="Device (torch.device) for computation.",
        type=str,
        required=False,
        default="cpu",
    )
    arguments_parser.add_argument(
        "-f",
        "--force_recompute",
        help="Whether to force recompute transcriptions or not",
        action="store_true",
        default=False,
    )
    arguments_parser.add_argument(
        "-s",
        "--suffix",
        help="Suffix of audio files (.wav or .mp3)",
        type=str,
        required=False,
        default=".wav",
        choices=[".wav", ".mp3"],
    )
    return arguments_parser.parse_args()


def main(src_dir: Path, device: str, suffix: str, force_recompute: bool) -> None:
    # Setup logger
    logging.basicConfig(level=logging.getLevelName("INFO"))
    logger = logging.getLogger(Path(__file__).name)

    # Collect audios
    audios = list(src_dir.rglob(f"*{suffix}"))

    if not force_recompute:
        audios = [audio_path for audio_path in audios if not audio_path.with_suffix(ASR_FILE_SUFFIX).exists()]

    logger.info(f"N. of audios to compute transcriptions: {len(audios)}")

    logger.info(f"Initializing ASR model interface...")
    interface = ASRModelInterface(device=device)

    logger.info(f"ASR model inference...")
    transcriptions = interface.get_transcription(audios)

    for transcription, audio_path in tqdm(zip(transcriptions, audios), desc="Saving transcription files..."):
        save_path = audio_path.with_suffix(ASR_FILE_SUFFIX)
        transcription.to_json(save_path)

    logger.info("Computing transcription files is done.")


def run():
    args = parse_args()
    main(**vars(args))


if __name__ == "__main__":
    run()
