#!/usr/bin/env python3
import argparse
from pathlib import Path

import sentencepiece as spm


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate bpe.vocab from a SentencePiece bpe.model.")
    parser.add_argument("--model", required=True, help="Path to bpe.model")
    parser.add_argument("--output", required=True, help="Path to generated bpe.vocab")
    args = parser.parse_args()

    model_path = Path(args.model)
    output_path = Path(args.output)
    processor = spm.SentencePieceProcessor(model_file=str(model_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Match the desktop-generated artifact checked by model-manifest.json.
    with output_path.open("w", encoding="utf-8", newline="\r\n") as handle:
        for idx in range(processor.GetPieceSize()):
            handle.write(f"{processor.IdToPiece(idx)}\t{processor.GetScore(idx)}\n")

    print(f"Generated {output_path} with {processor.GetPieceSize()} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
