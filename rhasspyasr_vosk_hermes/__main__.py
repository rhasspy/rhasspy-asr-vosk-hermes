"""Command-line interface to rhasspyasr-vosk-hermes"""
import argparse
import asyncio
import logging
from pathlib import Path

import paho.mqtt.client as mqtt
import vosk

import rhasspyhermes.cli as hermes_cli

from . import AsrHermesMqtt

_LOGGER = logging.getLogger("rhasspyasr_vosk_hermes")

# -----------------------------------------------------------------------------


def main():
    """Main method."""
    args = get_args()

    hermes_cli.setup_logging(args)
    _LOGGER.debug(args)

    run_mqtt(args)


# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog="rhasspy-asr-vosk-hermes")

    # Model settings
    parser.add_argument("--model", required=True, help="Path to the model directory")

    parser.add_argument("--words-json", help="Path to JSON file with word list(s)")

    parser.add_argument(
        "--no-overwrite-train",
        action="store_true",
        help="Don't overwrite words JSON during training",
    )

    parser.add_argument("--lang", help="Set lang in outgoing messages")

    hermes_cli.add_hermes_args(parser)

    return parser.parse_args()


# -----------------------------------------------------------------------------


def run_mqtt(args: argparse.Namespace):
    """Runs Hermes ASR MQTT service."""
    # Convert to Paths
    args.model = Path(args.model)

    if args.words_json:
        args.words_json = Path(args.words_json)

    if args.model.is_dir():
        # Load now
        _LOGGER.debug("Loading model from %s", args.model)
        model = vosk.Model(str(args.model))
    else:
        # Load later
        model = args.model

    # Listen for messages
    client = mqtt.Client()
    hermes = AsrHermesMqtt(
        client,
        model=model,
        words_json_path=args.words_json,
        no_overwrite_train=args.no_overwrite_train,
        site_ids=args.site_id,
        lang=args.lang,
    )

    _LOGGER.debug("Connecting to %s:%s", args.host, args.port)
    hermes_cli.connect(client, args)
    client.loop_start()

    try:
        # Run event loop
        asyncio.run(hermes.handle_messages_async())
    except KeyboardInterrupt:
        pass
    finally:
        _LOGGER.debug("Shutting down")
        client.loop_stop()


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
