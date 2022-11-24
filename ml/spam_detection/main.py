import sys
import modal

image = modal.Image.debian_slim().pip_install(
    [
        "loguru~=0.6.0",
    ]
)
stub = modal.Stub(name="example-spam-detect-llm", image=image)


def _get_logger():
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, colorize=True)
    return logger


@stub.function
def train():
    logger = _get_logger()

    print("\033[{}m> DocSearch: \033[0m{}\033[93m {} records\033[0m".format("96", "https://foo.com", 100))
    logger.opt(colors=True).info("Ready to detect <fg #9dc100><b>SPAM</b></fg #9dc100> from <fg #ffb6c1><b>HAM</b></fg #ffb6c1>?")


if __name__ == "__main__":
    with stub.run():
        train()
