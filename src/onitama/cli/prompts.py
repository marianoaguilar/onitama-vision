from __future__ import annotations


def prompt_int(prompt: str, default: int, lo: int, hi: int) -> int:
    while True:
        raw = input(f"\n{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a valid integer.")
            continue

        if value < lo or value > hi:
            print(f"Please enter a number between {lo} and {hi}.")
            continue

        return value


def prompt_choice(prompt: str, options: list[str], default_index: int = 0) -> str:
    assert options, "Options must not be empty."
    while True:
        print(prompt)
        for index, option in enumerate(options, start=1):
            mark = " (default)" if (index - 1) == default_index else ""
            print(f"  {index}) {option}{mark}")

        raw = input("Select an option: ").strip()
        if raw == "":
            return options[default_index]

        try:
            index = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue

        if index < 1 or index > len(options):
            print(f"Please enter a number between 1 and {len(options)}.")
            continue

        return options[index - 1]
