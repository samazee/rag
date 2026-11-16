import sys

if __name__ == "__main__":
    try:
        from fire import Fire
        from .cli import CLI
        Fire(CLI, name="rag")
    except KeyboardInterrupt:
        print("Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"{e.__class__.__name__}: {e}")
        sys.exit(1)
