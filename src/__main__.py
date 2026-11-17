import sys
import traceback

if __name__ == "__main__":
    try:
        from fire import Fire
        from .cli import CLI
        Fire(component=CLI, name="rag")
    except KeyboardInterrupt:
        print("Exiting...")
        sys.exit(1)
    except Exception as e:
        traceback.print_exc()
        print(f"{__file__}#{e.__traceback__.tb_lineno}:({e.__class__.__name__}){e}")
        sys.exit(1)
