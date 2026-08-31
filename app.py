import asyncio

from backend.agent import achat


async def main():
    print("Xyberos AI is running.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            break

        if not user_input:
            continue

        try:
            response = await achat(user_input)
            print(f"AI: {response}\n")
        except Exception as exc:
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    asyncio.run(main())