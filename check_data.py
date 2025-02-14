import asyncio
from data_collection import verify_datasets

async def main():
    await verify_datasets()

if __name__ == "__main__":
    asyncio.run(main()) 