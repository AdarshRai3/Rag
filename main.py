from vector_storage_service import VectorStorageService
import logging

# Setup logging
logger = logging.getLogger("main")

if __name__ == '__main__':
    service = VectorStorageService()

    # Uncomment to index from scratch:
    # service.store_from_json('./data/raw_data.json')

    # Load (or rebuild if needed) and use index
    index = service.load_index(
        persist_dir='./retrieval-engine-storage',
        json_path='./data/questions.json'
    )
    logger.info('Index ready for queries.')