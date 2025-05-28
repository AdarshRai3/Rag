from vector_storage_service import VectorStorageService
import logging

# Setup logging
logger = logging.getLogger("main")

if __name__ == '__main__':
    service = VectorStorageService()

    # To populate the index from JSON, uncomment:
    # service.store_from_json('./data/raw_data.json')

    # To load the existing index:
    index = service.load_index()
    logger.info('Index loaded successfully and ready for querying.')