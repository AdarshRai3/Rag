from vector_storage_service import VectorStorageService

if __name__ == "__main__":
    service = VectorStorageService()
    service.store_from_json("data/raw_data.json")