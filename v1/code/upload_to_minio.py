import os
from minio import Minio
from minio.error import S3Error
import shutil
from typing import List, Union
import tempfile


class MinioVersionManager:
    def __init__(self, minio_endpoint: str, access_key: str, secret_key: str, secure: bool = False):
        self.minio_client = Minio(
            minio_endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self.bucket_name = input("Enter bucket name: ")
        self.ensure_bucket_exists()

    def ensure_bucket_exists(self) -> None:
        try:
            if not self.minio_client.bucket_exists(self.bucket_name):
                self.minio_client.make_bucket(self.bucket_name)
                print(f"Bucket '{self.bucket_name}' created successfully")
            else:
                print(f"Using existing bucket: '{self.bucket_name}'")
        except S3Error as e:
            print(f"Error: {e}")
            raise

    def get_next_version(self) -> int:
        try:
            objects = self.minio_client.list_objects(self.bucket_name)
            versions = []
            for obj in objects:
                try:
                    version = int(obj.object_name.split('/')[0])
                    versions.append(version)
                except (ValueError, IndexError):
                    continue
            return max(versions + [0]) + 1
        except S3Error as e:
            print(f"Error getting next version: {e}")
            return 1

    def normalize_path(self, path: str) -> str:
        """Convert Windows path separators to forward slashes for S3 compatibility."""
        return path.replace('\\', '/')

    def save_files(self, files_to_upload: List[Union[str, List[str]]]) -> int:
        # Use tempfile to create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                version = self.get_next_version()
                version_str = str(version)

                # Process and copy files to temporary directory
                for item in files_to_upload:
                    if isinstance(item, list):
                        # Handle list of files
                        for file_path in item:
                            if os.path.exists(file_path):
                                shutil.copy2(file_path, temp_dir)
                            else:
                                print(f"Warning: File {file_path} not found")
                    else:
                        # Handle single file or directory
                        if os.path.exists(item):
                            if os.path.isdir(item):
                                dst_dir = os.path.join(temp_dir, os.path.basename(item))
                                shutil.copytree(item, dst_dir)
                            else:
                                shutil.copy2(item, temp_dir)
                        else:
                            print(f"Warning: File/directory {item} not found")

                # Upload files to MinIO
                print(f"Uploading files to MinIO version {version}...")
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, temp_dir)
                        # Normalize the path for S3 compatibility
                        normalized_path = self.normalize_path(relative_path)
                        object_name = f"v{version_str}/{normalized_path}"

                        try:
                            self.minio_client.fput_object(
                                self.bucket_name,
                                object_name,
                                file_path
                            )
                            print(f"Uploaded: {object_name}")
                        except Exception as e:
                            print(f"Error uploading {file_path}: {e}")

                print(f"Successfully saved version {version}")
                return version

            except Exception as e:
                print(f"Error saving files: {e}")
                raise


def main():
    # Setup MinIO manager
    minio_manager = MinioVersionManager(
        minio_endpoint="192.168.188.201:9000",
        access_key="minio",
        secret_key="minio123"
    )

    # Files to upload
    savemodel_dir = "model_output/saved_model"  # the name of folder to save the model
    code_file_dir = ["model.py", "upload_to_minio.py"]  # the names of code file

    files_to_upload = [savemodel_dir, code_file_dir]

    try:
        version = minio_manager.save_files(files_to_upload)
        print(f"Saved as version {version}")
    except Exception as e:
        print(f"Failed to upload files: {e}")


if __name__ == "__main__":
    main()
