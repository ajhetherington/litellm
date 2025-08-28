import asyncio
from typing import Any, Coroutine, Optional, Union

import httpx
import json
import io

from litellm import LlmProviders
from litellm.integrations.gcs_bucket.gcs_bucket_base import (
    GCSBucketBase,
    GCSLoggingConfig,
)
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.openai import CreateFileRequest, OpenAIFileObject, FileContentRequest
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
from litellm.llms.vertex_ai.common_utils import VertexAIError
from litellm.types.llms.vertex_ai import GcsBucketResponse
from litellm.llms.vertex_ai.common_utils import (
    _convert_vertex_datetime_to_openai_datetime,
)

from openai.types.file_deleted import FileDeleted
from openai.types.file_object import FileObject


from .transformation import VertexAIJsonlFilesTransformation, VertexAIFilesConfig, LiteLLMLoggingObj

vertex_ai_files_transformation = VertexAIJsonlFilesTransformation()


class VertexAIFilesHandler(GCSBucketBase):
    """
    Handles Calling VertexAI in OpenAI Files API format v1/files/*

    This implementation uploads files on GCS Buckets
    """

    def __init__(self):
        super().__init__()
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=LlmProviders.VERTEX_AI,
        )

    async def async_create_file(
        self,
        create_file_data: CreateFileRequest,
        api_base: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> OpenAIFileObject:
        gcs_logging_config: GCSLoggingConfig = await self.get_gcs_logging_config(
            kwargs={}
        )
        headers = await self.construct_request_headers(
            vertex_instance=gcs_logging_config["vertex_instance"],
            service_account_json=gcs_logging_config["path_service_account"],
        )
        bucket_name = gcs_logging_config["bucket_name"]
        (
            logging_payload,
            object_name,
        ) = vertex_ai_files_transformation.transform_openai_file_content_to_vertex_ai_file_content(
            openai_file_content=create_file_data.get("file")
        )
        gcs_upload_response = await self._log_json_data_on_gcs(
            headers=headers,
            bucket_name=bucket_name,
            object_name=object_name,
            logging_payload=logging_payload,
        )

        return vertex_ai_files_transformation.transform_gcs_bucket_response_to_openai_file_object(
            purpose=create_file_data.get("purpose"),
            gcs_upload_response=gcs_upload_response,
        )

    def create_file(
        self,
        _is_async: bool,
        create_file_data: CreateFileRequest,
        api_base: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> Union[OpenAIFileObject, Coroutine[Any, Any, OpenAIFileObject]]:
        """
        Creates a file on VertexAI GCS Bucket

        Only supported for Async litellm.acreate_file
        """

        if _is_async:
            return self.async_create_file(
                create_file_data=create_file_data,
                api_base=api_base,
                vertex_credentials=vertex_credentials,
                vertex_project=vertex_project,
                vertex_location=vertex_location,
                timeout=timeout,
                max_retries=max_retries,
            )
        else:
            return asyncio.run(
                self.async_create_file(
                    create_file_data=create_file_data,
                    api_base=api_base,
                    vertex_credentials=vertex_credentials,
                    vertex_project=vertex_project,
                    vertex_location=vertex_location,
                    timeout=timeout,
                    max_retries=max_retries,
                )
            )
    
    async def async_retrieve_file(
        self,
        file_id: str,
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ):
        # https://cloud.google.com/storage/docs/json_api/v1/objects
        # copied parts from transform_create_file_response
        raw_response = await self._get_gcs_object(file_id, return_content=False)
        if not raw_response:
            msg = "Failed to retrieve gcs object"
            raise Exception(msg)

        try:
            response_object = GcsBucketResponse(**raw_response.json())  # type: ignore
        except Exception as e:
            raise VertexAIError(
                status_code=raw_response.status_code,
                message=f"Error reading GCS bucket response: {e}",
                headers=raw_response.headers,
            )
        return vertex_ai_files_transformation.transform_gcs_bucket_response_to_openai_file_object(
            purpose="batch",
            gcs_upload_response=response_object, # type: ignore
        )

    def retrieve_file(
        self,
        _is_async: bool,
        file_id: str,
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> Union[OpenAIFileObject, Coroutine[Any, Any, OpenAIFileObject]]:
        if _is_async:
            return self.async_retrieve_file(
                file_id,
                vertex_credentials,
                vertex_project,
                vertex_location,
                timeout,
                max_retries,
            )
        else:
            return asyncio.run(
                self.async_retrieve_file(
                    file_id,
                    vertex_credentials,
                    vertex_project,
                    vertex_location,
                    timeout,
                    max_retries
                )
            )
    
    async def async_file_content(
        self,
        file_content_request: FileContentRequest,
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        logging_obj: LiteLLMLoggingObj
    ):
        if "file_id" not in file_content_request:
            raise Exception("file_id must be provided")

        raw_response = await self.download_gcs_object(file_content_request["file_id"])
        if not raw_response:
            raise Exception("bytes not returned")
        
        openai_jsonl_content = vertex_ai_files_transformation.transform_vertex_ai_file_content_to_openai_file_content(raw_response, logging_obj)

        # write to in-memory buffer for response, cannot form a HttpxBinaryResponseContent
        buffer = io.BytesIO()

        for record in openai_jsonl_content:
            jsonl_line = json.dumps(record) + '\n'
            encoded_line = jsonl_line.encode('utf-8')
            buffer.write(encoded_line)

        return buffer.getvalue()

    def file_content(
        self,
        _is_async: bool,
        file_content_request: FileContentRequest,
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        logging_obj: LiteLLMLoggingObj
    ):
        if _is_async:
            return self.async_file_content(
                file_content_request,
                vertex_credentials,
                vertex_project,
                vertex_location,
                timeout,
                max_retries,
                logging_obj,
            )
        else:
            return asyncio.run(
                self.async_file_content(
                    file_content_request,
                    vertex_credentials,
                    vertex_project,
                    vertex_location,
                    timeout,
                    max_retries,
                    logging_obj,
                )
            )

    async def async_delete_file(
        self,
        file_id: str,
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> FileDeleted:
        ret = await self.delete_gcs_object(file_id)
        return FileDeleted(
            id=file_id,
            deleted=True,
            object="file"
        )

    def delete_file(
        self,
        _is_async: bool,
        file_id: str,
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> Union[FileDeleted, Coroutine[Any, Any, FileDeleted]]:
        if _is_async:
            return self.async_delete_file(
                file_id,
                vertex_credentials,
                vertex_project,
                vertex_location,
                timeout,
                max_retries,
            )
        else:
            return asyncio.run(self.async_delete_file(
                file_id,
                vertex_credentials,
                vertex_project,
                vertex_location,
                timeout,
                max_retries,
            ))

