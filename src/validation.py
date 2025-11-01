"""
Input validation and error handling utilities.
"""

import os
import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

# Ensure src directory is in path for absolute imports
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import core dependencies
try:
    from .logger import get_logger
except ImportError:
    from logger import get_logger

logger = get_logger("validation")


class RAGError(Exception):
    """Unified error class for the RAG system."""

    def __init__(self,
                 message: str,
                 error_code: str,
                 error_type: str = "error",
                 field: Optional[str] = None,
                 value: Any = None,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.error_type = error_type
        self.field = field
        self.value = value
        self.context = context or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to standardized dictionary format."""
        return {
            "error": self.error_type,
            "code": self.error_code,
            "message": self.message,
            "field": self.field,
            "value": str(self.value) if self.value is not None else None,
            "context": self.context
        }


# Backward compatibility
ValidationError = RAGError


class InputValidator:
    """Comprehensive input validation utilities."""
    
    @staticmethod
    def validate_query(query: str) -> str:
        """Validate and sanitize user query.

        Args:
            query: User input query

        Returns:
            Sanitized query string

        Raises:
            RAGError: If query is invalid
        """
        # Type validation
        if not isinstance(query, str):
            raise RAGError("Query must be a string", "VALIDATION_INVALID_TYPE", "validation",
                          field="query", value=type(query))

        sanitized = query.strip()

        # Empty validation
        if not sanitized:
            raise RAGError("Query cannot be empty", "VALIDATION_EMPTY_QUERY", "validation",
                          field="query", value=query)

        # Length validation
        if len(sanitized) < 3:
            raise RAGError("Query too short (minimum 3 characters)", "VALIDATION_QUERY_TOO_SHORT", "validation",
                          field="query", value=len(sanitized))

        if len(sanitized) > 10000:  # Match test expectation
            raise RAGError("Query too long (max 10000 characters)", "VALIDATION_QUERY_TOO_LONG", "validation",
                          field="query", value=len(sanitized))

        logger.debug("Query validated", length=len(sanitized))
        return sanitized
    
    @staticmethod
    def _validate_path_common(path_input: Union[str, Path], path_type: str = "file", must_exist: bool = True) -> Path:
        """Common path validation logic for files and directories."""
        if not path_input:
            raise RAGError(f"{path_type.title()} path cannot be empty", "VALIDATION_INVALID_PATH", "validation")

        if not isinstance(path_input, (str, Path)):
            raise RAGError(f"{path_type.title()} path must be string or Path", "VALIDATION_INVALID_TYPE", "validation",
                          value=type(path_input))

        path = Path(path_input)

        # Security check - prevent directory traversal
        resolved_path = path.resolve()
        project_root = Path(__file__).parent.parent.resolve()

        # Ensure path is within project boundaries
        try:
            resolved_path.relative_to(project_root)
        except ValueError:
            raise RAGError(f"Invalid {path_type} path (outside allowed directory)", "VALIDATION_INVALID_PATH", "validation",
                          value=str(path))

        if must_exist:
            if not path.exists():
                raise RAGError(f"{path_type.title()} does not exist", "VALIDATION_FILE_NOT_FOUND", "validation",
                              value=str(path))

            if path_type == "file" and not path.is_file():
                raise RAGError("Path is not a file", "VALIDATION_INVALID_PATH", "validation",
                              value=str(path))
            elif path_type == "directory" and not path.is_dir():
                raise RAGError("Path is not a directory", "VALIDATION_INVALID_PATH", "validation",
                              value=str(path))

        return path

    @staticmethod
    def validate_file_path(file_path: str, must_exist: bool = True) -> Path:
        """Validate file path.

        Args:
            file_path: Path to validate
            must_exist: Whether file must exist

        Returns:
            Validated Path object

        Raises:
            RAGError: If path is invalid
        """
        return InputValidator._validate_path_common(file_path, "file", must_exist)
    
    @staticmethod
    def validate_directory_path(dir_path: str, must_exist: bool = True) -> Path:
        """Validate directory path.

        Args:
            dir_path: Directory path to validate
            must_exist: Whether directory must exist

        Returns:
            Validated Path object

        Raises:
            RAGError: If path is invalid
        """
        return InputValidator._validate_path_common(dir_path, "directory", must_exist)
    
    @staticmethod
    def validate_positive_int(value: Any, field_name: str, min_val: int = 1, max_val: Optional[int] = None) -> int:
        """Validate positive integer.

        Args:
            value: Value to validate
            field_name: Name of the field for error messages
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Validated integer

        Raises:
            RAGError: If value is invalid
        """
        try:
            value = int(value)
        except (ValueError, TypeError):
            raise RAGError(f"{field_name} must be an integer", "VALIDATION_INVALID_TYPE", "validation",
                          field=field_name, value=value)

        if value < min_val:
            raise RAGError(f"{field_name} must be >= {min_val}", "VALIDATION_INVALID_TYPE", "validation",
                          field=field_name, value=value)

        if max_val is not None and value > max_val:
            raise RAGError(f"{field_name} must be <= {max_val}", "VALIDATION_INVALID_TYPE", "validation",
                          field=field_name, value=value)

        return value

    @staticmethod
    def validate_float_range(value: Any, field_name: str, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Validate float within range.

        Args:
            value: Value to validate
            field_name: Name of the field
            min_val: Minimum value
            max_val: Maximum value

        Returns:
            Validated float

        Raises:
            RAGError: If value is invalid
        """
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise RAGError(f"{field_name} must be a number", "VALIDATION_INVALID_TYPE", "validation",
                          field=field_name, value=value)

        if value < min_val or value > max_val:
            raise RAGError(f"{field_name} must be between {min_val} and {max_val}", "VALIDATION_INVALID_TYPE", "validation",
                          field=field_name, value=value)

        return value
    
    @staticmethod
    def validate_model_name(model_name: str) -> str:
        """Validate model name format.

        Args:
            model_name: Model name to validate

        Returns:
            Validated model name

        Raises:
            RAGError: If model name is invalid
        """
        if not model_name:
            raise RAGError("Model name cannot be empty", "VALIDATION_INVALID_TYPE", "validation",
                          field="model_name", value=model_name)

        if not isinstance(model_name, str):
            raise RAGError("Model name must be a string", "VALIDATION_INVALID_TYPE", "validation",
                          field="model_name", value=type(model_name))

        model_name = model_name.strip()

        if not model_name:
            raise RAGError("Model name cannot be empty", "VALIDATION_INVALID_TYPE", "validation",
                          field="model_name", value=model_name)

        if not re.match(r'^[a-zA-Z0-9\-_.]+$', model_name):
            raise RAGError("Invalid model name format", "VALIDATION_INVALID_TYPE", "validation",
                          field="model_name", value=model_name)

        if len(model_name) > 100:
            raise RAGError("Model name too long", "VALIDATION_INVALID_TYPE", "validation",
                          field="model_name", value=len(model_name))

        return model_name
    
    @staticmethod
    def validate_retriever_type(retriever_type: str) -> str:
        """Validate retriever type.

        Args:
            retriever_type: Type of retriever

        Returns:
            Validated retriever type

        Raises:
            RAGError: If type is invalid
        """
        valid_types = ["tfidf", "bm25", "embeddings", "hybrid"]

        if not retriever_type or not isinstance(retriever_type, str):
            raise RAGError("Retriever type must be a non-empty string", "VALIDATION_INVALID_TYPE", "validation",
                          field="retriever_type", value=retriever_type)

        retriever_type = retriever_type.lower().strip()

        if retriever_type not in valid_types:
            raise RAGError(f"Invalid retriever type. Must be one of: {valid_types}", "VALIDATION_INVALID_TYPE", "validation",
                          field="retriever_type", value=retriever_type)

        return retriever_type
    
    @staticmethod
    def validate_api_key(api_key: str, service_name: str = "API") -> str:
        """Validate API key format.
        
        Args:
            api_key: API key to validate
            service_name: Name of the service for error messages
            
        Returns:
            Validated API key
            
        Raises:
            ValidationError: If API key is invalid
        """
        if not api_key:
            raise ValidationError(f"{service_name} key cannot be empty")
        
        if not isinstance(api_key, str):
            raise ValidationError(f"{service_name} key must be string", value=type(api_key))
        
        api_key = api_key.strip()
        
        # Basic length check (most API keys are at least 20 characters)
        if len(api_key) < 20:
            raise ValidationError(f"{service_name} key too short", value=len(api_key))
        
        # Check for obvious placeholder values
        placeholder_patterns = [
            "your_api_key", "api_key_here", "insert_key", "replace_me",
            "xxxx", "****", "placeholder", "example"
        ]
        
        api_key_lower = api_key.lower()
        for pattern in placeholder_patterns:
            if pattern in api_key_lower:
                raise ValidationError(f"API key appears to be a placeholder", value=pattern)
        
        return api_key


class ErrorHandler:
    """Unified error handling and logging system."""

    # Standard error codes
    ERROR_CODES = {
        "VALIDATION_EMPTY_QUERY": "Empty query provided",
        "VALIDATION_INVALID_TYPE": "Invalid data type",
        "VALIDATION_QUERY_TOO_LONG": "Query exceeds maximum length",
        "VALIDATION_QUERY_TOO_SHORT": "Query is too short",
        "VALIDATION_FILE_NOT_FOUND": "File does not exist",
        "VALIDATION_INVALID_PATH": "Invalid file path",
        "VALIDATION_API_KEY_INVALID": "Invalid API key",
        "API_CONNECTION_FAILED": "Failed to connect to API",
        "API_RATE_LIMITED": "API rate limit exceeded",
        "API_AUTHENTICATION_FAILED": "API authentication failed",
        "SYSTEM_RESOURCE_ERROR": "System resource unavailable",
        "SYSTEM_CONFIGURATION_ERROR": "Configuration error",
        "SYSTEM_INTERNAL_ERROR": "Internal system error"
    }

    @staticmethod
    def handle_validation_error(error: ValidationError, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle validation errors with proper logging (backward compatibility).

        Args:
            error: ValidationError instance
            context: Additional context for logging

        Returns:
            Error response dictionary with keys: error_type, message, field, value
        """
        error_dict = {
            "error_type": "validation_error",
            "message": error.message if hasattr(error, 'message') else str(error),
            "field": error.field if hasattr(error, 'field') else None,
            "value": str(error.value) if hasattr(error, 'value') and error.value is not None else None
        }
        if context:
            error_dict.update(context)

        logger.warning("Validation error", **error_dict)
        return error_dict
    
    @staticmethod
    def handle_error(error: Exception,
                    operation: Optional[str] = None,
                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle any error with standardized formatting.

        Args:
            error: Exception instance
            operation: Name of the operation that failed
            context: Additional context

        Returns:
            Standardized error response dictionary
        """
        if isinstance(error, RAGError):
            error_dict = error.to_dict()
            if context:
                error_dict["context"].update(context)
            if operation:
                error_dict["operation"] = operation

            log_level = "warning" if error.error_type == "validation" else "error"
            logger.log(getattr(logger, log_level), f"{error.error_type.title()} error",
                      exception=error, **error_dict)
            return error_dict

        # Handle non-RAG errors
        error_dict = {
            "error": "system_error",
            "code": "SYSTEM_INTERNAL_ERROR",
            "message": ErrorHandler.ERROR_CODES.get("SYSTEM_INTERNAL_ERROR", str(error)),
            "operation": operation,
            "exception_type": type(error).__name__,
            "context": context or {}
        }

        logger.error("System error occurred", exception=error, **error_dict)
        return error_dict

    @staticmethod
    def handle_api_error(error: Exception, operation: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle API errors (backward compatibility).

        Returns:
            Dictionary with keys: error_type, operation, message, exception_type
        """
        error_dict = {
            "error_type": "api_error",
            "operation": operation,
            "message": str(error),
            "exception_type": type(error).__name__
        }
        if context:
            error_dict.update(context)

        logger.error("API error", exception=error, **error_dict)
        return error_dict
    
    @staticmethod
    def handle_file_error(error: Exception, file_path: str, operation: str) -> Dict[str, Any]:
        """Handle file operation errors (backward compatibility)."""
        context = {"file_path": file_path}
        return ErrorHandler.handle_error(error, operation, context)

    @staticmethod
    def create_safe_error_response(error: Exception,
                                  hide_details: bool = True,
                                  operation: Optional[str] = None) -> Dict[str, Any]:
        """Create a safe error response that doesn't leak sensitive information.

        Args:
            error: Exception instance
            hide_details: Whether to hide detailed error information
            operation: Operation that failed

        Returns:
            Safe error response dictionary
        """
        if isinstance(error, RAGError) and error.error_type == "validation":
            return {
                "error": error.message,
                "message": error.message,
                "field": error.field
            }

        if hide_details:
            return {
                "error": "Internal error",
                "message": "An error occurred while processing your request"
            }
        else:
            return {
                "error": type(error).__name__,
                "message": str(error),
                "operation": operation
            }

    @staticmethod
    def create_error(error_code: str,
                    message: Optional[str] = None,
                    error_type: str = "error",
                    **kwargs) -> RAGError:
        """Create a standardized RAG error.

        Args:
            error_code: Error code from ERROR_CODES
            message: Custom error message (uses default if None)
            error_type: Type of error
            **kwargs: Additional error parameters

        Returns:
            RAGError instance
        """
        default_message = ErrorHandler.ERROR_CODES.get(error_code, "Unknown error")
        return RAGError(
            message=message or default_message,
            error_code=error_code,
            error_type=error_type,
            **kwargs
        )


def validate_rag_query_params(query: str, top_k: int, retriever_type: str) -> Dict[str, Any]:
    """Validate parameters for RAG query.
    
    Args:
        query: User query
        top_k: Number of results to retrieve
        retriever_type: Type of retriever to use
        
    Returns:
        Dictionary of validated parameters
        
    Raises:
        ValidationError: If any parameter is invalid
    """
    try:
        validated_params = {
            "query": InputValidator.validate_query(query),
            "top_k": InputValidator.validate_positive_int(top_k, "top_k", min_val=1, max_val=50),
            "retriever_type": InputValidator.validate_retriever_type(retriever_type)
        }
        
        logger.debug("Query parameters validated", **validated_params)
        return validated_params
        
    except ValidationError as e:
        logger.warning("Query parameter validation failed", error=str(e))
        raise