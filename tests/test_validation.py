"""
Tests for input validation and error handling.
"""

import pytest
from pathlib import Path

from src.validation import (
    InputValidator, ValidationError, ErrorHandler, 
    validate_rag_query_params
)


class TestInputValidator:
    """Test input validation functions."""
    
    def test_validate_query_valid(self):
        """Test valid query validation."""
        query = "What is machine learning?"
        result = InputValidator.validate_query(query)
        assert result == query
    
    def test_validate_query_empty(self):
        """Test empty query validation."""
        with pytest.raises(ValidationError, match="Query cannot be empty"):
            InputValidator.validate_query("")
    
    def test_validate_query_whitespace_only(self):
        """Test whitespace-only query."""
        with pytest.raises(ValidationError, match="Query too short"):
            InputValidator.validate_query("   ")
    
    def test_validate_query_too_short(self):
        """Test query that's too short."""
        with pytest.raises(ValidationError, match="Query too short"):
            InputValidator.validate_query("hi")
    
    def test_validate_query_too_long(self):
        """Test query that's too long."""
        long_query = "x" * 10001
        with pytest.raises(ValidationError, match="Query too long"):
            InputValidator.validate_query(long_query)
    
    def test_validate_query_non_string(self):
        """Test non-string query."""
        with pytest.raises(ValidationError, match="Query must be a string"):
            InputValidator.validate_query(123)
    
    def test_validate_positive_int_valid(self):
        """Test valid positive integer."""
        result = InputValidator.validate_positive_int(5, "test_field")
        assert result == 5
    
    def test_validate_positive_int_string(self):
        """Test string that can be converted to int."""
        result = InputValidator.validate_positive_int("5", "test_field")
        assert result == 5
    
    def test_validate_positive_int_too_small(self):
        """Test integer below minimum."""
        with pytest.raises(ValidationError, match="test_field must be >= 1"):
            InputValidator.validate_positive_int(0, "test_field")
    
    def test_validate_positive_int_too_large(self):
        """Test integer above maximum."""
        with pytest.raises(ValidationError, match="test_field must be <= 10"):
            InputValidator.validate_positive_int(15, "test_field", max_val=10)
    
    def test_validate_positive_int_invalid(self):
        """Test invalid integer value."""
        with pytest.raises(ValidationError, match="test_field must be an integer"):
            InputValidator.validate_positive_int("invalid", "test_field")
    
    def test_validate_float_range_valid(self):
        """Test valid float range."""
        result = InputValidator.validate_float_range(0.5, "alpha", 0.0, 1.0)
        assert result == 0.5
    
    def test_validate_float_range_out_of_bounds(self):
        """Test float outside valid range."""
        with pytest.raises(ValidationError, match="alpha must be between 0.0 and 1.0"):
            InputValidator.validate_float_range(1.5, "alpha", 0.0, 1.0)
    
    def test_validate_model_name_valid(self):
        """Test valid model name."""
        result = InputValidator.validate_model_name("gpt-4o")
        assert result == "gpt-4o"
    
    def test_validate_model_name_empty(self):
        """Test empty model name."""
        with pytest.raises(ValidationError, match="Model name cannot be empty"):
            InputValidator.validate_model_name("")
    
    def test_validate_model_name_invalid_chars(self):
        """Test model name with invalid characters."""
        with pytest.raises(ValidationError, match="Invalid model name format"):
            InputValidator.validate_model_name("gpt@4o!")
    
    def test_validate_retriever_type_valid(self):
        """Test valid retriever types."""
        for retriever_type in ["tfidf", "bm25", "embeddings", "hybrid"]:
            result = InputValidator.validate_retriever_type(retriever_type)
            assert result == retriever_type
    
    def test_validate_retriever_type_case_insensitive(self):
        """Test case-insensitive retriever type validation."""
        result = InputValidator.validate_retriever_type("TFIDF")
        assert result == "tfidf"
    
    def test_validate_retriever_type_invalid(self):
        """Test invalid retriever type."""
        with pytest.raises(ValidationError, match="Invalid retriever type"):
            InputValidator.validate_retriever_type("invalid_type")
    
    def test_validate_api_key_valid(self):
        """Test valid API key."""
        api_key = "sk-" + "x" * 40
        result = InputValidator.validate_api_key(api_key, "OpenAI")
        assert result == api_key
    
    def test_validate_api_key_too_short(self):
        """Test API key that's too short."""
        with pytest.raises(ValidationError, match="OpenAI key too short"):
            InputValidator.validate_api_key("short", "OpenAI")
    
    def test_validate_api_key_placeholder(self):
        """Test API key that looks like a placeholder."""
        with pytest.raises(ValidationError, match="API key appears to be a placeholder"):
            InputValidator.validate_api_key("your_api_key_here_replace_me", "OpenAI")
    
    def test_validate_file_path_valid(self, temp_dir):
        """Test valid file path."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        
        result = InputValidator.validate_file_path(str(test_file))
        assert result == test_file
    
    def test_validate_file_path_not_exists(self, temp_dir):
        """Test file path that doesn't exist."""
        test_file = temp_dir / "nonexistent.txt"
        
        with pytest.raises(ValidationError, match="File does not exist"):
            InputValidator.validate_file_path(str(test_file))
    
    def test_validate_directory_path_valid(self, temp_dir):
        """Test valid directory path."""
        result = InputValidator.validate_directory_path(str(temp_dir))
        assert result == temp_dir


class TestErrorHandler:
    """Test error handling utilities."""
    
    def test_handle_validation_error(self):
        """Test validation error handling."""
        error = ValidationError("Test error", field="test_field", value="test_value")
        result = ErrorHandler.handle_validation_error(error)
        
        assert result["error_type"] == "validation_error"
        assert result["message"] == "Test error"
        assert result["field"] == "test_field"
        assert result["value"] == "test_value"
    
    def test_handle_api_error(self):
        """Test API error handling."""
        error = Exception("API connection failed")
        result = ErrorHandler.handle_api_error(error, "embedding_generation")
        
        assert result["error_type"] == "api_error"
        assert result["operation"] == "embedding_generation"
        assert result["message"] == "API connection failed"
        assert result["exception_type"] == "Exception"
    
    def test_create_safe_error_response_validation(self):
        """Test safe error response for validation errors."""
        error = ValidationError("Invalid input", field="query")
        result = ErrorHandler.create_safe_error_response(error)
        
        assert result["error"] == "Invalid input"
        assert result["message"] == "Invalid input"
        assert result["field"] == "query"
    
    def test_create_safe_error_response_generic_hidden(self):
        """Test safe error response with details hidden."""
        error = Exception("Internal database error")
        result = ErrorHandler.create_safe_error_response(error, hide_details=True)
        
        assert result["error"] == "Internal error"
        assert result["message"] == "An error occurred while processing your request"
    
    def test_create_safe_error_response_generic_shown(self):
        """Test safe error response with details shown."""
        error = Exception("Internal database error")
        result = ErrorHandler.create_safe_error_response(error, hide_details=False)
        
        assert result["error"] == "Exception"
        assert result["message"] == "Internal database error"


class TestRAGQueryParamsValidation:
    """Test RAG query parameter validation."""
    
    def test_validate_rag_query_params_valid(self):
        """Test valid RAG query parameters."""
        params = validate_rag_query_params(
            query="What is machine learning?",
            top_k=5,
            retriever_type="hybrid"
        )
        
        assert params["query"] == "What is machine learning?"
        assert params["top_k"] == 5
        assert params["retriever_type"] == "hybrid"
    
    def test_validate_rag_query_params_invalid_query(self):
        """Test invalid query parameter."""
        with pytest.raises(ValidationError):
            validate_rag_query_params(
                query="",
                top_k=5,
                retriever_type="hybrid"
            )
    
    def test_validate_rag_query_params_invalid_top_k(self):
        """Test invalid top_k parameter."""
        with pytest.raises(ValidationError):
            validate_rag_query_params(
                query="What is machine learning?",
                top_k=0,
                retriever_type="hybrid"
            )
    
    def test_validate_rag_query_params_invalid_retriever_type(self):
        """Test invalid retriever type parameter."""
        with pytest.raises(ValidationError):
            validate_rag_query_params(
                query="What is machine learning?",
                top_k=5,
                retriever_type="invalid_type"
            )