import pytest
from unittest.mock import Mock, patch
import gradio as gr
import torch

# Import the functions to be tested
from app import respond, cancel_inference, clear_conversation, update_chat_info

@pytest.fixture
def mock_inference_client():
    with patch('app.InferenceClient') as mock:
        yield mock

@pytest.fixture
def mock_pipeline():
    with patch('app.pipeline') as mock:
        yield mock

def test_respond_api_inference(mock_inference_client):
    mock_client = Mock()
    mock_client.chat_completion.return_value = iter([
        Mock(choices=[Mock(delta=Mock(content="Hello"))]),
        Mock(choices=[Mock(delta=Mock(content=" world"))]),
    ])
    mock_inference_client.return_value = mock_client

    history = []
    message = "Hi there"
    generator = respond(message, history, use_local_model=False)
    
    result = list(generator)
    assert result[-1] == [(message, "Hello world")]

def test_respond_local_inference(mock_pipeline):
    mock_pipe = Mock()
    mock_pipe.return_value = [{'generated_text': [{'content': 'Hello'}, {'content': ' world'}]}]
    mock_pipeline.return_value = mock_pipe

    history = []
    message = "Hi there"
    generator = respond(message, history, use_local_model=True)
    
    result = list(generator)
    assert result[-1] == [(message, "Hello world")]

def test_cancel_inference():
    global stop_inference
    stop_inference = False
    cancel_inference()
    assert stop_inference == True

def test_clear_conversation():
    assert clear_conversation() is None

def test_update_chat_info():
    history = [
        ("Hello", "Hi there!"),
        ("How are you?", "I'm doing well, thank you for asking!")
    ]
    message_count, word_count = update_chat_info(history)
    assert message_count == 2
    assert word_count == 13  # Total words in the conversation

def test_update_chat_info_empty():
    message_count, word_count = update_chat_info(None)
    assert message_count == 0
    assert word_count == 0

@pytest.mark.parametrize("use_local_model", [True, False])
def test_respond_cancellation(use_local_model, mock_inference_client, mock_pipeline):
    global stop_inference
    stop_inference = False

    if use_local_model:
        mock_pipe = Mock()
        mock_pipe.return_value = [{'generated_text': [{'content': 'Hello'}, {'content': ' world'}]}]
        mock_pipeline.return_value = mock_pipe
    else:
        mock_client = Mock()
        mock_client.chat_completion.return_value = iter([
            Mock(choices=[Mock(delta=Mock(content="Hello"))]),
            Mock(choices=[Mock(delta=Mock(content=" world"))]),
        ])
        mock_inference_client.return_value = mock_client

    history = []
    message = "Hi there"
    generator = respond(message, history, use_local_model=use_local_model)
    
    # Simulate cancellation after first token
    next(generator)
    stop_inference = True
    
    result = list(generator)
    assert result[-1] == [(message, "Inference cancelled.")]