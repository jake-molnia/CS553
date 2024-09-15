import pytest
from unittest.mock import patch, MagicMock
from app.py import respond, cancel_inference, client, pipe

@pytest.fixture
def mock_inference_client():
    with patch('your_chatbot_file.InferenceClient') as mock:
        yield mock

@pytest.fixture
def mock_pipeline():
    with patch('your_chatbot_file.pipeline') as mock:
        yield mock

def test_respond_api(mock_inference_client):
    mock_client = MagicMock()
    mock_inference_client.return_value = mock_client
    mock_client.chat_completion.return_value = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))])
    ]

    response = list(respond("Hi", [], use_local_model=False))
    assert len(response) == 1
    assert response[0][0][0] == "Hi"
    assert response[0][0][1] == "Hello"

def test_respond_local(mock_pipeline):
    mock_pipe = MagicMock()
    mock_pipeline.return_value = mock_pipe
    mock_pipe.return_value = [{'generated_text': [{'content': 'Hello'}]}]

    response = list(respond("Hi", [], use_local_model=True))
    assert len(response) == 1
    assert response[0][0][0] == "Hi"
    assert response[0][0][1] == "Hello"

def test_cancel_inference():
    global stop_inference
    stop_inference = False
    cancel_inference()
    assert stop_inference == True

def test_respond_cancellation():
    global stop_inference
    stop_inference = True
    response = list(respond("Hi", [], use_local_model=False))
    assert len(response) == 1
    assert response[0][0][0] == "Hi"
    assert response[0][0][1] == "Inference cancelled."

@pytest.mark.parametrize("use_local_model", [True, False])
def test_respond_parameters(use_local_model, mock_inference_client, mock_pipeline):
    if use_local_model:
        mock_pipe = MagicMock()
        mock_pipeline.return_value = mock_pipe
        mock_pipe.return_value = [{'generated_text': [{'content': 'Hello'}]}]
    else:
        mock_client = MagicMock()
        mock_inference_client.return_value = mock_client
        mock_client.chat_completion.return_value = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))])
        ]

    response = list(respond(
        "Hi",
        [],
        system_message="Test system message",
        max_tokens=100,
        temperature=0.5,
        top_p=0.9,
        use_local_model=use_local_model
    ))

    assert len(response) == 1
    assert response[0][0][0] == "Hi"
    assert response[0][0][1] == "Hello"

    if use_local_model:
        mock_pipe.assert_called_once()
        call_args = mock_pipe.call_args[1]
        assert call_args['max_new_tokens'] == 100
        assert call_args['temperature'] == 0.5
        assert call_args['top_p'] == 0.9
    else:
        mock_client.chat_completion.assert_called_once()
        call_args = mock_client.chat_completion.call_args[1]
        assert call_args['max_tokens'] == 100
        assert call_args['temperature'] == 0.5
        assert call_args['top_p'] == 0.9

def test_gradio_components():
    import gradio as gr
    with gr.Blocks() as demo
        components = demo.component_dict
    
    assert 'chat_history' in components
    assert 'user_input' in components
    assert 'system_message' in components
    assert 'use_local_model' in components
    assert 'max_tokens' in components
    assert 'temperature' in components
    assert 'top_p' in components
