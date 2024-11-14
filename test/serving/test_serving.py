from unittest.mock import patch

import pytest

from hotel_reservations.serving.serving import Serving


@pytest.fixture
def mock_post():
    with patch("requests.post") as mock_post:
        yield mock_post


@pytest.fixture
def mock_time():
    with patch("time.time", side_effect=[1, 2]) as mock_time:
        yield mock_time


@pytest.fixture
def mock_random_choice():
    with patch("random.choice", return_value="random_id") as mock_random_choice:
        yield mock_random_choice


@pytest.fixture
def mock_send_request():
    with patch.object(Serving, "send_request", return_value=(200, "success", 1.5)) as mock_send_request:
        yield mock_send_request


@pytest.fixture
def serving_instance():
    # Setup the Serving instance within a fixture to ensure it doesn't initialize
    # in a way that requires Databricks authentication.
    with patch("hotel_reservations.serving.serving.Serving.__init__", return_value=None) as mock_init:  # type: ignore # noqa: F841
        serving_instance = Serving()
        serving_instance.serving_endpoint_name = "test_endpoint"
        serving_instance.num_requests = 5
        serving_instance.host = "test_host"
        serving_instance.token = "test_token"
        serving_instance.primary_key = "test_pk"
        yield serving_instance


def test_send_request(mock_post, mock_time, serving_instance):
    pk_value = "test_pk_value"
    mock_post.return_value.status_code = 200
    mock_post.return_value.text = "success"

    # Call the method
    response_status, response_text, latency = serving_instance.send_request(pk_value)

    # Verify the request details
    mock_post.assert_called_once_with(
        "https://test_host/serving-endpoints/test_endpoint/invocations",
        headers={"Authorization": "Bearer test_token"},
        json={"dataframe_records": [{"test_pk": pk_value}]},
    )

    # Check response and latency
    assert response_status == 200
    assert response_text == "success"
    assert latency == 1  # Since we mocked time.time to return 1 and 2, latency = 2 - 1 = 1


def test_send_request_random_id(mock_send_request, mock_random_choice, serving_instance):
    id_list = ["id1", "id2", "id3"]

    # Call the method
    response_status, response_text, latency = serving_instance.send_request_random_id(id_list)

    # Verify that random.choice was called and send_request was called with the selected id
    mock_random_choice.assert_called_once_with(id_list)
    mock_send_request.assert_called_once_with("random_id")

    # Check response and latency
    assert response_status == 200
    assert response_text == "success"
    assert latency == 1.5
