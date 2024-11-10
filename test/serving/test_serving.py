from unittest.mock import patch

import pytest

from hotel_reservations.serving.serving import Serving


@pytest.fixture
def serving_instance():
    return Serving(
        serving_endpoint_name="test_endpoint",
        num_requests=5,
        host="test_host",
        token="test_token",
        primary_key="test_pk",
    )


@patch("requests.post")  # Mock requests.post
@patch("time.time", side_effect=[1, 2])  # Mock time to control latency calculation
def test_send_request(mock_time, mock_post, serving_instance):
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


@patch("random.choice", return_value="random_id")  # Mock random.choice
@patch.object(Serving, "send_request", return_value=(200, "success", 1.5))  # Mock send_request method
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
